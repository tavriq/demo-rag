"""FastAPI app. Run: uvicorn --factory app.main:create_app --no-proxy-headers

``--no-proxy-headers`` matters: client IP resolution (and X-Forwarded-For
trust) is handled by app.netutil according to TRUSTED_PROXY.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.embeddings import make_embedder
from app.guard import CostGuard
from app.index import Index
from app.llm import SYSTEM_PROMPT, Answer, Generator, make_generator
from app.netutil import client_ip
from app.pricing import estimate_max_cost
from app.render import anchor_for, render_answer_html, render_evals_page
from app.retrieval import Retriever, SearchHit

log = logging.getLogger("demo_rag")
STATIC_DIR = Path(__file__).parent / "static"
# Hard cap on the raw request field; longer input is rejected, shorter is truncated to max_question_chars.
RAW_QUESTION_LIMIT = 5000

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


class AskRequest(BaseModel):
    question: str = Field(..., max_length=RAW_QUESTION_LIMIT)
    top_k: int | None = Field(default=None, ge=1, le=20)


def _fragment_json(hit: SearchHit) -> dict:
    c = hit.chunk
    return {
        "anchor": anchor_for(c.chunk_id),
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "article": c.article,
        "header": c.header,
        "chapter": c.chapter,
        "text": c.body,
        "part": c.part,
        "parts_total": c.parts_total,
        "source_url": c.source_url,
        "edition_date": c.edition_date,
        "score": round(hit.score, 5),
        "bm25_rank": hit.bm25_rank,
        "dense_rank": hit.dense_rank,
        "bm25_score": None if hit.bm25_score is None else round(hit.bm25_score, 3),
        "dense_score": None if hit.dense_score is None else round(hit.dense_score, 4),
    }


def _answer_json(answer: Answer, hits: list[SearchHit]) -> dict:
    anchors: dict[str, str] = {}
    for hit in hits:
        anchors.setdefault(hit.chunk.article, anchor_for(hit.chunk.chunk_id))
    return {
        "text": answer.text,
        "html": render_answer_html(answer.text, anchors),
        "citations": [{"article": a, "anchor": anchors.get(a), "found": a in anchors} for a in answer.citations],
        "no_answer": answer.no_answer,
        "stop_reason": answer.stop_reason,
    }


def _load_retriever(settings: Settings) -> Retriever:
    index = Index.load(settings.index_dir)
    embedder = make_embedder(
        settings.embedding_backend, settings.embedding_model, settings.models_dir, settings.embed_threads
    )
    return Retriever(index, embedder, candidates=settings.candidates, rrf_k=settings.rrf_k)


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    guard: CostGuard | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    index_error: str | None = None
    if retriever is None:
        try:
            retriever = _load_retriever(settings)
        except Exception as exc:  # the page must still open and explain what is missing
            index_error = f"{type(exc).__name__}: {exc}"
            log.error("index not loaded: %s", index_error)
    generator = generator or make_generator(settings.llm_model, settings.max_tokens, settings.has_api_key)
    guard = guard or CostGuard(settings.guard_db, settings.daily_budget_usd, settings.rate_limit_per_hour)

    app = FastAPI(title="RAG по Трудовому кодексу РФ", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.guard = guard
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def _top_k(requested: int | None) -> int:
        return min(requested or settings.top_k, settings.max_top_k)

    @app.get("/", include_in_schema=False)
    def index_page():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health():
        body = {
            "status": "ok" if retriever is not None else "degraded",
            "mode": generator.mode,
            "model": generator.model,
            "budget": guard.status(),
            "max_question_chars": settings.max_question_chars,
            "top_k": settings.top_k,
        }
        if retriever is not None:
            meta = retriever.index.meta
            body["index"] = {
                "n_docs": meta.get("n_docs"),
                "n_chunks": meta.get("n_chunks"),
                "embedding_model": meta.get("embedding_model"),
                "built_at": meta.get("built_at"),
            }
        else:
            body["index"] = None
            body["error"] = "Индекс не загружен: выполните python3 -m app.build_index"
        return body

    @app.post("/api/ask")
    def ask(payload: AskRequest, request: Request):
        started = time.perf_counter()
        ip = client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            settings.trusted_proxies,
        )
        question = " ".join(payload.question.split())
        if not question:
            return JSONResponse(status_code=400, content={"error": "empty", "message": "Введите вопрос."})
        truncated = len(question) > settings.max_question_chars
        if truncated:
            question = question[: settings.max_question_chars]
        if retriever is None:
            return JSONResponse(
                status_code=503,
                content={"error": "no_index", "message": "Индекс ещё не построен, попробуйте позже."},
            )
        top_k = _top_k(payload.top_k)

        live = generator.mode == "live"
        estimate = 0.0
        if live:
            # Upper bound before retrieval: top_k full-size chunks + question + system prompt.
            prompt_chars = len(SYSTEM_PROMPT) + len(question) + top_k * (settings.chunk_max_chars + 300)
            estimate = estimate_max_cost(settings.llm_model, prompt_chars, settings.max_tokens)
        decision = guard.check_and_reserve(ip, estimate, charge=live)
        if not decision.allowed and decision.code == "rate_limited":
            headers = {"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={"error": decision.code, "message": decision.message, "budget": guard.status()},
            )

        hits = retriever.search(question, top_k=top_k)
        fragments = [_fragment_json(h) for h in hits]

        if not decision.allowed:  # budget exhausted: polite refusal, fragments still shown
            return JSONResponse(
                status_code=429,
                content={
                    "error": decision.code,
                    "message": decision.message,
                    "question": question,
                    "truncated": truncated,
                    "fragments": fragments,
                    "budget": guard.status(),
                },
            )

        try:
            answer = generator.generate(question, hits)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            # An error response is not billed: release the reservation. On a connection error or
            # timeout the request may still have been processed, so the worst-case reservation stays.
            if isinstance(exc, anthropic.APIStatusError) and decision.reservation_id is not None:
                guard.settle(decision.reservation_id, 0.0)
            log.warning("generation failed: %s status=%s", type(exc).__name__, getattr(exc, "status_code", None))
            return JSONResponse(
                status_code=502,
                content={
                    "error": "llm_unavailable",
                    "message": "Модель сейчас недоступна, попробуйте позже. Найденные фрагменты показаны ниже.",
                    "question": question,
                    "truncated": truncated,
                    "fragments": fragments,
                    "budget": guard.status(),
                },
            )
        if decision.reservation_id is not None:
            guard.settle(
                decision.reservation_id,
                answer.cost_usd,
                answer.usage.input_tokens if answer.usage else None,
                answer.usage.output_tokens if answer.usage else None,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "ask mode=%s hits=%d cost=%.6f latency_ms=%d", answer.mode, len(hits), answer.cost_usd, latency_ms
        )
        return {
            "question": question,
            "truncated": truncated,
            "max_question_chars": settings.max_question_chars,
            "mode": answer.mode,
            "model": generator.model,
            "answer": _answer_json(answer, hits),
            "fragments": fragments,
            "usage": None
            if answer.usage is None
            else {"input_tokens": answer.usage.input_tokens, "output_tokens": answer.usage.output_tokens},
            "cost_usd": round(answer.cost_usd, 6),
            "latency_ms": latency_ms,
            "budget": guard.status(),
        }

    def _latest_evals() -> dict | None:
        path = settings.evals_dir / "latest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @app.get("/evals", response_class=HTMLResponse)
    def evals_page():
        return HTMLResponse(render_evals_page(_latest_evals()))

    @app.get("/api/evals")
    def evals_json():
        latest = _latest_evals()
        if latest is None:
            return JSONResponse(status_code=404, content={"error": "no_evals"})
        return latest

    return app

