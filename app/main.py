"""FastAPI app. Run: uvicorn --factory app.main:create_app --no-proxy-headers

``--no-proxy-headers`` matters: client IP resolution (and X-Forwarded-For
trust) is handled by app.netutil according to TRUSTED_PROXY.
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.cache import AnswerCache, fingerprint
from app.config import Settings
from app.embeddings import make_embedder
from app.examples import ui_examples
from app.guard import CostGuard
from app.index import Index
from app.llm import SYSTEM_PROMPT, Generator, extract_citations, is_no_answer, make_generator
from app.netutil import client_ip, rate_limit_key
from app.pricing import estimate_max_cost
from app.render import anchor_for, render_answer_html, render_evals_page
from app.retrieval import Retriever, SearchHit

log = logging.getLogger("demo_rag")
STATIC_DIR = Path(__file__).parent / "static"
# Hard cap on the raw request field (same as the textarea maxlength); longer input is rejected,
# shorter is truncated to max_question_chars.
RAW_QUESTION_LIMIT = 2000
# Whole request body. 2000 Cyrillic characters as \uXXXX JSON escapes take 12 KB.
MAX_BODY_BYTES = 16 * 1024

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
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., max_length=RAW_QUESTION_LIMIT)
    top_k: int | None = Field(default=None, ge=1, le=20)


class BodySizeLimit:
    """Pure ASGI middleware: reject request bodies over ``max_bytes`` before they are buffered.

    A declared Content-Length over the limit gets 413 at once. A body without
    Content-Length (chunked) is cut off as soon as the running total passes the
    limit: the HTTPException raised from ``receive`` surfaces in the route while
    FastAPI reads the body and becomes a 413 response.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = self.max_bytes + 1
                if declared > self.max_bytes:
                    response = JSONResponse(
                        status_code=413, content={"error": "too_large", "message": "Запрос слишком большой."}
                    )
                    await response(scope, receive, send)
                    return
                break

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise HTTPException(status_code=413, detail="Запрос слишком большой.")
            return message

        await self.app(scope, limited_receive, send)


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


def _answer_json(text: str, stop_reason: str | None, fragments: list[dict]) -> dict:
    """Answer text + citations linked to the fragments it was generated from."""
    anchors: dict[str, str] = {}
    for fragment in fragments:
        anchors.setdefault(fragment["article"], fragment["anchor"])
    return {
        "text": text,
        "html": render_answer_html(text, anchors),
        "citations": [
            {"article": a, "anchor": anchors.get(a), "found": a in anchors} for a in extract_citations(text)
        ],
        "no_answer": is_no_answer(text),
        "stop_reason": stop_reason,
    }


def _load_retriever(settings: Settings) -> Retriever:
    index = Index.load(settings.index_dir)
    embedder = make_embedder(
        settings.embedding_backend, settings.embedding_model, settings.models_dir, settings.embed_threads
    )
    return Retriever(
        index,
        embedder,
        candidates=settings.candidates,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
    )


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    guard: CostGuard | None = None,
    cache: AnswerCache | None = None,
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
    guard = guard or CostGuard(
        settings.guard_db,
        settings.daily_budget_usd,
        settings.rate_limit_per_hour,
        global_paid_per_hour=settings.global_paid_per_hour,
    )
    cache = cache or AnswerCache(settings.guard_db)

    # Longest fragment as it is sent to the model (+ tag overhead): basis of the worst-case cost estimate.
    max_fragment_chars = settings.chunk_max_chars + 300
    if retriever is not None and retriever.index.chunks:
        max_fragment_chars = max(len(c.display_text) for c in retriever.index.chunks) + 150

    # Everything that changes an answer besides the question and top_k: part of the cache key.
    index_meta = retriever.index.meta if retriever is not None else {}
    config_fingerprint = fingerprint(
        {
            "model": generator.model,
            "max_tokens": settings.max_tokens,
            "system_prompt": SYSTEM_PROMPT,
            "corpus_sha256": index_meta.get("corpus_sha256"),
            "n_chunks": index_meta.get("n_chunks"),
            "embedding_model": index_meta.get("embedding_model"),
            "chunk_max_chars": index_meta.get("chunk_max_chars"),
            "candidates": settings.candidates,
            "rrf_k": settings.rrf_k,
            "weights": [settings.bm25_weight, settings.dense_weight],
        }
    )
    proxy_state = {"forwarded_for_ignored": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_cache and generator.mode == "live" and retriever is not None:
            thread = threading.Thread(target=warm_up_cache, name="cache-warm-up", daemon=True)
            app.state.warmup_thread = thread
            thread.start()
        yield

    app = FastAPI(
        title="RAG по Трудовому кодексу РФ", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.guard = guard
    app.state.cache = cache
    app.state.warmup_thread = None
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def _top_k(requested: int | None) -> int:
        return min(requested or settings.top_k, settings.max_top_k)

    def _estimate(question: str, top_k: int) -> float:
        # Upper bound before retrieval: top_k longest fragments + escaped question + system prompt.
        prompt_chars = len(SYSTEM_PROMPT) + len(html.escape(question)) + 100 + top_k * max_fragment_chars
        return estimate_max_cost(settings.llm_model, prompt_chars, settings.max_tokens)

    def answer_question(
        question: str, top_k: int, client_key: str | None, truncated: bool = False
    ) -> tuple[int, dict, dict | None]:
        """Cost guard, cache, retrieval and generation. Returns (status, body, headers).

        ``client_key`` None skips the per-client rate limit (cache warm-up at start).
        """
        started = time.perf_counter()
        base = {"question": question, "truncated": truncated, "max_question_chars": settings.max_question_chars}
        live = generator.mode == "live"
        cache_key = AnswerCache.make_key(question, top_k, config_fingerprint) if live else None
        cached = cache.get(cache_key) if cache_key else None
        paid = live and cached is None
        decision = guard.check_and_reserve(client_key, _estimate(question, top_k) if paid else 0.0, charge=paid)
        if not decision.allowed and decision.code == "rate_limited":
            headers = {"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None
            body = {"error": decision.code, "message": decision.message, "budget": guard.status()}
            return 429, body, headers

        if cached is not None:  # a repeated question: free, no model call, no reservation
            return 200, {
                **base,
                "mode": "live",
                "model": cached.get("model"),
                "cached": True,
                "cached_at": cached.get("created_at"),
                "answer": _answer_json(cached["text"], cached.get("stop_reason"), cached["fragments"]),
                "fragments": cached["fragments"],
                "usage": None,
                "cost_usd": 0.0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "budget": guard.status(),
            }, None

        hits = retriever.search(question, top_k=top_k)
        fragments = [_fragment_json(h) for h in hits]

        if not decision.allowed:  # budget or global hourly cap: polite refusal, fragments still shown
            headers = {"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None
            body = {**base, "error": decision.code, "message": decision.message, "fragments": fragments,
                    "budget": guard.status()}
            return 429, body, headers

        try:
            answer = generator.generate(question, hits)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            # An error response is not billed: release the reservation. On a connection error or
            # timeout the request may still have been processed, so the worst-case reservation stays.
            if isinstance(exc, anthropic.APIStatusError) and decision.reservation_id is not None:
                guard.release(decision.reservation_id)
            log.warning("generation failed: %s status=%s", type(exc).__name__, getattr(exc, "status_code", None))
            body = {
                **base,
                "error": "llm_unavailable",
                "message": "Модель сейчас недоступна, попробуйте позже. Найденные фрагменты показаны ниже.",
                "fragments": fragments,
                "budget": guard.status(),
            }
            return 502, body, None
        if decision.reservation_id is not None:
            guard.settle(
                decision.reservation_id,
                answer.cost_usd,
                answer.usage.input_tokens if answer.usage else None,
                answer.usage.output_tokens if answer.usage else None,
            )
        if cache_key and answer.stop_reason == "end_turn":
            cache.put(
                cache_key,
                {
                    "text": answer.text,
                    "stop_reason": answer.stop_reason,
                    "fragments": fragments,
                    "model": generator.model,
                    "cost_usd": answer.cost_usd,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "ask mode=%s hits=%d cost=%.6f latency_ms=%d", answer.mode, len(hits), answer.cost_usd, latency_ms
        )
        return 200, {
            **base,
            "mode": answer.mode,
            "model": generator.model,
            "cached": False,
            "answer": _answer_json(answer.text, answer.stop_reason, fragments),
            "fragments": fragments,
            "usage": None
            if answer.usage is None
            else {"input_tokens": answer.usage.input_tokens, "output_tokens": answer.usage.output_tokens},
            "cost_usd": round(answer.cost_usd, 6),
            "latency_ms": latency_ms,
            "budget": guard.status(),
        }, None

    def warm_up_cache() -> None:
        """Answer the example questions once, so clicks on them are free and survive budget exhaustion."""
        for example in ui_examples():
            question = " ".join(example.split())[: settings.max_question_chars]
            try:
                status, body, _ = answer_question(question, settings.top_k, None)
                log.info("cache warm-up: status=%s cached=%s", status, body.get("cached"))
            except Exception as exc:  # never take the app down because of the warm-up
                log.warning("cache warm-up failed: %s", type(exc).__name__)

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
            "cached_answers": len(cache),
            # True once a request came with X-Forwarded-For while TRUSTED_PROXY is empty: behind a
            # proxy that means every visitor shares one rate-limit bucket.
            "proxy_headers_ignored": proxy_state["forwarded_for_ignored"],
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
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for and not settings.trusted_proxies and not proxy_state["forwarded_for_ignored"]:
            proxy_state["forwarded_for_ignored"] = True
            log.warning(
                "X-Forwarded-For received but TRUSTED_PROXY is empty: header ignored; behind a reverse proxy "
                "all visitors would share one rate-limit bucket"
            )
        ip = client_ip(request.client.host if request.client else None, forwarded_for, settings.trusted_proxies)
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
        status, body, headers = answer_question(question, _top_k(payload.top_k), rate_limit_key(ip), truncated)
        if status == 200:
            return body
        return JSONResponse(status_code=status, content=body, headers=headers)

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
