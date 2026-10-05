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
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.cache import AnswerCache, fingerprint
from app.checks import check_answer
from app.config import Settings
from app.context import NEIGHBOUR_PARTS, ArticleStore, ContextArticle
from app.embeddings import make_embedder
from app.examples import ui_examples
from app.guard import CostGuard
from app.index import Index
from app.llm import SYSTEM_PROMPT, Answer, Generator, LLMError, extract_citations, is_no_answer, make_generator
from app.mcp_server import build_mcp_server, http_security
from app.netutil import client_ip, rate_limit_key
from app.pricing import Prices, estimate_max_tokens
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
        "pinned": hit.pinned,
    }


def _articles_json(context: list[ContextArticle], fragments: list[dict]) -> list[dict]:
    """What the model read, with the first retrieved fragment of each article for the page."""
    first_anchor: dict[str, str] = {}
    for fragment in fragments:
        first_anchor.setdefault(fragment["article"], fragment["anchor"])
    return [{**a.to_json(), "anchor": first_anchor.get(a.article)} for a in context]


def _answer_json(text: str, stop_reason: str | None, articles: list[dict], checks: dict | None) -> dict:
    """Answer text, its HTML, citations linked to the articles the model read, and the code checks."""
    links = {a["article"]: a for a in articles}
    checks = checks or {}
    return {
        "text": text,
        "html": render_answer_html(text, links, checks.get("missing_by_line")),
        "citations": [
            {
                "article": a,
                "anchor": links.get(a, {}).get("anchor"),
                "header": links.get(a, {}).get("header"),
                "source_url": links.get(a, {}).get("source_url"),
                "found": a in links,
            }
            for a in extract_citations(text)
        ],
        "checks": checks,
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
        mode=settings.search_mode,
        article_router=settings.article_router,
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
    generator = generator or make_generator(settings)
    prices = Prices(settings.price_rub_per_1m_input, settings.price_rub_per_1m_output)
    guard = guard or CostGuard(
        settings.guard_db,
        settings.daily_token_budget,
        settings.rate_limit_per_hour,
        hourly_token_budget=settings.hourly_token_budget,
        prices=prices,
    )
    cache = cache or AnswerCache(settings.guard_db)
    store = ArticleStore(retriever.index.chunks) if retriever is not None else None
    # MCP over Streamable HTTP: stateless JSON, so it works behind nginx without sticky sessions.
    mcp_server = mcp_http = None
    if store is not None and settings.mcp_enabled:
        mcp_server = build_mcp_server(retriever, store)
        mcp_http = mcp_server.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
            max_request_body_size=MAX_BODY_BYTES,
            transport_security=http_security(list(settings.mcp_allowed_hosts)),
        )

    # Everything that changes an answer besides the question and top_k: part of the cache key.
    index_meta = retriever.index.meta if retriever is not None else {}
    config_fingerprint = fingerprint(
        {
            "model": generator.model,
            "max_tokens": settings.max_tokens,
            "temperature": settings.llm_temperature,
            "reasoning_effort": settings.llm_reasoning_effort,
            "system_prompt": SYSTEM_PROMPT,
            "corpus_sha256": index_meta.get("corpus_sha256"),
            "n_chunks": index_meta.get("n_chunks"),
            "embedding_model": index_meta.get("embedding_model"),
            "chunk_max_chars": index_meta.get("chunk_max_chars"),
            "candidates": settings.candidates,
            "rrf_k": settings.rrf_k,
            "weights": [settings.bm25_weight, settings.dense_weight],
            "search_mode": settings.search_mode,
            "article_router": settings.article_router,
            "context": [settings.context_max_chars, settings.full_article_chars, NEIGHBOUR_PARTS],
            "answer_checks": 1,
        }
    )
    proxy_state = {"forwarded_for_ignored": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warm_cache and generator.mode == "live" and retriever is not None:
            thread = threading.Thread(target=warm_up_cache, name="cache-warm-up", daemon=True)
            app.state.warmup_thread = thread
            thread.start()
        async with AsyncExitStack() as stack:
            if mcp_server is not None:
                await stack.enter_async_context(mcp_server.session_manager.run())
            yield

    app = FastAPI(
        title="RAG по Трудовому кодексу РФ", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.guard = guard
    app.state.cache = cache
    app.state.store = store
    app.state.warmup_thread = None
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    if mcp_http is not None:
        app.router.routes.extend(mcp_http.routes)  # the /mcp endpoint; its session manager runs in lifespan
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    def _top_k(requested: int | None) -> int:
        return min(requested or settings.top_k, settings.max_top_k)

    def _estimate(question: str, top_k: int) -> int:
        # Upper bound before retrieval: the whole context budget (+ tags and a few […] marks per
        # article) + escaped question + system prompt + the full completion limit.
        prompt_chars = (
            len(SYSTEM_PROMPT) + len(html.escape(question)) + 100 + settings.context_max_chars + top_k * 300
        )
        return estimate_max_tokens(prompt_chars, settings.max_tokens)

    def answer_question(
        question: str, top_k: int, client_key: str | None, truncated: bool = False
    ) -> tuple[int, dict, dict | None]:
        """Non-streaming answer: (status, body, headers)."""
        for name, data in answer_events(question, top_k, client_key, truncated, stream=False):
            if name == "result":
                return data
        raise RuntimeError("answer_events ended without a result")

    def answer_events(question: str, top_k: int, client_key: str | None, truncated: bool = False, stream: bool = False):
        """Cost guard, cache, retrieval, context and generation as a sequence of events.

        Yields ("meta", {...}) once the articles are known, ("delta", text) while a streamed
        answer arrives, and always ends with ("result", (status, body, headers)).
        ``client_key`` None skips the per-client rate limit (cache warm-up at start).
        """
        started = time.perf_counter()
        base = {"question": question, "truncated": truncated, "max_question_chars": settings.max_question_chars}
        live = generator.mode == "live"
        cache_key = AnswerCache.make_key(question, top_k, config_fingerprint) if live else None
        cached = cache.get(cache_key) if cache_key else None
        paid = live and cached is None
        decision = guard.check_and_reserve(client_key, _estimate(question, top_k) if paid else 0, charge=paid)
        if not decision.allowed and decision.code == "rate_limited":
            headers = {"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None
            body = {"error": decision.code, "message": decision.message, "budget": guard.status()}
            yield "result", (429, body, headers)
            return

        if cached is not None:  # a repeated question: free, no model call, no reservation
            articles = cached.get("articles") or []
            yield "meta", {"articles": articles, "fragments": cached["fragments"], "cached": True}
            yield "result", (200, {
                **base,
                "mode": "live",
                "model": cached.get("model"),
                "cached": True,
                "cached_at": cached.get("created_at"),
                "answer": _answer_json(cached["text"], cached.get("stop_reason"), articles, cached.get("checks")),
                "articles": articles,
                "fragments": cached["fragments"],
                "usage": None,
                "tokens": 0,
                "cost_rub": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "budget": guard.status(),
            }, None)
            return

        hits = retriever.search(question, top_k=top_k)
        fragments = [_fragment_json(h) for h in hits]
        context = store.build_context(hits, settings.context_max_chars, settings.full_article_chars)
        articles = _articles_json(context, fragments)

        if not decision.allowed:  # budget or global hourly cap: polite refusal, fragments still shown
            headers = {"Retry-After": str(decision.retry_after_s)} if decision.retry_after_s else None
            body = {**base, "error": decision.code, "message": decision.message, "fragments": fragments,
                    "articles": articles, "budget": guard.status()}
            yield "result", (429, body, headers)
            return

        yield "meta", {"articles": articles, "fragments": fragments, "cached": False}
        answer: Answer | None = None
        try:
            if stream:
                for item in generator.stream(question, context):
                    if isinstance(item, str):
                        yield "delta", item
                    else:
                        answer = item
            else:
                answer = generator.generate(question, context)
        except LLMError as exc:
            # A rejected request (4xx) is not billed: release the reservation. On 5xx, a connection
            # error or a timeout the request may still have been processed, so the reservation stays.
            if not exc.billed and decision.reservation_id is not None:
                guard.release(decision.reservation_id)
            log.warning("generation failed: %s status=%s", type(exc).__name__, getattr(exc, "status_code", None))
            body = {
                **base,
                "error": "llm_unavailable",
                "message": "Модель сейчас недоступна, попробуйте позже. Найденные статьи показаны ниже.",
                "fragments": fragments,
                "articles": articles,
                "budget": guard.status(),
            }
            yield "result", (502, body, None)
            return
        # A client that disconnects mid-stream closes this generator before here: the
        # worst-case reservation then stays, which only overestimates the spend.
        if answer is None:
            raise RuntimeError("generator returned no answer")
        text, checks = check_answer(answer.text, context)
        checks_json = checks.to_json()
        usage = answer.usage
        # No usage in the response: the worst-case reservation stays instead of a zero.
        if decision.reservation_id is not None and usage is not None and usage.total > 0:
            guard.settle(decision.reservation_id, usage.input_tokens, usage.output_tokens)
        if cache_key and answer.complete:
            cache.put(
                cache_key,
                {
                    "text": text,
                    "stop_reason": answer.stop_reason,
                    "fragments": fragments,
                    "articles": articles,
                    "checks": checks_json,
                    "model": generator.model,
                    "tokens": usage.total if usage else 0,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "ask mode=%s stream=%s articles=%d context_chars=%d tokens=%d latency_ms=%d",
            answer.mode, stream, len(context), sum(len(a.text) for a in context),
            usage.total if usage else 0, latency_ms,
        )
        yield "result", (200, {
            **base,
            "mode": answer.mode,
            "model": generator.model,
            "cached": False,
            "answer": _answer_json(text, answer.stop_reason, articles, checks_json),
            "articles": articles,
            "fragments": fragments,
            "usage": None
            if usage is None
            else {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
            "tokens": usage.total if usage else 0,
            "cost_rub": None if usage is None else prices.cost_rub(usage.input_tokens, usage.output_tokens),
            "latency_ms": latency_ms,
            "budget": guard.status(),
        }, None)

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
            "search_mode": settings.search_mode,
            "article_router": settings.article_router,
            "cached_answers": len(cache),
            "mcp": "/mcp" if mcp_http is not None else None,
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

    def _prepare(payload: AskRequest, request: Request) -> tuple[str, bool, str] | JSONResponse:
        """Normalised question, truncation flag and rate-limit key, or the error response."""
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
        return question, truncated, rate_limit_key(ip)

    @app.post("/api/ask")
    def ask(payload: AskRequest, request: Request):
        prepared = _prepare(payload, request)
        if isinstance(prepared, JSONResponse):
            return prepared
        question, truncated, key = prepared
        status, body, headers = answer_question(question, _top_k(payload.top_k), key, truncated)
        if status == 200:
            return body
        return JSONResponse(status_code=status, content=body, headers=headers)

    @app.post("/api/ask/stream")
    def ask_stream(payload: AskRequest, request: Request):
        """Server-sent events: meta (articles), delta (answer text), then done or error (same body as /api/ask)."""
        prepared = _prepare(payload, request)
        if isinstance(prepared, JSONResponse):
            return prepared
        question, truncated, key = prepared

        def events():
            for name, data in answer_events(question, _top_k(payload.top_k), key, truncated, stream=True):
                if name == "delta":
                    data = {"t": data}
                elif name == "result":
                    status, body, _headers = data
                    name, data = ("done", body) if status == 200 else ("error", {**body, "status": status})
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        # X-Accel-Buffering: nginx would otherwise hold the chunks and send the answer at once.
        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

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
