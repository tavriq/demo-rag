"""Runtime settings, read once from environment variables.

The API key and the gateway URL are deliberately NOT stored here, only the fact
that both are present: app.llm.make_generator reads them from the environment,
so neither can end up in a repr, a log line or an error message.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_network
from pathlib import Path
from typing import Mapping

# Any OpenAI-compatible /chat/completions gateway; the model name is the gateway's.
DEFAULT_LLM_MODEL = "openai/gpt-5.6-terra"
REASONING_EFFORTS = ("minimal", "low", "medium", "high")
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small@qint8"


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip()
    return value if value else default


def _get_int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = _get(env, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(env: Mapping[str, str], name: str, default: float, minimum: float = 0.0) -> float:
    raw = _get(env, name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_optional_float(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name, "").strip()
    return None if not raw else _get_float(env, name, 0.0)


def parse_temperature(raw: str) -> float | None:
    """LLM_TEMPERATURE: a number, or "none" to leave it out (some models accept only the default)."""
    raw = raw.strip().lower()
    if raw in ("none", "omit", "off"):
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"LLM_TEMPERATURE must be a number or 'none', got {raw!r}") from exc


def parse_trusted_proxies(raw: str) -> tuple[IPv4Network | IPv6Network, ...]:
    """Parse TRUSTED_PROXY: comma-separated IPs or CIDR ranges. Empty = trust nobody."""
    networks = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            networks.append(ip_network(part, strict=False))
    return tuple(networks)


@dataclass(frozen=True)
class Settings:
    var_dir: Path
    corpus_path: Path
    index_dir: Path
    models_dir: Path
    guard_db: Path
    evals_dir: Path
    embedding_backend: str
    embedding_model: str
    embed_threads: int
    chunk_max_chars: int
    top_k: int
    max_top_k: int
    context_max_chars: int
    full_article_chars: int
    candidates: int
    rrf_k: int
    bm25_weight: float
    dense_weight: float
    search_mode: str
    article_router: bool
    llm_model: str
    llm_reasoning_effort: str | None
    llm_temperature: float | None
    llm_timeout_s: float
    max_tokens: int
    max_question_chars: int
    daily_token_budget: int
    hourly_token_budget: int
    price_rub_per_1m_input: float | None
    price_rub_per_1m_output: float | None
    rate_limit_per_hour: int
    warm_cache: bool
    trusted_proxies: tuple[IPv4Network | IPv6Network, ...]
    mcp_enabled: bool
    mcp_allowed_hosts: tuple[str, ...]
    has_api_key: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        var_dir = Path(_get(env, "VAR_DIR", "var"))
        # dense + article router chosen on the dev split, see evals/tuning.md (iteration 4)
        search_mode = _get(env, "SEARCH_MODE", "dense")
        if search_mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"SEARCH_MODE must be bm25, dense or hybrid, got {search_mode!r}")
        effort = env.get("LLM_REASONING_EFFORT", "").strip().lower() or None
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(f"LLM_REASONING_EFFORT must be one of {REASONING_EFFORTS}, got {effort!r}")
        return cls(
            var_dir=var_dir,
            corpus_path=Path(_get(env, "CORPUS_PATH", "data/corpus.jsonl")),
            index_dir=Path(_get(env, "INDEX_DIR", str(var_dir / "index"))),
            models_dir=Path(_get(env, "MODELS_DIR", str(var_dir / "models"))),
            guard_db=Path(_get(env, "GUARD_DB", str(var_dir / "guard.sqlite"))),
            evals_dir=Path(_get(env, "EVALS_DIR", "evals")),
            embedding_backend=_get(env, "EMBEDDING_BACKEND", "onnx"),
            embedding_model=_get(env, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embed_threads=_get_int(env, "EMBED_THREADS", 2, minimum=1),
            chunk_max_chars=_get_int(env, "CHUNK_MAX_CHARS", 350, minimum=200),
            top_k=_get_int(env, "TOP_K", 8, minimum=1),
            max_top_k=_get_int(env, "MAX_TOP_K", 8, minimum=1),
            # The model reads whole articles (short) or windows around the hits (long), app/context.py
            context_max_chars=_get_int(env, "CONTEXT_MAX_CHARS", 12000, minimum=1000),
            full_article_chars=_get_int(env, "FULL_ARTICLE_CHARS", 4000, minimum=500),
            candidates=_get_int(env, "RETRIEVAL_CANDIDATES", 30, minimum=1),
            rrf_k=_get_int(env, "RRF_K", 60, minimum=1),
            bm25_weight=_get_float(env, "BM25_WEIGHT", 1.0),
            dense_weight=_get_float(env, "DENSE_WEIGHT", 1.5),
            search_mode=search_mode,
            article_router=_get(env, "ARTICLE_ROUTER", "1").lower() not in ("0", "false", "no"),
            llm_model=_get(env, "LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_reasoning_effort=effort,
            llm_temperature=parse_temperature(_get(env, "LLM_TEMPERATURE", "0")),
            llm_timeout_s=_get_float(env, "LLM_TIMEOUT_S", 30.0, minimum=1.0),
            max_tokens=_get_int(env, "MAX_TOKENS", 1000, minimum=50),
            max_question_chars=_get_int(env, "MAX_QUESTION_CHARS", 500, minimum=20),
            daily_token_budget=_get_int(env, "DAILY_TOKEN_BUDGET", 300_000),
            hourly_token_budget=_get_int(env, "HOURLY_TOKEN_BUDGET", 100_000),
            price_rub_per_1m_input=_get_optional_float(env, "PRICE_RUB_PER_1M_INPUT"),
            price_rub_per_1m_output=_get_optional_float(env, "PRICE_RUB_PER_1M_OUTPUT"),
            rate_limit_per_hour=_get_int(env, "RATE_LIMIT_PER_HOUR", 10, minimum=1),
            warm_cache=_get(env, "WARM_CACHE", "1").lower() not in ("0", "false", "no"),
            trusted_proxies=parse_trusted_proxies(env.get("TRUSTED_PROXY", "")),
            # MCP over HTTP at /mcp (app/mcp_server.py); Host headers allowed for it, comma-separated
            mcp_enabled=_get(env, "MCP_HTTP", "1").lower() not in ("0", "false", "no"),
            mcp_allowed_hosts=tuple(
                h.strip() for h in _get(env, "MCP_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*,[::1]:*").split(",")
                if h.strip()
            ),
            # Live answers need both; with only one of them the app stays in mock mode.
            has_api_key=bool(env.get("LLM_API_KEY", "").strip()) and bool(env.get("LLM_BASE_URL", "").strip()),
        )
