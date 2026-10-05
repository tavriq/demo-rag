"""Runtime settings, read once from environment variables.

The API key is deliberately NOT stored here: only the fact that it is present.
The Anthropic SDK reads ANTHROPIC_API_KEY from the environment by itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_network
from pathlib import Path
from typing import Mapping

LLM_MODEL = "claude-haiku-4-5-20251001"
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
    candidates: int
    rrf_k: int
    bm25_weight: float
    dense_weight: float
    llm_model: str
    max_tokens: int
    max_question_chars: int
    daily_budget_usd: float
    rate_limit_per_hour: int
    global_paid_per_hour: int
    warm_cache: bool
    trusted_proxies: tuple[IPv4Network | IPv6Network, ...]
    has_api_key: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        var_dir = Path(_get(env, "VAR_DIR", "var"))
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
            top_k=_get_int(env, "TOP_K", 5, minimum=1),
            max_top_k=_get_int(env, "MAX_TOP_K", 8, minimum=1),
            candidates=_get_int(env, "RETRIEVAL_CANDIDATES", 30, minimum=1),
            rrf_k=_get_int(env, "RRF_K", 60, minimum=1),
            bm25_weight=_get_float(env, "BM25_WEIGHT", 1.0),
            dense_weight=_get_float(env, "DENSE_WEIGHT", 1.5),
            llm_model=_get(env, "LLM_MODEL", LLM_MODEL),
            max_tokens=_get_int(env, "MAX_TOKENS", 600, minimum=50),
            max_question_chars=_get_int(env, "MAX_QUESTION_CHARS", 500, minimum=20),
            daily_budget_usd=_get_float(env, "DAILY_BUDGET_USD", 1.00),
            rate_limit_per_hour=_get_int(env, "RATE_LIMIT_PER_HOUR", 10, minimum=1),
            global_paid_per_hour=_get_int(env, "GLOBAL_PAID_PER_HOUR", 20, minimum=1),
            warm_cache=_get(env, "WARM_CACHE", "1").lower() not in ("0", "false", "no"),
            trusted_proxies=parse_trusted_proxies(env.get("TRUSTED_PROXY", "")),
            has_api_key=bool(env.get("ANTHROPIC_API_KEY", "").strip()),
        )
