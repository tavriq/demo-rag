"""Model prices and cost accounting from API usage.

Prices are USD per 1M tokens from Anthropic's public price list
(checked 2026-10-05): Claude Haiku 4.5 — input $1, output $5,
5-minute cache write $1.25, cache read $0.10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float
    cache_read: float


PRICES_PER_MTOK: dict[str, Price] = {
    "claude-haiku-4-5-20251001": Price(input=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
}

# Conservative pre-call estimate for Russian text: ~2 characters per token.
# Real Cyrillic tokenization is usually denser, so the estimate over-reserves.
CHARS_PER_TOKEN_ESTIMATE = 2.0


def price_for(model: str) -> Price:
    try:
        return PRICES_PER_MTOK[model]
    except KeyError as exc:
        raise KeyError(f"no price configured for model {model!r}; add it to app/pricing.py") from exc


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    p = price_for(model)
    total = (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_creation_input_tokens * p.cache_write
        + cache_read_input_tokens * p.cache_read
    )
    return total / 1_000_000


def estimate_max_cost(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Upper-bound cost of one call: estimated input + the full max_tokens of output."""
    input_tokens = math.ceil(prompt_chars / CHARS_PER_TOKEN_ESTIMATE) + 50
    return cost_usd(model, input_tokens, max_tokens)
