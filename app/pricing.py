"""Token estimates for the cost guard and optional prices in rubles.

The budget is counted in tokens: the gateway's prices are not known to this
project, and a made-up price would make a made-up limit. If the operator knows
the price, PRICE_RUB_PER_1M_INPUT and PRICE_RUB_PER_1M_OUTPUT turn the token
counts into rubles for display; the limits stay in tokens either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Conservative pre-call estimate for Russian text: 2 characters per token.
# Live runs compare it with usage.prompt_tokens (evals report, "estimate check").
CHARS_PER_TOKEN_ESTIMATE = 2.0
# Chat formatting around the system and user messages.
PROMPT_OVERHEAD_TOKENS = 50


def estimate_input_tokens(prompt_chars: int) -> int:
    return math.ceil(prompt_chars / CHARS_PER_TOKEN_ESTIMATE) + PROMPT_OVERHEAD_TOKENS


def estimate_max_tokens(prompt_chars: int, max_completion_tokens: int) -> int:
    """Upper bound of one call: estimated input + the full completion limit (reasoning included)."""
    return estimate_input_tokens(prompt_chars) + max_completion_tokens


@dataclass(frozen=True)
class Prices:
    """Rubles per 1M tokens, set by the operator. None anywhere means "price unknown"."""

    input_rub_per_1m: float | None = None
    output_rub_per_1m: float | None = None

    @property
    def known(self) -> bool:
        return self.input_rub_per_1m is not None and self.output_rub_per_1m is not None

    def cost_rub(self, input_tokens: int, output_tokens: int) -> float | None:
        if not self.known:
            return None
        return (input_tokens * self.input_rub_per_1m + output_tokens * self.output_rub_per_1m) / 1_000_000
