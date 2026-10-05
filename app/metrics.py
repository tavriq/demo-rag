"""Retrieval and answer metrics. Pure functions, no I/O."""

from __future__ import annotations

import math


def first_relevant_rank(ranked_ids: list[str], expected_ids: list[str]) -> int | None:
    """1-based rank of the first expected id in the ranking, or None."""
    expected = set(expected_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in expected:
            return rank
    return None


def all_found(found_ids: list[str], expected_ids: list[str]) -> bool:
    """Every expected id is present (multi-article questions need all of them)."""
    return set(expected_ids) <= set(found_ids)


def hit_at_k(ranks: list[int | None], k: int) -> float | None:
    if not ranks:
        return None
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], depth: int = 10) -> float | None:
    if not ranks:
        return None
    return sum(1.0 / r for r in ranks if r is not None and r <= depth) / len(ranks)


def retrieval_summary(ranks: list[int | None]) -> dict:
    return {
        "n": len(ranks),
        "hit@1": hit_at_k(ranks, 1),
        "hit@3": hit_at_k(ranks, 3),
        "hit@5": hit_at_k(ranks, 5),
        "mrr@10": mrr(ranks, 10),
    }


def rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[idx]


def is_negative(item: dict) -> bool:
    return item.get("type") == "negative" or not item.get("expected_ids")
