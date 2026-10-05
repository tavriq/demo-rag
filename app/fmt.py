"""Number formatting for Russian reports: decimal comma, plural forms."""

from __future__ import annotations


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """plural_ru(534, "статья", "статьи", "статей") -> "статьи"."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def dec(value, digits: int = 3) -> str:
    """0.6734 -> "0,673"; None -> "—"."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value) -> str:
    """0.814 -> "81,4%"; None -> "—"."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%".replace(".", ",")


def num(value) -> str:
    """1.5 -> "1,5", 1.0 -> "1"."""
    if value is None:
        return "—"
    return f"{value:g}".replace(".", ",")
