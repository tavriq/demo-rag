"""Article-number router: «статья 186.1», «ст. 84.1», «ст. 341.1-1» → that article first.

A deterministic rule, not a tuned weight. Dense search confuses numbers with a
dot, and in RRF a chunk found only by BM25 can never outrank the dense
candidates (see evals/tuning.md), so a question that names an article gets that
article's fragments at the top of the list regardless of retrieval mode.

Article numbers in the Labour Code look like 81, 186.1 and 341.1-1. Only numbers
that exist in the corpus are taken. A hyphenated pair that is not an article
itself («ст. 80-81») is read as two articles. A list after one keyword
(«ст. 80, 81») gives each article in it; «и» does not continue a list, because
«ст. 80 и 3 дня» would otherwise pin article 3.
"""

from __future__ import annotations

import re

MAX_PINNED_ARTICLES = 3

_NUMBER = r"\d+(?:\.\d+)*(?:\s*[-–]\s*\d+(?:\.\d+)*)?"
_REF_RE = re.compile(
    r"\bст(?:\.|атья|атьи|атье|атью|атьей|атей|атьям|атьями|атьях)?\s*"
    rf"({_NUMBER}(?:\s*,\s*{_NUMBER})*)",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(_NUMBER)


def _candidates(token: str) -> list[str]:
    token = re.sub(r"\s+", "", token).replace("–", "-")
    if "-" not in token:
        return [token]
    return [token] + token.split("-", 1)


def find_article_refs(query: str, known_articles: set[str] | frozenset[str]) -> list[str]:
    """Article numbers named in the query that exist in the corpus, in order of mention."""
    found: list[str] = []
    for match in _REF_RE.finditer(query):
        for item in _ITEM_RE.findall(match.group(1)):
            options = _candidates(item)
            if options[0] in known_articles:
                picked = [options[0]]
            else:  # «80-81»: not an article itself, read as two articles
                picked = [o for o in options[1:] if o in known_articles]
            for article in picked:
                if article not in found:
                    found.append(article)
    return found[:MAX_PINNED_ARTICLES]
