"""What the model reads: retrieved fragments grouped into articles.

Search works on fragments (350 characters), but a fragment alone often misses the
exception or the deadline from the next part of the same article. So the model gets
articles instead: a short article in full, a long one as the retrieved parts plus
one neighbour on each side. Articles go in the order of their best fragment until
the character budget runs out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.chunking import Chunk
from app.retrieval import SearchHit

FULL_ARTICLE_CHARS = 4000  # median article is ~1 000 characters, p90 ~3 550
CONTEXT_MAX_CHARS = 12000
NEIGHBOUR_PARTS = 1
GAP = "[…]"


def base_header(chunk: Chunk) -> str:
    """Article header without the "(часть i из n)" suffix."""
    title = chunk.title
    if title.lower().startswith("статья"):
        return title
    return f"Статья {chunk.article}. {title}"


@dataclass(frozen=True)
class ContextArticle:
    article: str
    doc_id: str
    header: str
    chapter: str
    source_url: str
    edition_date: str
    text: str  # what the model reads
    full: bool  # the whole article, not a window of parts
    parts: tuple[int, ...]  # parts in the window, 1-based; empty when full
    parts_total: int

    @property
    def coverage(self) -> str:
        if self.full or self.parts_total == 1:
            return "целиком"
        return f"части {format_ranges(self.parts)} из {self.parts_total}"

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("text")
        data["parts"] = list(self.parts)
        data["coverage"] = self.coverage
        data["chars"] = len(self.text)
        return data


def format_ranges(parts: tuple[int, ...]) -> str:
    """(1, 2, 3, 7) -> "1–3, 7"."""
    out: list[str] = []
    start = prev = None
    for p in parts:
        if start is None:
            start = prev = p
        elif p == prev + 1:
            prev = p
        else:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = p
    if start is not None:
        out.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(out)


class ArticleStore:
    """Articles rebuilt from the index chunks: same text the search sees, no second source."""

    def __init__(self, chunks: list[Chunk]):
        by_article: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_article.setdefault(chunk.article, []).append(chunk)
        self._chunks = {a: sorted(cs, key=lambda c: c.part) for a, cs in by_article.items()}

    def __contains__(self, article: str) -> bool:
        return article in self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def chunks(self, article: str) -> list[Chunk]:
        return self._chunks.get(article, [])

    def full_text(self, article: str) -> str:
        return "\n".join(c.body for c in self.chunks(article))

    def article(self, article: str) -> ContextArticle | None:
        chunks = self.chunks(article)
        if not chunks:
            return None
        return self._make(chunks, self.full_text(article), full=True, parts=())

    def _make(self, chunks: list[Chunk], text: str, full: bool, parts: tuple[int, ...]) -> ContextArticle:
        first = chunks[0]
        return ContextArticle(
            article=first.article,
            doc_id=first.doc_id,
            header=base_header(first),
            chapter=first.chapter,
            source_url=first.source_url,
            edition_date=first.edition_date,
            text=text,
            full=full,
            parts=parts,
            parts_total=first.parts_total,
        )

    def window(self, article: str, hit_parts: set[int], max_chars: int | None = None) -> ContextArticle:
        """Retrieved parts plus NEIGHBOUR_PARTS on each side, in part order, gaps marked."""
        chunks = self.chunks(article)
        total = len(chunks)
        wanted: set[int] = set()
        for p in hit_parts:
            for q in range(p - NEIGHBOUR_PARTS, p + NEIGHBOUR_PARTS + 1):
                if 1 <= q <= total:
                    wanted.add(q)
        parts = tuple(sorted(wanted))
        pieces: list[str] = []
        prev = None
        used = 0
        kept: list[int] = []
        for p in parts:
            body = chunks[p - 1].body
            piece = body if prev is None or p == prev + 1 else f"{GAP}\n{body}"
            if p == parts[0] and p > 1:
                piece = f"{GAP}\n{piece}"
            if max_chars is not None and kept and used + len(piece) + 1 > max_chars:
                break
            pieces.append(piece)
            kept.append(p)
            used += len(piece) + 1
            prev = p
        if kept and kept[-1] < total:
            pieces.append(GAP)
        return self._make(chunks, "\n".join(pieces), full=False, parts=tuple(kept))

    def build_context(
        self,
        hits: list[SearchHit],
        max_chars: int = CONTEXT_MAX_CHARS,
        full_article_chars: int = FULL_ARTICLE_CHARS,
    ) -> list[ContextArticle]:
        """Group hits into articles in rank order and fit them into ``max_chars``.

        The first article always goes in (as a window cut to the budget if needed);
        a later one that does not fit even as a window is skipped, so a shorter
        article further down can still use the rest of the budget.
        """
        order: list[str] = []
        hit_parts: dict[str, set[int]] = {}
        for hit in hits:
            a = hit.chunk.article
            if a not in hit_parts:
                order.append(a)
                hit_parts[a] = set()
            hit_parts[a].add(hit.chunk.part)

        out: list[ContextArticle] = []
        used = 0
        for a in order:
            if a not in self:
                continue
            left = max_chars - used
            full = self.full_text(a)
            if len(self.chunks(a)) == 1 or len(full) <= full_article_chars:
                candidate = self._make(self.chunks(a), full, full=True, parts=())
                if len(candidate.text) > left:
                    candidate = self.window(a, hit_parts[a], max_chars=left if out else max_chars)
            else:
                candidate = self.window(a, hit_parts[a], max_chars=left if out else max_chars)
            if out and len(candidate.text) > left:
                continue
            out.append(candidate)
            used += len(candidate.text)
        return out
