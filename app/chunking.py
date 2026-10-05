"""Corpus loading and chunking.

One chunk = one article. Articles longer than ``max_chars`` are split on
paragraph boundaries (then sentences, then words as a last resort); every part
keeps the article header so a lone chunk still says which article it is from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_FIELDS = ("id", "article", "title", "text")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+")


@dataclass(frozen=True)
class Article:
    id: str
    article: str
    title: str
    chapter: str
    text: str
    source_url: str
    edition_date: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    article: str
    title: str
    chapter: str
    part: int
    parts_total: int
    header: str
    body: str
    source_url: str
    edition_date: str

    @property
    def display_text(self) -> str:
        return f"{self.header}\n{self.body}"

    @property
    def index_text(self) -> str:
        """Text fed to BM25 and the embedder: header + chapter + body."""
        parts = [self.header]
        if self.chapter:
            parts.append(self.chapter)
        parts.append(self.body)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def load_corpus(path: str | Path) -> list[Article]:
    path = Path(path)
    articles: list[Article] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            missing = [f for f in REQUIRED_FIELDS if not str(row.get(f, "")).strip()]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing fields {missing}")
            doc_id = str(row["id"]).strip()
            if doc_id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {doc_id}")
            seen.add(doc_id)
            articles.append(
                Article(
                    id=doc_id,
                    article=str(row["article"]).strip(),
                    title=str(row["title"]).strip(),
                    chapter=str(row.get("chapter", "") or "").strip(),
                    text=str(row["text"]).strip(),
                    source_url=str(row.get("source_url", "") or "").strip(),
                    edition_date=str(row.get("edition_date", "") or "").strip(),
                )
            )
    if not articles:
        raise ValueError(f"{path}: corpus is empty")
    return articles


def article_header(article: Article) -> str:
    title = article.title
    if title.lower().startswith("статья"):
        return title
    return f"Статья {article.article}. {title}"


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split on whitespace into pieces of at most max_chars (a single huge word is cut)."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _units(paragraph: str, max_chars: int) -> list[str]:
    """Break a paragraph into units no longer than max_chars."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    units: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_hard_split(sentence, max_chars))
    return units


def split_text(text: str, max_chars: int) -> list[str]:
    """Greedy packing of paragraphs into parts of at most max_chars characters."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        for unit in _units(paragraph, max_chars):
            # +1 for the newline (or space) that joins units
            added = len(unit) + (1 if current else 0)
            if current and current_len + added > max_chars:
                parts.append("\n".join(current))
                current, current_len = [], 0
                added = len(unit)
            current.append(unit)
            current_len += added
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_article(article: Article, max_chars: int) -> list[Chunk]:
    header = article_header(article)
    bodies = split_text(article.text, max_chars) or [article.text]
    total = len(bodies)
    return [
        Chunk(
            chunk_id=f"{article.id}#{i}",
            doc_id=article.id,
            article=article.article,
            title=article.title,
            chapter=article.chapter,
            part=i + 1,
            parts_total=total,
            header=header if total == 1 else f"{header} (часть {i + 1} из {total})",
            body=body,
            source_url=article.source_url,
            edition_date=article.edition_date,
        )
        for i, body in enumerate(bodies)
    ]


def chunk_corpus(articles: list[Article], max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, max_chars))
    return chunks
