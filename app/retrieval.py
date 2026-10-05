"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.chunking import Chunk
from app.embeddings import Embedder
from app.index import Index
from app.text import tokenize

MODES = ("bm25", "dense", "hybrid")


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Returns (item, score) sorted by score desc; ties keep first-seen order.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, candidates: int = 30, rrf_k: int = 60):
        if index.meta.get("embedding_model") != embedder.name:
            raise ValueError(
                f"index was built with {index.meta.get('embedding_model')!r}, "
                f"but the embedder is {embedder.name!r}; rebuild the index"
            )
        self.index = index
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _bm25_ranking(self, query: str) -> tuple[list[int], np.ndarray]:
        scores = self.index.bm25.scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[: self.candidates]
        return [int(i) for i in order if scores[i] > 0], scores

    def _dense_ranking(self, query: str) -> tuple[list[int], np.ndarray]:
        query_vec = self.embedder.embed_query(query)
        scores = self.index.embeddings @ query_vec
        order = np.argsort(-scores, kind="stable")[: self.candidates]
        return [int(i) for i in order], scores

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[SearchHit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        bm25_rank: list[int] = []
        dense_rank: list[int] = []
        bm25_scores = dense_scores = None
        if mode in ("bm25", "hybrid"):
            bm25_rank, bm25_scores = self._bm25_ranking(query)
        if mode in ("dense", "hybrid"):
            dense_rank, dense_scores = self._dense_ranking(query)
        rankings = [r for r in (bm25_rank, dense_rank) if r]
        fused = rrf_fuse(rankings, k=self.rrf_k)[:top_k]
        bm25_pos = {item: pos for pos, item in enumerate(bm25_rank, start=1)}
        dense_pos = {item: pos for pos, item in enumerate(dense_rank, start=1)}
        return [
            SearchHit(
                chunk=self.index.chunks[item],
                score=score,
                bm25_rank=bm25_pos.get(item),
                dense_rank=dense_pos.get(item),
                bm25_score=float(bm25_scores[item]) if bm25_scores is not None and item in bm25_pos else None,
                dense_score=float(dense_scores[item]) if dense_scores is not None else None,
            )
            for item, score in fused
        ]

    def rank_docs(self, query: str, mode: str = "hybrid", depth: int = 10) -> list[str]:
        """Distinct article ids in rank order (a doc counts at its best chunk)."""
        seen: list[str] = []
        for hit in self.search(query, top_k=self.candidates * 2, mode=mode):
            if hit.chunk.doc_id not in seen:
                seen.append(hit.chunk.doc_id)
            if len(seen) >= depth:
                break
        return seen
