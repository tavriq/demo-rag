"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.chunking import Chunk
from app.embeddings import Embedder
from app.index import Index
from app.text import tokenize

MODES = ("bm25", "dense", "hybrid")


def rrf_fuse(
    rankings: list[list[int]], k: int = 60, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of w / (k + rank), rank from 1.

    Weights default to 1 (classic RRF). Returns (item, score) sorted by score
    desc; ties keep first-seen order.
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking expected")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
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
    def __init__(
        self,
        index: Index,
        embedder: Embedder,
        candidates: int = 30,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        if index.meta.get("embedding_model") != embedder.name:
            raise ValueError(
                f"index was built with {index.meta.get('embedding_model')!r}, "
                f"but the embedder is {embedder.name!r}; rebuild the index"
            )
        self.index = index
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

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
        pairs = [(r, w) for r, w in ((bm25_rank, self.bm25_weight), (dense_rank, self.dense_weight)) if r]
        fused = rrf_fuse([r for r, _ in pairs], k=self.rrf_k, weights=[w for _, w in pairs])[:top_k]
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
