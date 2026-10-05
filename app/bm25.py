"""Minimal BM25 (Okapi, Lucene-style non-negative IDF) over pre-tokenized documents."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.doc_len = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # term -> list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for idx, tokens in enumerate(docs_tokens):
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((idx, tf))
        self.idf = {
            term: math.log((self.n_docs - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
            for term, p in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avgdl == 0:
            return scores
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores
