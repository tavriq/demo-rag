"""Dense embedders.

* ``FastEmbedEmbedder`` — multilingual ONNX model via fastembed (CPU, no torch).
* ``HashEmbedder`` — deterministic hashing of stems and character trigrams.
  It needs no download and exists for tests and offline smoke runs; its
  retrieval quality is far below a real model and it is not used in the demo.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from app.text import normalize, tokenize


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | Path, threads: int = 2, batch_size: int = 16):
        from fastembed import TextEmbedding  # heavy import, keep it lazy

        self.name = model_name
        self.batch_size = batch_size
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)
        # E5-family models expect role prefixes; MiniLM/mpnet paraphrase models do not.
        self._is_e5 = "e5" in model_name.lower()

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts, batch_size=self.batch_size))
        return _l2_normalize(np.vstack(vectors))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_e5:
            text = f"query: {text}"
        return self._embed([text])[0]


class HashEmbedder:
    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            idx, sign = self._bucket("w:" + token)
            vec[idx] += 2.0 * sign
        norm = normalize(text)
        for i in range(len(norm) - 2):
            gram = norm[i : i + 3]
            if gram.strip():
                idx, sign = self._bucket("c:" + gram)
                vec[idx] += 0.2 * sign
        return vec

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vector(text)[None, :])[0]


def make_embedder(backend: str, model_name: str, cache_dir: str | Path, threads: int = 2) -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name, cache_dir=cache_dir, threads=threads)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBEDDING_BACKEND {backend!r} (expected 'fastembed' or 'hash')")
