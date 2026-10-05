"""On-disk index: meta.json + chunks.json (with BM25 tokens) + embeddings.npy.

No vector database: for a corpus of a few thousand chunks a brute-force dot
product over a float32 matrix takes well under a millisecond.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.bm25 import BM25
from app.chunking import Chunk
from app.embeddings import Embedder
from app.text import tokenize

INDEX_VERSION = 1
META_FILE = "meta.json"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    out_dir: str | Path,
    *,
    corpus_path: str | Path,
    chunk_max_chars: int,
) -> dict:
    """Embed and tokenize chunks, write the index atomically, return its meta."""
    out_dir = Path(out_dir)
    started = time.perf_counter()
    embeddings = embedder.embed_passages([c.index_text for c in chunks])
    rows = []
    for chunk in chunks:
        row = chunk.to_dict()
        row["tokens"] = tokenize(chunk.index_text)
        rows.append(row)
    meta = {
        "version": INDEX_VERSION,
        "embedding_model": embedder.name,
        "dim": int(embeddings.shape[1]),
        "corpus_path": str(corpus_path),
        "corpus_sha256": file_sha256(corpus_path),
        "n_docs": len({c.doc_id for c in chunks}),
        "n_chunks": len(chunks),
        "chunk_max_chars": chunk_max_chars,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=".index-", dir=out_dir.parent))
    try:
        np.save(tmp_dir / EMBEDDINGS_FILE, embeddings.astype(np.float32))
        (tmp_dir / CHUNKS_FILE).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (tmp_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return meta


def read_meta(index_dir: str | Path) -> dict | None:
    path = Path(index_dir) / META_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Index:
    meta: dict
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25

    @classmethod
    def load(cls, index_dir: str | Path) -> "Index":
        index_dir = Path(index_dir)
        meta = read_meta(index_dir)
        if meta is None:
            raise FileNotFoundError(f"index not found in {index_dir} (run: python3 -m app.build_index)")
        if meta.get("version") != INDEX_VERSION:
            raise ValueError(f"index version {meta.get('version')} != {INDEX_VERSION}, rebuild it")
        rows = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        embeddings = np.load(index_dir / EMBEDDINGS_FILE)
        if embeddings.shape[0] != len(rows):
            raise ValueError("index is inconsistent: embeddings and chunks differ in length")
        chunks = [Chunk.from_dict(r) for r in rows]
        bm25 = BM25([r["tokens"] for r in rows])
        return cls(meta=meta, chunks=chunks, embeddings=embeddings, bm25=bm25)

    def article_to_doc_id(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for chunk in self.chunks:
            mapping.setdefault(chunk.article, chunk.doc_id)
        return mapping
