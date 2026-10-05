"""Build the search index: python3 -m app.build_index --corpus data/corpus.jsonl"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

from app.chunking import chunk_corpus, load_corpus
from app.config import Settings
from app.embeddings import HashEmbedder, make_embedder
from app.index import build_index, file_sha256, read_meta


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Build BM25 + dense index from a JSONL corpus.")
    parser.add_argument("--corpus", default=str(settings.corpus_path), help="path to corpus.jsonl")
    parser.add_argument("--out", default=str(settings.index_dir), help="index directory")
    parser.add_argument("--backend", default=settings.embedding_backend, choices=["fastembed", "hash"])
    parser.add_argument("--model", default=settings.embedding_model, help="fastembed model name")
    parser.add_argument("--chunk-max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="skip if the index already matches this corpus and embedding model",
    )
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 2

    if args.if_missing:
        meta = read_meta(args.out)
        expected_model = HashEmbedder.name if args.backend == "hash" else args.model
        if (
            meta
            and meta.get("corpus_sha256") == file_sha256(corpus)
            and meta.get("embedding_model") == expected_model
            and meta.get("chunk_max_chars") == args.chunk_max_chars
        ):
            print(f"index is up to date: {args.out} ({meta['n_chunks']} chunks)")
            return 0

    started = time.perf_counter()
    articles = load_corpus(corpus)
    chunks = chunk_corpus(articles, args.chunk_max_chars)
    embedder = make_embedder(args.backend, args.model, settings.models_dir, settings.embed_threads)
    meta = build_index(chunks, embedder, args.out, corpus_path=corpus, chunk_max_chars=args.chunk_max_chars)
    print(
        f"index built: {meta['n_docs']} articles, {meta['n_chunks']} chunks, model={meta['embedding_model']}, "
        f"{time.perf_counter() - started:.1f}s, peak RSS {peak_rss_mb():.0f} MB -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
