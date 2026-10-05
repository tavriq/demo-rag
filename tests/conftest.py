from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_corpus, load_corpus  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import HashEmbedder  # noqa: E402
from app.index import Index, build_index  # noqa: E402
from app.retrieval import Retriever  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
CORPUS = FIXTURES / "corpus_small.jsonl"
EVALS = FIXTURES / "evals_small.jsonl"
CHUNK_MAX = 600


@pytest.fixture(scope="session")
def articles():
    return load_corpus(CORPUS)


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory, articles):
    out = tmp_path_factory.mktemp("index") / "idx"
    build_index(chunk_corpus(articles, CHUNK_MAX), HashEmbedder(), out, corpus_path=CORPUS, chunk_max_chars=CHUNK_MAX)
    return out


@pytest.fixture(scope="session")
def retriever(index_dir):
    return Retriever(Index.load(index_dir), HashEmbedder(), candidates=30, rrf_k=60)


@pytest.fixture
def settings(tmp_path, index_dir):
    base = Settings.from_env({})
    return replace(
        base,
        var_dir=tmp_path,
        index_dir=index_dir,
        guard_db=tmp_path / "guard.sqlite",
        evals_dir=tmp_path / "evals",
        embedding_backend="hash",
        chunk_max_chars=CHUNK_MAX,
        has_api_key=False,
    )


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0):  # 2026-09-21 UTC, mid-day
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


class FakeMessages:
    """Stands in for client.messages: returns a canned response, records calls."""

    def __init__(self, text: str, input_tokens: int = 1000, output_tokens: int = 200, stop_reason: str = "end_turn"):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        )


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = FakeMessages(**kwargs)


@pytest.fixture
def fake_client_factory():
    return FakeAnthropic
