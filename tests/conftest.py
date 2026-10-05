from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_corpus, load_corpus  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import HashEmbedder  # noqa: E402
from app.index import Index, build_index  # noqa: E402
from app.llm import ChatCompletionsGenerator  # noqa: E402
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


class FakeGateway:
    """An OpenAI-compatible /chat/completions endpoint behind httpx2.MockTransport.

    Returns a canned completion (or an error status), records request bodies.
    """

    def __init__(
        self,
        text: str | None,
        input_tokens: int = 1000,
        output_tokens: int = 200,
        finish_reason: str = "stop",
        status: int = 200,
        reasoning_tokens: int = 0,
        usage: bool = True,
    ):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.finish_reason = finish_reason
        self.status = status
        self.reasoning_tokens = reasoning_tokens
        self.usage = usage
        self.calls: list[dict] = []
        self.paths: list[str] = []
        self.headers: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(json.loads(request.content))
        self.paths.append(request.url.path)
        self.headers.append(dict(request.headers))
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "fail"}})
        body = {
            "id": "x",
            "object": "chat.completion",
            "model": self.calls[-1]["model"],
            "choices": [
                {"index": 0, "finish_reason": self.finish_reason,
                 "message": {"role": "assistant", "content": self.text}}
            ],
        }
        if self.usage:
            body["usage"] = {
                "prompt_tokens": self.input_tokens,
                "completion_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        return httpx2.Response(200, json=body)

    def client(self) -> httpx2.Client:
        return httpx2.Client(
            base_url="https://gateway.test/v1/",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx2.MockTransport(self.handler),
        )


@pytest.fixture
def fake_llm_factory():
    """fake_llm_factory(text=..., ...) -> (ChatCompletionsGenerator, FakeGateway)."""

    def _make(text="Ответ [ст. 115].", model="test/model", max_tokens=600, reasoning_effort=None, **kwargs):
        gateway = FakeGateway(text, **kwargs)
        gen = ChatCompletionsGenerator(
            model=model, max_tokens=max_tokens, client=gateway.client(), reasoning_effort=reasoning_effort
        )
        return gen, gateway

    return _make
