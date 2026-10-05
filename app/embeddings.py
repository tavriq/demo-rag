"""Dense embedders.

* ``OnnxE5Embedder`` — intfloat/multilingual-e5-small, int8-quantized ONNX,
  run with onnxruntime; tokenized with SentencePiece directly. The Hugging Face
  ``tokenizers`` JSON tokenizer for this 250k-token vocabulary alone took
  ~270–390 MB of RSS in our measurements, SentencePiece takes a few MB; that
  is what keeps the process under the 500 MB budget (see docs/design.md).
* ``HashEmbedder`` — deterministic hashing of stems and character trigrams.
  It needs no download and exists for tests and offline smoke runs; its
  retrieval quality is far below a real model and it is not used in the demo.

Model files are downloaded once from a pinned Hugging Face revision and
checked against their SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class OnnxModelSpec:
    name: str
    repo: str
    revision: str
    onnx: ModelFile
    spm: ModelFile
    dim: int
    max_tokens: int = 512
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


E5_SMALL_QINT8 = OnnxModelSpec(
    name="intfloat/multilingual-e5-small@qint8",
    repo="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    onnx=ModelFile(
        "onnx/model_qint8_avx512_vnni.onnx",
        "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88",
        118346824,
    ),
    spm=ModelFile(
        "sentencepiece.bpe.model",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
        5069051,
    ),
    dim=384,
)

ONNX_MODELS = {E5_SMALL_QINT8.name: E5_SMALL_QINT8}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_model_file(spec: OnnxModelSpec, item: ModelFile, cache_dir: str | Path) -> Path:
    """Return a local path to ``item``, downloading and verifying it if needed."""
    target = Path(cache_dir) / spec.repo.replace("/", "--") / spec.revision / item.path
    if target.exists() and target.stat().st_size == item.size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{item.path}"
    tmp = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:  # noqa: S310 (fixed https URL)
        shutil.copyfileobj(resp, out, length=1 << 20)
    actual = _sha256(tmp)
    if actual != item.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {item.path}: {actual} != {item.sha256}")
    tmp.replace(target)
    return target


class XlmrTokenizer:
    """XLM-RoBERTa ids from the SentencePiece model, same scheme as transformers' XLMRobertaTokenizer.

    fairseq layout: <s>=0, <pad>=1, </s>=2, <unk>=3, then SentencePiece ids shifted by +1
    (SentencePiece's own <unk> id 0 maps to 3).
    """

    BOS, PAD, EOS, UNK = 0, 1, 2, 3
    OFFSET = 1

    def __init__(self, model_file: str | Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))

    def encode(self, text: str, max_tokens: int) -> list[int]:
        ids = [i + self.OFFSET if i else self.UNK for i in self.sp.encode(text)]
        return [self.BOS] + ids[: max_tokens - 2] + [self.EOS]


class OnnxE5Embedder:
    """One text per session run: no padding, and since the int8 model quantizes
    activations dynamically, each embedding does not depend on batch neighbours.
    On the fixture this was as fast as batches of 16."""

    def __init__(self, spec: OnnxModelSpec, cache_dir: str | Path, threads: int = 2):
        import onnxruntime as ort

        self.spec = spec
        self.name = spec.name
        self.tokenizer = XlmrTokenizer(fetch_model_file(spec, spec.spm, cache_dir))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(fetch_model_file(spec, spec.onnx, cache_dir)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _embed_one(self, text: str) -> np.ndarray:
        ids = np.array([self.tokenizer.encode(text, self.spec.max_tokens)], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0][0]  # (tokens, dim)
        return hidden.mean(axis=0)  # E5 uses mean pooling

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._embed_one(self.spec.passage_prefix + t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._embed_one(self.spec.query_prefix + text)[None, :])[0]


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
    if backend == "onnx":
        try:
            spec = ONNX_MODELS[model_name]
        except KeyError as exc:
            raise ValueError(f"unknown EMBEDDING_MODEL {model_name!r}; known: {sorted(ONNX_MODELS)}") from exc
        return OnnxE5Embedder(spec, cache_dir=cache_dir, threads=threads)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBEDDING_BACKEND {backend!r} (expected 'onnx' or 'hash')")
