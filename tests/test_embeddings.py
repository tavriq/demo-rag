import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from app import embeddings
from app.embeddings import E5_SMALL_QINT8, ModelFile, OnnxModelSpec, XlmrTokenizer, fetch_model_file, make_embedder

MODELS_DIR = Path(__file__).resolve().parents[1] / "var" / "models"


class FakeSp:
    def encode(self, text):
        return [0, 5, 10, 7]


def test_xlmr_ids_follow_fairseq_layout():
    tok = XlmrTokenizer.__new__(XlmrTokenizer)
    tok.sp = FakeSp()
    # <s>, sp-unk(0) -> 3, other ids shifted by +1, </s>
    assert tok.encode("x", 512) == [0, 3, 6, 11, 8, 2]
    assert tok.encode("x", 4) == [0, 3, 6, 2]


def _spec(payload: bytes) -> OnnxModelSpec:
    item = ModelFile("file.bin", hashlib.sha256(payload).hexdigest(), len(payload))
    return OnnxModelSpec(name="t", repo="org/model", revision="abc", onnx=item, spm=item, dim=4)


def test_fetch_downloads_and_verifies(tmp_path, monkeypatch):
    payload = b"model-bytes"
    spec = _spec(payload)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", fake_urlopen)
    path = fetch_model_file(spec, spec.onnx, tmp_path)
    assert path.read_bytes() == payload
    assert calls == ["https://huggingface.co/org/model/resolve/abc/file.bin"]
    fetch_model_file(spec, spec.onnx, tmp_path)  # cached: no second download
    assert len(calls) == 1


def test_fetch_rejects_checksum_mismatch(tmp_path, monkeypatch):
    spec = _spec(b"expected")
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_model_file(spec, spec.onnx, tmp_path)
    assert not list(tmp_path.rglob("file.bin*"))


def test_unknown_embedding_model_or_backend():
    with pytest.raises(ValueError, match="unknown EMBEDDING_MODEL"):
        make_embedder("onnx", "nope", "/tmp")
    with pytest.raises(ValueError, match="unknown EMBEDDING_BACKEND"):
        make_embedder("torch", "nope", "/tmp")


_model_cached = (MODELS_DIR / "intfloat--multilingual-e5-small" / E5_SMALL_QINT8.revision / E5_SMALL_QINT8.onnx.path).exists()


@pytest.mark.skipif(not _model_cached, reason="e5-small is not downloaded to var/models (python3 -m app.build_index)")
def test_real_model_prefers_relevant_passage():
    emb = make_embedder("onnx", E5_SMALL_QINT8.name, MODELS_DIR)
    passages = emb.embed_passages([
        "Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней.",
        "Продолжительность сверхурочной работы не должна превышать 120 часов в год.",
    ])
    query = emb.embed_query("сколько дней длится отпуск")
    assert passages.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(passages, axis=1), 1.0, rtol=1e-5)
    assert query @ passages[0] > query @ passages[1]
