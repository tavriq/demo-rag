import numpy as np
import pytest

from app.bm25 import BM25
from app.embeddings import HashEmbedder
from app.index import Index
from app.retrieval import Retriever, rrf_fuse
from app.text import tokenize


def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_between_lists():
    # item 7 is 2nd in both lists, items 1 and 9 are 1st in one list only
    order = [item for item, _ in rrf_fuse([[1, 7, 2], [9, 7, 3]], k=60)]
    assert order[0] == 7
    assert set(order[1:3]) == {1, 9}


def test_rrf_small_k_favours_top_ranks():
    # with tiny k, one first place beats two third places
    small = [item for item, _ in rrf_fuse([[1, 5, 2], [3, 6, 2]], k=1)]
    assert small[0] in (1, 3)
    large = [item for item, _ in rrf_fuse([[1, 5, 2], [3, 6, 2]], k=1000)]
    assert large[0] == 2


def test_rrf_empty_and_single_list():
    assert rrf_fuse([]) == []
    assert [i for i, _ in rrf_fuse([[4, 2, 9]])] == [4, 2, 9]


def test_bm25_prefers_matching_document():
    docs = [tokenize("ежегодный оплачиваемый отпуск 28 дней"), tokenize("сверхурочная работа 120 часов в год")]
    scores = BM25(docs).scores(tokenize("сколько дней отпуска"))
    assert scores[0] > 0 and scores[1] == 0


def test_hybrid_search_finds_expected_article(retriever):
    hits = retriever.search("Сколько дней длится ежегодный оплачиваемый отпуск?", top_k=3)
    assert hits[0].chunk.doc_id == "TK-115"
    assert hits[0].bm25_rank == 1
    assert hits[0].dense_rank is not None
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_search_modes_and_top_k(retriever):
    q = "сверхурочная работа сколько часов в год"
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.search(q, top_k=4, mode=mode)
        assert 1 <= len(hits) <= 4
    bm25_only = retriever.search(q, top_k=4, mode="bm25")
    assert all(h.dense_rank is None for h in bm25_only)
    with pytest.raises(ValueError):
        retriever.search(q, mode="magic")


def test_rank_docs_deduplicates_long_articles(retriever):
    docs = retriever.rank_docs("основания увольнения по инициативе работодателя прогул", depth=10)
    assert len(docs) == len(set(docs))
    assert "TK-81" in docs[:3]


def test_index_roundtrip_and_embedder_mismatch(index_dir):
    index = Index.load(index_dir)
    assert index.meta["n_docs"] == 15
    assert index.meta["n_chunks"] == len(index.chunks) > 15
    assert index.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, rtol=1e-5)
    other = HashEmbedder()
    other.name = "some-other-model"
    with pytest.raises(ValueError, match="rebuild"):
        Retriever(index, other)


def test_weighted_rrf():
    fused = dict(rrf_fuse([[1, 2], [2, 1]], k=60, weights=[2.0, 1.0]))
    assert fused[1] == pytest.approx(2 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(2 / 62 + 1 / 61)
    assert fused[1] > fused[2]
    with pytest.raises(ValueError):
        rrf_fuse([[1]], weights=[1.0, 2.0])


def test_retriever_weights_shift_hybrid_towards_one_side(index_dir):
    index = Index.load(index_dir)
    q = "Что считается удалёнкой по закону?"
    dense_top = Retriever(index, HashEmbedder()).search(q, top_k=1, mode="dense")[0].chunk.chunk_id
    heavy_dense = Retriever(index, HashEmbedder(), bm25_weight=0.0001, dense_weight=1.0)
    assert heavy_dense.search(q, top_k=1)[0].chunk.chunk_id == dense_top


def test_build_index_if_missing_rebuilds_on_version_change(tmp_path, monkeypatch, capsys):
    import json

    from app import build_index as cli
    from tests.conftest import CORPUS

    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    out = tmp_path / "idx"
    args = ["--corpus", str(CORPUS), "--out", str(out), "--backend", "hash", "--chunk-max-chars", "600"]
    assert cli.main(args) == 0
    capsys.readouterr()
    assert cli.main(args + ["--if-missing"]) == 0
    assert "up to date" in capsys.readouterr().out  # skipped
    meta = json.loads((out / "meta.json").read_text())
    meta["version"] = 1
    (out / "meta.json").write_text(json.dumps(meta))
    assert cli.main(args + ["--if-missing"]) == 0
    assert "index built" in capsys.readouterr().out
    assert json.loads((out / "meta.json").read_text())["version"] != 1


KNOWN = frozenset({"3", "80", "81", "84.1", "186.1", "312.1", "341.1", "341.1-1", "348.11-1"})


@pytest.mark.parametrize(
    "query, expected",
    [
        ("статья 186.1", ["186.1"]),
        ("ст. 84.1 ТК РФ", ["84.1"]),
        ("Ст.80", ["80"]),
        ("ст 81", ["81"]),
        ("что гласит статья 341.1-1", ["341.1-1"]),
        ("по статье 348.11-1", ["348.11-1"]),
        ("п. 6 ч. 1 ст. 81 — прогул", ["81"]),
        ("ст. 80, 81", ["80", "81"]),
        ("ст.ст. 80, 81", ["80", "81"]),
        ("статьи 80-81", ["80", "81"]),  # not an article itself: two articles
        ("ст. 80 и 3 дня", ["80"]),  # «и» does not continue the list
        ("статья 999", []),  # not in the corpus
        ("стаж 5 лет", []),
        ("пост 81", []),
        ("статья 81 и статья 80 и ст. 3 и ст. 312.1", ["81", "80", "3"]),  # at most three articles
        ("Сколько дней отпуск?", []),
    ],
)
def test_article_router_finds_numbers(query, expected):
    from app.router import find_article_refs

    assert find_article_refs(query, KNOWN) == expected


def test_router_pins_named_article_first_in_every_mode(index_dir):
    index = Index.load(index_dir)
    retriever = Retriever(index, HashEmbedder())
    article = "81"
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.search(f"что сказано в ст. {article}", top_k=5, mode=mode)
        assert hits[0].chunk.article == article and hits[0].pinned
        pinned = [h for h in hits if h.pinned]
        assert 1 <= len(pinned) <= 3 and all(h.chunk.article == article for h in pinned)
        assert retriever.rank_docs(f"ст. {article}", mode=mode)[0] == f"TK-{article}"


def test_router_can_be_switched_off(index_dir):
    retriever = Retriever(Index.load(index_dir), HashEmbedder(), article_router=False)
    assert not any(h.pinned for h in retriever.search("ст. 81", top_k=5))


def test_default_mode_comes_from_the_retriever(index_dir):
    index = Index.load(index_dir)
    q = "сверхурочная работа сколько часов в год"
    dense = Retriever(index, HashEmbedder(), mode="dense")
    assert [h.chunk.chunk_id for h in dense.search(q, top_k=3)] == [
        h.chunk.chunk_id for h in dense.search(q, top_k=3, mode="dense")
    ]
    assert all(h.bm25_rank is None for h in dense.search(q, top_k=3))
    with pytest.raises(ValueError):
        Retriever(index, HashEmbedder(), mode="magic")
