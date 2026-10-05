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
