import json
from datetime import datetime, timezone

import pytest

from app.evals import assign_splits, load_evals, render_markdown, run, select_split, write_report
from app.llm import ClaudeGenerator
from app.metrics import first_relevant_rank, hit_at_k, is_negative, mrr, percentile
from tests.conftest import EVALS


def test_rank_and_hit_metrics():
    assert first_relevant_rank(["a", "b", "c"], ["c", "x"]) == 3
    assert first_relevant_rank(["a"], ["z"]) is None
    ranks = [1, 2, None, 4]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.5
    assert hit_at_k(ranks, 5) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert mrr([11], depth=10) == 0
    assert hit_at_k([], 1) is None and mrr([]) is None


def test_percentile_nearest_rank():
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert percentile([5.0], 50) == 5.0
    assert percentile([], 95) is None


def test_negative_detection():
    assert is_negative({"type": "negative", "expected_ids": []})
    assert is_negative({"type": "direct", "expected_ids": []})
    assert not is_negative({"type": "direct", "expected_ids": ["TK-1"]})


NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _run(retriever, **kwargs):
    items = load_evals(EVALS)
    params = dict(evals_path=str(EVALS), top_k=5, candidates=30, rrf_k=60, answers=False, generator=None,
                  max_usd=0.5, now=NOW)
    params.update(kwargs)
    return items, run(retriever, items, **params)


def test_retrieval_evals_on_fixture_exclude_negatives(retriever):
    items, report = _run(retriever)
    n_neg = sum(1 for it in items if it["type"] == "negative")
    assert n_neg == 3
    assert report["evals"]["n_negative"] == n_neg
    for mode in ("bm25", "dense", "hybrid"):
        m = report["retrieval"][mode]
        assert m["n"] == len(items) - n_neg
        assert 0 <= m["hit@1"] <= m["hit@3"] <= m["hit@5"] <= 1
        assert 0 <= m["mrr@10"] <= 1
    assert set(report["retrieval"]["hybrid_by_type"]) == {"direct", "paraphrase"}
    # BM25 with stemming must find most direct questions on this small corpus
    assert report["retrieval"]["bm25"]["hit@5"] >= 0.7
    assert report["answers"]["status"] == "not_run"


def test_answers_without_key_are_marked_not_run(retriever):
    _, report = _run(retriever, answers=True, generator=None)
    assert report["answers"] == {"status": "not_run", "reason": "нужен ключ ANTHROPIC_API_KEY"}
    assert "Не прогонялось: нужен ключ" in render_markdown(report)


def test_answer_metrics_with_fake_model(retriever, fake_client_factory):
    fake = fake_client_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    _, report = _run(retriever, answers=True, generator=gen)
    a = report["answers"]
    assert a["status"] == "ok"
    assert a["n"] == 17 and a["n_negative"] == 3
    # only the one question about TK-115 is a citation hit
    assert a["citation_hit_rate"] == pytest.approx(1 / 14)
    assert a["negative_refusal_rate"] == 0
    assert a["avg_cost_usd"] == pytest.approx(0.0015)


def test_answer_run_stops_at_spend_cap(retriever, fake_client_factory):
    fake = fake_client_factory(text="В базе нет ответа на этот вопрос.", input_tokens=100_000, output_tokens=0)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    _, report = _run(retriever, answers=True, generator=gen, max_usd=0.25)
    assert report["answers"]["status"] == "partial"
    assert report["answers"]["n"] == 3  # $0.10 each, stops once >= $0.25


def test_report_files_written(retriever, tmp_path):
    _, report = _run(retriever)
    paths = write_report(report, tmp_path)
    names = sorted(p.name for p in paths)
    assert names == ["latest.json", "latest.md", "results-2026-10-05.json"]
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["date"] == "2026-10-05"
    md = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "| hybrid |" in md and "2026-10-05" in md


def test_split_is_stratified_deterministic_and_order_independent():
    items = load_evals(EVALS)
    splits = {it["q"]: it["split"] for it in items}
    assert set(splits.values()) == {"dev", "test"}
    for qtype in {it["type"] for it in items}:
        group = [it["split"] for it in items if it["type"] == qtype]
        assert abs(group.count("dev") - group.count("test")) <= 1
    reordered = list(reversed([dict(q=it["q"], type=it["type"], expected_ids=it["expected_ids"]) for it in items]))
    assign_splits(reordered)
    assert {it["q"]: it["split"] for it in reordered} == splits
    dev, test = select_split(items, "dev"), select_split(items, "test")
    assert len(dev) + len(test) == len(items) and not {i["q"] for i in dev} & {i["q"] for i in test}
    assert select_split(items, "all") == items


def test_report_has_hybrid_by_split(retriever):
    _, report = _run(retriever)
    by_split = report["retrieval"]["hybrid_by_split"]
    assert set(by_split) == {"dev", "test"}
    assert sum(m["n"] for m in by_split.values()) == report["retrieval"]["hybrid"]["n"]
    assert "Гибрид по частям набора" in render_markdown(report)
    assert all("split" in m for m in report["retrieval"]["misses"])
