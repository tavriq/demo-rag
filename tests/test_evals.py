import json
from datetime import datetime, timezone

import pytest

from app.evals import assign_splits, load_evals, render_markdown, run, select_split, write_report
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
    items = load_evals(EVALS)[: kwargs.get("limit")]
    params = dict(evals_path=str(EVALS), top_k=5, candidates=30, rrf_k=60, answers=False, generator=None,
                  max_run_tokens=150_000, now=NOW)
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
    assert report["retrieval"]["main_mode"] == "hybrid"
    assert set(report["retrieval"]["main_by_type"]) == {"direct", "paraphrase"}
    # BM25 with stemming must find most direct questions on this small corpus
    assert report["retrieval"]["bm25"]["hit@5"] >= 0.7
    assert report["answers"]["status"] == "not_run"


def test_answers_without_key_are_marked_not_run(retriever):
    _, report = _run(retriever, answers=True, generator=None)
    assert report["answers"] == {"status": "not_run", "reason": "нужны LLM_API_KEY и LLM_BASE_URL"}
    assert "Не прогонялось: нужны LLM_API_KEY" in render_markdown(report)


def test_answer_metrics_with_fake_model(retriever, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    _, report = _run(retriever, answers=True, generator=gen, gateway="gateway.test")
    a = report["answers"]
    assert a["status"] == "ok"
    assert a["n"] == 17 and a["n_negative"] == 3
    # only the one question about TK-115 is a citation hit
    assert a["citation_hit_rate"] == pytest.approx(1 / 14)
    assert a["negative_refusal_rate"] == 0
    assert a["uncited_answer_rate"] == 0 and a["incomplete_rate"] == 0
    assert a["avg_input_tokens"] == 1000 and a["avg_output_tokens"] == 100
    assert a["total_tokens"] == 17 * 1100 and a["total_cost_rub"] is None
    assert a["p50_latency_s"] is not None and a["p95_latency_s"] >= a["p50_latency_s"]
    assert 0 < a["input_estimate_ratio_max"] < 1  # the fake reports fewer tokens than the reserve
    assert a["gateway"] == "gateway.test" and a["model"] == "test/model"
    md = render_markdown(report)
    assert "p50" in md and "шлюз gateway.test" in md and "Ответы без единой ссылки" in md


def test_uncited_and_refusal_metrics(retriever, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Ответ без ссылок.")
    a = _run(retriever, answers=True, generator=gen)[1]["answers"]
    assert a["uncited_answer_rate"] == 1 and a["false_no_answer_rate"] == 0
    gen, _ = fake_llm_factory(text="В базе нет ответа на этот вопрос.")
    a = _run(retriever, answers=True, generator=gen)[1]["answers"]
    assert a["negative_refusal_rate"] == 1 and a["false_no_answer_rate"] == 1
    assert a["uncited_answer_rate"] is None  # every answer is a refusal


def test_answer_run_stops_at_token_cap(retriever, fake_llm_factory):
    gen, _ = fake_llm_factory(text="В базе нет ответа на этот вопрос.", input_tokens=10_000, output_tokens=0)
    _, report = _run(retriever, answers=True, generator=gen, max_run_tokens=25_000)
    assert report["answers"]["status"] == "partial"
    assert "25000 токенов" in report["answers"]["reason"]
    # 10k each: after two calls 20k are spent and the third still fits by its worst-case estimate
    # (about 1.7k on this fixture); after it the next one does not
    assert report["answers"]["n"] == 3


def test_report_files_written(retriever, tmp_path):
    _, report = _run(retriever)
    paths = write_report(report, tmp_path)
    names = sorted(p.name for p in paths)
    platform_slug = report["runtime"]["platform"].lower().replace(" ", "-").replace("_", "-")
    assert names == [f"2026-10-05T120000Z-all-{platform_slug}.json", "latest.json", "latest.md"]
    assert (tmp_path / "runs" / names[0]).exists()
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["date"] == "2026-10-05"
    md = (tmp_path / "latest.md").read_text(encoding="utf-8")
    assert "| hybrid (режим демо) |" in md and "| dense |" in md and "2026-10-05" in md


def test_partial_runs_do_not_overwrite_latest(retriever, tmp_path):
    _, full = _run(retriever)
    write_report(full, tmp_path)
    before = (tmp_path / "latest.json").read_text(encoding="utf-8")
    items = select_split(load_evals(EVALS), "dev")
    dev = run(retriever, items, evals_path=str(EVALS), top_k=5, candidates=30, rrf_k=60, answers=False,
              generator=None, max_run_tokens=150_000, now=datetime(2026, 10, 5, 13, 0, tzinfo=timezone.utc),
              split="dev")
    paths = write_report(dev, tmp_path)
    assert [p.name for p in paths] == [paths[0].name] and "-dev-" in paths[0].name
    _, limited = _run(retriever, limit=3, now=datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc))
    assert "limit3" in write_report(limited, tmp_path)[0].name
    assert (tmp_path / "latest.json").read_text(encoding="utf-8") == before
    assert len(list((tmp_path / "runs").glob("*.json"))) == 3


def test_context_and_multi_article_metrics(retriever):
    _, report = _run(retriever)
    r = report["retrieval"]
    assert r["context"]["top_k"] == 5 and r["context"]["n"] == r["hybrid"]["n"]
    assert 1 <= r["context"]["distinct_articles_min"] <= r["context"]["distinct_articles_avg"] <= 5
    # fragments sent to the model can only cover fewer articles than the top-5 articles
    assert r["context"]["hit"] <= r["hybrid"]["hit@5"]
    assert len(r["items"]) == r["hybrid"]["n"]
    assert set(r["items"][0]["rank"]) == {"bm25", "dense", "hybrid"}
    md = render_markdown(report)
    assert "фрагментов, которые уходят в модель" in md


def test_all_found_for_multi_article_questions():
    from app.metrics import all_found

    assert all_found(["TK-80", "TK-81"], ["TK-81", "TK-80"])
    assert not all_found(["TK-80", "TK-3"], ["TK-81", "TK-80"])


def test_spot_checks_are_reported(retriever):
    spots = [{"q": "ежегодный оплачиваемый отпуск", "expected_id": "TK-115", "type": "ui_example"}]
    _, report = _run(retriever, spot_checks=spots)
    row = report["spot_checks"][0]
    assert row["expected_id"] == "TK-115" and set(row["rank"]) == {"bm25", "dense", "hybrid"}
    assert row["ok"] == (row["rank"]["hybrid"] == 1)
    assert "Точечные проверки" in render_markdown(report)


def test_answer_run_checks_worst_case_before_the_call(retriever, fake_llm_factory):
    gen, gateway = fake_llm_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    _, report = _run(retriever, answers=True, generator=gen, max_run_tokens=1000)  # below one worst case
    assert report["answers"]["status"] == "partial" and report["answers"]["n"] == 0
    assert gateway.calls == []


def test_answer_run_records_gateway_errors(retriever, fake_llm_factory):
    gen, _ = fake_llm_factory(status=503)
    a = _run(retriever, answers=True, generator=gen, limit=2)[1]["answers"]
    assert a["status"] == "partial" and a["n"] == 0
    assert a["errors"][0] == {"q": a["errors"][0]["q"], "error": "LLMStatusError", "status": 503}


def test_answer_run_file_name_has_model_and_no_latest_flag(retriever, fake_llm_factory, tmp_path):
    gen, _ = fake_llm_factory(model="openai/gpt-5.6-terra")
    _, report = _run(retriever, answers=True, generator=gen)
    paths = write_report(report, tmp_path, update_latest=False)
    assert len(paths) == 1 and "-answers-openai-gpt-5-6-terra-" in paths[0].name
    assert not (tmp_path / "latest.json").exists()


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


def test_report_has_main_mode_by_split(retriever):
    _, report = _run(retriever)
    by_split = report["retrieval"]["main_by_split"]
    assert set(by_split) == {"dev", "test"}
    assert sum(m["n"] for m in by_split.values()) == report["retrieval"]["hybrid"]["n"]
    assert "Режим демо (гибрид) по частям набора" in render_markdown(report)


def test_breakdowns_follow_the_demo_mode(index_dir):
    from app.embeddings import HashEmbedder
    from app.index import Index
    from app.retrieval import Retriever

    dense = Retriever(Index.load(index_dir), HashEmbedder(), mode="dense")
    items = load_evals(EVALS)
    report = run(dense, items, evals_path=str(EVALS), top_k=5, candidates=30, rrf_k=60, answers=False,
                 generator=None, max_run_tokens=1, now=NOW)
    r = report["retrieval"]
    assert r["main_mode"] == "dense" and report["config"]["search_mode"] == "dense"
    ranks = [row["rank"]["dense"] for row in r["items"]]
    assert r["main_by_split"]["dev"]["n"] + r["main_by_split"]["test"]["n"] == len(ranks)
    assert len(r["misses"]) == sum(1 for x in ranks if x is None or x > 5)
    md = render_markdown(report)
    assert "| dense (режим демо) |" in md and "Режим поиска демо: dense" in md
    assert all("split" in m for m in report["retrieval"]["misses"])


def test_report_records_runtime(retriever):
    _, report = _run(retriever)
    rt = report["runtime"]
    assert rt["platform"] and rt["python"] and rt["onnxruntime"]
    assert f"Платформа: {rt['platform']}" in render_markdown(report)
