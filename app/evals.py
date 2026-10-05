"""Eval runner.

    python3 -m app.evals [--evals data/evals.jsonl] [--answers]

Retrieval metrics (hit@1/3/5, MRR@10) need no LLM and compare BM25, dense
and hybrid on the same questions; negative questions are excluded from them.
``--answers`` also asks the model (needs ANTHROPIC_API_KEY) and measures
citation accuracy, refusals on negative questions, cost and latency.
Writes evals/results-<date>.json, evals/latest.json and evals/latest.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from app.config import Settings
from app.embeddings import HashEmbedder, make_embedder
from app.index import Index
from app.llm import ClaudeGenerator, Generator
from app.metrics import first_relevant_rank, is_negative, percentile, rate, retrieval_summary
from app.retrieval import MODES, Retriever

RANK_DEPTH = 10


def load_evals(path: str | Path) -> list[dict]:
    items = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not str(row.get("q", "")).strip():
                raise ValueError(f"{path}:{lineno}: empty 'q'")
            row.setdefault("expected_ids", [])
            row.setdefault("type", "negative" if not row["expected_ids"] else "unspecified")
            items.append(row)
    return items


def run_retrieval(retriever: Retriever, items: list[dict]) -> dict:
    positives = [it for it in items if not is_negative(it)]
    result: dict = {}
    hybrid_rows = []
    for mode in MODES:
        ranks = []
        for item in positives:
            ranked = retriever.rank_docs(item["q"], mode=mode, depth=RANK_DEPTH)
            rank = first_relevant_rank(ranked, item["expected_ids"])
            ranks.append(rank)
            if mode == "hybrid":
                hybrid_rows.append((item, rank, ranked))
        result[mode] = retrieval_summary(ranks)

    by_type: dict[str, list] = {}
    for item, rank, _ in hybrid_rows:
        by_type.setdefault(item["type"], []).append(rank)
    result["hybrid_by_type"] = {t: retrieval_summary(r) for t, r in by_type.items()}
    result["misses"] = [
        {"q": item["q"], "type": item["type"], "expected_ids": item["expected_ids"], "top": ranked[:5]}
        for item, rank, ranked in hybrid_rows
        if rank is None or rank > 5
    ]
    return result


def run_answers(
    retriever: Retriever,
    generator: Generator,
    items: list[dict],
    top_k: int,
    max_usd: float,
) -> dict:
    article_to_doc = retriever.index.article_to_doc_id()
    rows = []
    errors: list[dict] = []
    total_cost = 0.0
    status, reason = "ok", None
    for item in items:
        if total_cost >= max_usd:
            status, reason = "partial", f"остановлено по лимиту ${max_usd:.2f} на прогон"
            break
        hits = retriever.search(item["q"], top_k=top_k)
        retrieved_ids = {h.chunk.doc_id for h in hits}
        try:
            answer = generator.generate(item["q"], hits)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            errors.append({"q": item["q"], "error": type(exc).__name__})
            continue
        total_cost += answer.cost_usd
        cited_ids = [article_to_doc.get(a, f"?{a}") for a in answer.citations]
        rows.append(
            {
                "q": item["q"],
                "type": item["type"],
                "expected_ids": item["expected_ids"],
                "negative": is_negative(item),
                "cited_ids": cited_ids,
                "citation_hit": bool(set(cited_ids) & set(item["expected_ids"])),
                "citations_valid": all(c in retrieved_ids for c in cited_ids),
                "no_answer": answer.no_answer,
                "cost_usd": answer.cost_usd,
                "latency_s": round(answer.latency_s, 3),
                "answer": answer.text,
            }
        )
    if errors and status == "ok":
        status, reason = "partial", f"ошибки API на {len(errors)} вопросах, они не вошли в метрики"
    pos = [r for r in rows if not r["negative"]]
    neg = [r for r in rows if r["negative"]]
    cited = [r for r in rows if r["cited_ids"]]
    latencies = [r["latency_s"] for r in rows]
    return {
        "status": status,
        "reason": reason,
        "model": generator.model,
        "n": len(rows),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "citation_hit_rate": rate([r["citation_hit"] for r in pos]),
        "citation_valid_rate": rate([r["citations_valid"] for r in cited]),
        "false_no_answer_rate": rate([r["no_answer"] for r in pos]),
        "negative_refusal_rate": rate([r["no_answer"] for r in neg]),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": total_cost / len(rows) if rows else None,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "p95_latency_s": percentile(latencies, 95),
        "errors": errors,
        "items": rows,
    }


def _fmt_pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def render_markdown(report: dict) -> str:
    lines = [
        f"# Evals — {report['date']}",
        "",
        f"Корпус: `{report['corpus']['path']}` — {report['corpus']['n_docs']} статей, "
        f"{report['corpus']['n_chunks']} чанков. Вопросы: `{report['evals']['path']}` — "
        f"{report['evals']['n_total']} (с ответом {report['evals']['n_positive']}, "
        f"negative {report['evals']['n_negative']}).",
        f"Эмбеддинги: `{report['config']['embedding_model']}`, top-k {report['config']['top_k']}, "
        f"RRF k={report['config']['rrf_k']}, веса BM25/dense {report['config'].get('bm25_weight', 1.0)}"
        f"/{report['config'].get('dense_weight', 1.0)}.",
        "",
        "## Поиск (без LLM, negative не учитываются)",
        "",
        "| Режим | n | hit@1 | hit@3 | hit@5 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        m = report["retrieval"][mode]
        mrr = "—" if m["mrr@10"] is None else f"{m['mrr@10']:.3f}"
        lines.append(
            f"| {mode} | {m['n']} | {_fmt_pct(m['hit@1'])} | {_fmt_pct(m['hit@3'])} | "
            f"{_fmt_pct(m['hit@5'])} | {mrr} |"
        )
    lines += ["", "## Ответы модели", ""]
    a = report["answers"]
    if a.get("status") in ("ok", "partial"):
        avg_cost = "—" if a["avg_cost_usd"] is None else f"${a['avg_cost_usd']:.5f}"
        avg_lat = "—" if a["avg_latency_s"] is None else f"{a['avg_latency_s']:.2f} с"
        lines += [
            f"- Модель: `{a['model']}`, вопросов: {a['n']}" + (f" (неполный: {a['reason']})" if a["reason"] else ""),
            f"- Цитата попадает в ожидаемую статью: {_fmt_pct(a['citation_hit_rate'])}",
            f"- Все цитаты ведут на найденные фрагменты: {_fmt_pct(a['citation_valid_rate'])}",
            f"- Ложный отказ на вопросах с ответом: {_fmt_pct(a['false_no_answer_rate'])}",
            f"- Корректный отказ на negative: {_fmt_pct(a['negative_refusal_rate'])}",
            f"- Средняя стоимость: {avg_cost}, всего ${a['total_cost_usd']:.4f}",
            f"- Средняя латентность: {avg_lat}",
        ]
    else:
        lines.append(f"Не прогонялось: {a.get('reason')}.")
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    paths = [out_dir / f"results-{report['date']}.json", out_dir / "latest.json", out_dir / "latest.md"]
    paths[0].write_text(payload, encoding="utf-8")
    paths[1].write_text(payload, encoding="utf-8")
    paths[2].write_text(render_markdown(report), encoding="utf-8")
    return paths


def run(
    retriever: Retriever,
    items: list[dict],
    *,
    evals_path: str,
    top_k: int,
    candidates: int,
    rrf_k: int,
    answers: bool,
    generator: Generator | None,
    max_usd: float,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now(timezone.utc)
    meta = retriever.index.meta
    negatives = [it for it in items if is_negative(it)]
    types: dict[str, int] = {}
    for it in items:
        types[it["type"]] = types.get(it["type"], 0) + 1
    report = {
        "date": now.date().isoformat(),
        "generated_at": now.isoformat(timespec="seconds"),
        "corpus": {
            "path": meta.get("corpus_path"),
            "sha256": meta.get("corpus_sha256"),
            "n_docs": meta.get("n_docs"),
            "n_chunks": meta.get("n_chunks"),
        },
        "evals": {
            "path": evals_path,
            "n_total": len(items),
            "n_positive": len(items) - len(negatives),
            "n_negative": len(negatives),
            "types": types,
        },
        "config": {
            "embedding_model": meta.get("embedding_model"),
            "chunk_max_chars": meta.get("chunk_max_chars"),
            "top_k": top_k,
            "candidates": candidates,
            "rrf_k": rrf_k,
            "bm25_weight": retriever.bm25_weight,
            "dense_weight": retriever.dense_weight,
        },
        "retrieval": run_retrieval(retriever, items),
    }
    if not answers:
        report["answers"] = {"status": "not_run", "reason": "не запрашивалось (запуск без --answers)"}
    elif generator is None:
        report["answers"] = {"status": "not_run", "reason": "нужен ключ ANTHROPIC_API_KEY"}
    else:
        report["answers"] = run_answers(retriever, generator, items, top_k, max_usd)
    return report


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Run retrieval (and optionally answer) evals.")
    parser.add_argument("--evals", default="data/evals.jsonl")
    parser.add_argument("--index", default=str(settings.index_dir))
    parser.add_argument("--out", default=str(settings.evals_dir))
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--answers", action="store_true", help="also evaluate model answers (needs API key)")
    parser.add_argument("--max-usd", type=float, default=0.50, help="spend cap for one --answers run")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args(argv)

    index = Index.load(args.index)
    backend = "hash" if index.meta.get("embedding_model") == HashEmbedder.name else "onnx"
    embedder = make_embedder(backend, index.meta["embedding_model"], settings.models_dir, settings.embed_threads)
    retriever = Retriever(
        index,
        embedder,
        candidates=settings.candidates,
        rrf_k=settings.rrf_k,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
    )
    items = load_evals(args.evals)[: args.limit]

    generator = None
    if args.answers:
        if settings.has_api_key:
            generator = ClaudeGenerator(model=settings.llm_model, max_tokens=settings.max_tokens)
        else:
            print("--answers: ANTHROPIC_API_KEY не задан, ответы не прогоняются", file=sys.stderr)

    report = run(
        retriever,
        items,
        evals_path=args.evals,
        top_k=args.top_k,
        candidates=settings.candidates,
        rrf_k=settings.rrf_k,
        answers=args.answers,
        generator=generator,
        max_usd=args.max_usd,
    )
    for path in write_report(report, args.out):
        print(f"written {path}")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
