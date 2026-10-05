"""Eval runner.

    python3 -m app.evals [--evals data/evals.jsonl] [--answers]

Retrieval metrics (hit@1/3/5, MRR@10) need no LLM and compare BM25, dense
and hybrid on the same questions; negative questions are excluded from them.
Two stricter numbers show what the model actually gets: hit among the TOP_K
fragments sent to it, and for multi_hop questions whether both articles are found.
``--answers`` also asks the model (needs ANTHROPIC_API_KEY) and measures
citation accuracy, refusals on negative questions, cost and latency.

Every run is saved as evals/runs/<UTC time>-<split>-<platform>.json with
per-question ranks. evals/latest.json and evals/latest.md (shown on /evals and
quoted in README) are rewritten only by a full run: split "all", no --limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from app.config import Settings
from app.embeddings import HashEmbedder, make_embedder
from app.examples import SPOT_CHECKS_PATH, load_spot_checks
from app.fmt import dec, num, pct, plural_ru
from app.index import Index
from app.llm import SYSTEM_PROMPT, ClaudeGenerator, Generator, build_user_message
from app.metrics import all_found, first_relevant_rank, is_negative, percentile, rate, retrieval_summary
from app.pricing import estimate_max_cost
from app.retrieval import MODES, Retriever

RANK_DEPTH = 10
SPLITS = ("all", "dev", "test")


def assign_splits(items: list[dict]) -> None:
    """Deterministic stratified dev/test split, stored in item["split"].

    Within each question type items are ordered by sha256 of the question text;
    even positions go to dev, odd to test. Retriever settings are tuned on dev
    only, test stays held out (see evals/tuning.md). The split does not depend on
    the order of lines in the file.
    """
    by_type: dict[str, list[dict]] = {}
    for item in items:
        by_type.setdefault(item["type"], []).append(item)
    for group in by_type.values():
        group.sort(key=lambda it: hashlib.sha256(it["q"].encode("utf-8")).hexdigest())
        for pos, item in enumerate(group):
            item["split"] = "dev" if pos % 2 == 0 else "test"


def runtime_info() -> dict:
    """Where the run happened: int8 embeddings differ slightly between CPUs (see evals/tuning.md)."""
    try:
        import onnxruntime

        ort_version = onnxruntime.__version__
    except ImportError:  # pragma: no cover - onnxruntime is a hard dependency
        ort_version = None
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "onnxruntime": ort_version,
    }


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
    assign_splits(items)
    return items


def select_split(items: list[dict], split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return items if split == "all" else [it for it in items if it.get("split") == split]


def run_retrieval(retriever: Retriever, items: list[dict], top_k: int = 5) -> dict:
    positives = [it for it in items if not is_negative(it)]
    result: dict = {}
    per_item: list[dict] = []
    for item in positives:
        row = {
            "q": item["q"],
            "type": item["type"],
            "split": item.get("split"),
            "expected_ids": item["expected_ids"],
            "rank": {},
        }
        for mode in MODES:
            ranked = retriever.rank_docs(item["q"], mode=mode, depth=RANK_DEPTH)
            row["rank"][mode] = first_relevant_rank(ranked, item["expected_ids"])
            if mode == "hybrid":
                row["hybrid_top10"] = ranked
        # What the model actually receives: TOP_K fragments, several may come from one article.
        context_docs: list[str] = []
        for hit in retriever.search(item["q"], top_k=top_k):
            if hit.chunk.doc_id not in context_docs:
                context_docs.append(hit.chunk.doc_id)
        row["context_docs"] = context_docs
        per_item.append(row)

    for mode in MODES:
        result[mode] = retrieval_summary([row["rank"][mode] for row in per_item])

    by_type: dict[str, list] = {}
    by_split: dict[str, list] = {}
    for row in per_item:
        by_type.setdefault(row["type"], []).append(row["rank"]["hybrid"])
        if row.get("split"):
            by_split.setdefault(row["split"], []).append(row["rank"]["hybrid"])
    result["hybrid_by_type"] = {t: retrieval_summary(r) for t, r in by_type.items()}
    result["hybrid_by_split"] = {s: retrieval_summary(by_split[s]) for s in sorted(by_split)}

    distinct = [len(row["context_docs"]) for row in per_item]
    result["context"] = {
        "top_k": top_k,
        "n": len(per_item),
        "hit": rate([bool(set(row["context_docs"]) & set(row["expected_ids"])) for row in per_item]),
        "distinct_articles_avg": sum(distinct) / len(distinct) if distinct else None,
        "distinct_articles_min": min(distinct) if distinct else None,
    }
    multi = [row for row in per_item if len(row["expected_ids"]) > 1]
    result["multi_article"] = {
        "n": len(multi),
        "any@5": rate([row["rank"]["hybrid"] is not None and row["rank"]["hybrid"] <= 5 for row in multi]),
        "all@5": rate([all_found(row["hybrid_top10"][:5], row["expected_ids"]) for row in multi]),
        "all_in_context": rate([all_found(row["context_docs"], row["expected_ids"]) for row in multi]),
    }
    result["misses"] = [
        {
            "q": row["q"],
            "type": row["type"],
            "split": row["split"],
            "expected_ids": row["expected_ids"],
            "top": row["hybrid_top10"][:5],
        }
        for row in per_item
        if row["rank"]["hybrid"] is None or row["rank"]["hybrid"] > 5
    ]
    result["items"] = per_item
    return result


def run_spot_checks(retriever: Retriever, rows: list[dict]) -> list[dict]:
    """Single queries outside the eval set: page examples and "статья N" lookups."""
    out = []
    for row in rows:
        ranks = {}
        for mode in MODES:
            ranked = retriever.rank_docs(row["q"], mode=mode, depth=RANK_DEPTH)
            ranks[mode] = first_relevant_rank(ranked, [row["expected_id"]])
        top_fragment = retriever.search(row["q"], top_k=1)
        out.append(
            {
                "q": row["q"],
                "type": row["type"],
                "expected_id": row["expected_id"],
                "rank": ranks,
                "top_fragment_doc": top_fragment[0].chunk.doc_id if top_fragment else None,
                "ok": ranks["hybrid"] == 1,
            }
        )
    return out


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
    max_tokens = getattr(generator, "max_tokens", 0)
    for item in items:
        hits = retriever.search(item["q"], top_k=top_k)
        # Stop before a call whose worst case would cross the cap, not after the cap is crossed.
        prompt_chars = len(SYSTEM_PROMPT) + len(build_user_message(item["q"], hits))
        if total_cost + estimate_max_cost(generator.model, prompt_chars, max_tokens) > max_usd:
            status, reason = "partial", f"остановлено по лимиту ${max_usd:.2f} на прогон"
            break
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
                "split": item.get("split"),
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


def _runtime_note(report: dict) -> str:
    rt = report.get("runtime") or {}
    if not rt:
        return ""
    return f" Платформа: {rt.get('platform')}, Python {rt.get('python')}, onnxruntime {rt.get('onnxruntime')}."


def _split_note(report: dict) -> str:
    split = report["evals"].get("split", "all")
    note = "" if split == "all" else f", только часть `{split}`"
    if report["evals"].get("limit"):
        note += f", только первые {report['evals']['limit']}"
    return note


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['n']} | {pct(m['hit@1'])} | {pct(m['hit@3'])} | {pct(m['hit@5'])} | {dec(m['mrr@10'])} |"
    )


def render_markdown(report: dict) -> str:
    corpus, evals, config = report["corpus"], report["evals"], report["config"]
    n_docs = corpus["n_docs"] or 0
    lines = [
        f"# Evals — {report['date']}",
        "",
        f"Корпус: `{corpus['path']}` — {n_docs} {plural_ru(n_docs, 'статья', 'статьи', 'статей')}, "
        f"{corpus['n_chunks']} фрагментов. Вопросы: `{evals['path']}` — "
        f"{evals['n_total']} (с ответом {evals['n_positive']}, negative {evals['n_negative']})"
        + _split_note(report) + ".",
        f"Фрагмент до {config.get('chunk_max_chars')} символов. Эмбеддинги: `{config['embedding_model']}`, "
        f"в модель уходит {config['top_k']} фрагментов, RRF k={config['rrf_k']}, веса BM25 : dense "
        f"{num(config.get('bm25_weight', 1.0))} : {num(config.get('dense_weight', 1.0))}." + _runtime_note(report),
        "",
        "## Поиск (без LLM, negative не учитываются)",
        "",
        "Статья засчитывается по лучшему из своих фрагментов; hit@5 — нужная статья среди первых пяти разных статей.",
        "",
        "| Режим | n | hit@1 | hit@3 | hit@5 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        lines.append(_row(mode, report["retrieval"][mode]))
    by_split = report["retrieval"].get("hybrid_by_split") or {}
    if len(by_split) > 1:
        lines += [
            "",
            "Гибрид по частям набора (настройки поиска подбирались только на dev, test отложен):",
            "",
            "| Часть | n | hit@1 | hit@3 | hit@5 | MRR@10 |",
            "|---|---|---|---|---|---|",
        ]
        for name, m in by_split.items():
            lines.append(_row(name, m))
    ctx = report["retrieval"].get("context")
    multi = report["retrieval"].get("multi_article")
    if ctx:
        lines += [
            "",
            "Что реально получает модель (гибрид):",
            "",
            f"- Нужная статья среди {ctx['top_k']} фрагментов, которые уходят в модель: {pct(ctx['hit'])} "
            f"(n={ctx['n']}). Эти {ctx['top_k']} фрагментов взяты в среднем из {dec(ctx['distinct_articles_avg'], 1)} "
            f"статьи, минимум из {ctx['distinct_articles_min']}.",
        ]
    if multi and multi["n"]:
        lines.append(
            f"- Вопросы на две статьи (n={multi['n']}): хотя бы одна в top-5 статей — {pct(multi['any@5'])}, "
            f"обе в top-5 статей — {pct(multi['all@5'])}, обе среди фрагментов для модели — "
            f"{pct(multi['all_in_context'])}."
        )
    spots = report.get("spot_checks") or []
    if spots:
        lines += [
            "",
            "## Точечные проверки (не входят в метрики)",
            "",
            "Примеры со страницы и из README, запросы по номеру статьи. Ранг статьи; «—» — нет в top-10.",
            "",
            "| Запрос | Тип | Ожидалась | BM25 | dense | гибрид |",
            "|---|---|---|---|---|---|",
        ]
        for row in spots:
            r = row["rank"]
            lines.append(
                f"| {row['q']} | {row['type']} | {row['expected_id']} | {r['bm25'] or '—'} | "
                f"{r['dense'] or '—'} | {r['hybrid'] or '—'} |"
            )
    lines += ["", "## Ответы модели", ""]
    a = report["answers"]
    if a.get("status") in ("ok", "partial"):
        avg_cost = "—" if a["avg_cost_usd"] is None else f"${a['avg_cost_usd']:.5f}"
        avg_lat = "—" if a["avg_latency_s"] is None else f"{dec(a['avg_latency_s'], 2)} с"
        lines += [
            f"- Модель: `{a['model']}`, вопросов: {a['n']}" + (f" (неполный: {a['reason']})" if a["reason"] else ""),
            f"- Цитата попадает в ожидаемую статью: {pct(a['citation_hit_rate'])}",
            f"- Все цитаты ведут на найденные фрагменты: {pct(a['citation_valid_rate'])}",
            f"- Ложный отказ на вопросах с ответом: {pct(a['false_no_answer_rate'])}",
            f"- Корректный отказ на negative: {pct(a['negative_refusal_rate'])}",
            f"- Средняя стоимость: {avg_cost}, всего ${a['total_cost_usd']:.4f}",
            f"- Средняя латентность: {avg_lat}",
        ]
    else:
        lines.append(f"Не прогонялось: {a.get('reason')}.")
    return "\n".join(lines) + "\n"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def run_file_name(report: dict) -> str:
    stamp = report["generated_at"].replace("+00:00", "Z").replace(":", "")
    parts = [stamp, report["evals"].get("split", "all")]
    if report["evals"].get("limit"):
        parts.append(f"limit{report['evals']['limit']}")
    if report["answers"].get("status") in ("ok", "partial"):
        parts.append("answers")
    parts.append(_slug((report.get("runtime") or {}).get("platform") or "unknown"))
    return "-".join(parts) + ".json"


def is_full_run(report: dict) -> bool:
    return report["evals"].get("split", "all") == "all" and not report["evals"].get("limit")


def write_report(report: dict, out_dir: str | Path) -> list[Path]:
    """Always save the run under runs/; rewrite latest.* only for a full run."""
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    run_path = runs_dir / run_file_name(report)
    run_path.write_text(payload, encoding="utf-8")
    paths = [run_path]
    if is_full_run(report):
        (out_dir / "latest.json").write_text(payload, encoding="utf-8")
        (out_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
        paths += [out_dir / "latest.json", out_dir / "latest.md"]
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
    split: str = "all",
    limit: int | None = None,
    spot_checks: list[dict] | None = None,
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
            "split": split,
            "limit": limit,
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
        "runtime": runtime_info(),
        "retrieval": run_retrieval(retriever, items, top_k=top_k),
        "spot_checks": run_spot_checks(retriever, spot_checks or []),
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
    parser.add_argument(
        "--split", choices=SPLITS, default="all", help="dev (for tuning), test (held out) or all (default)"
    )
    parser.add_argument("--spot-checks", default=str(SPOT_CHECKS_PATH), help="page examples and lookups by number")
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
    items = select_split(load_evals(args.evals), args.split)[: args.limit]

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
        split=args.split,
        limit=args.limit,
        spot_checks=load_spot_checks(args.spot_checks),
    )
    for path in write_report(report, args.out):
        print(f"written {path}")
    if not is_full_run(report):
        print("latest.* not touched: only a full run (split all, no --limit) updates them")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
