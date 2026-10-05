"""Server-side HTML rendering. Every piece of user, model or file text is escaped first."""

from __future__ import annotations

import html
import re

from app.llm import ARTICLE_NUMBER_RE, CITATION_RE

_ANCHOR_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def anchor_for(chunk_id: str) -> str:
    return "frag-" + _ANCHOR_UNSAFE_RE.sub("-", chunk_id)


def render_answer_html(text: str, article_anchors: dict[str, str]) -> str:
    """Escape the answer, then turn [ст. N] into links to the matching fragment.

    A citation of an article that is not among the retrieved fragments is shown
    as a highlighted span, not a link: it is a visible signal of a bad citation.
    """
    # quote=False: only &, <, > become entities, so no digits appear inside entities
    escaped = html.escape(text, quote=False)

    def link_numbers(match: re.Match) -> str:
        inner = match.group(1)

        def one(num_match: re.Match) -> str:
            number = num_match.group(0)
            anchor = article_anchors.get(number)
            if anchor:
                return f'<a class="cite" href="#{esc(anchor)}" data-anchor="{esc(anchor)}">{number}</a>'
            return f'<span class="cite cite-missing" title="Этой статьи нет среди найденных фрагментов">{number}</span>'

        return "[" + ARTICLE_NUMBER_RE.sub(one, inner) + "]"

    linked = CITATION_RE.sub(link_numbers, escaped)
    return linked.replace("\n", "<br>")


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


_PAGE_HEAD = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="/static/app.css">
</head>
<body>
<main class="wrap">
"""

_PAGE_TAIL = """
</main>
</body>
</html>
"""


def render_evals_page(latest: dict | None) -> str:
    out = [_PAGE_HEAD.format(title="Метрики RAG")]
    out.append('<p class="back"><a href="/">← к вопросам</a></p>')
    out.append("<h1>Метрики качества</h1>")
    if not latest:
        out.append(
            '<p class="notice">Прогонов ещё не было: файла evals/latest.json нет. '
            "Запуск: <code>python3 -m app.evals</code>.</p>"
        )
        out.append(_PAGE_TAIL)
        return "".join(out)

    corpus = latest.get("corpus", {})
    evals = latest.get("evals", {})
    config = latest.get("config", {})
    out.append(
        '<p class="lead">Прогон от <strong>{date}</strong>. Корпус: {docs} статей, {chunks} чанков '
        "(<code>{cpath}</code>). Вопросов: {total}, из них с ожидаемой статьёй {pos}, "
        "без ответа в базе (negative) {neg}.</p>".format(
            date=esc(latest.get("date")),
            docs=esc(corpus.get("n_docs")),
            chunks=esc(corpus.get("n_chunks")),
            cpath=esc(corpus.get("path")),
            total=esc(evals.get("n_total")),
            pos=esc(evals.get("n_positive")),
            neg=esc(evals.get("n_negative")),
        )
    )
    out.append(
        "<p class=\"muted\">Чанк до {chunk} символов. Эмбеддинги: <code>{model}</code>, top-k {k}, кандидатов на ретривер {cand}, "
        "RRF k={rrf}, веса BM25/dense {wb}/{wd}. Negative-вопросы в hit@k не входят.</p>".format(
            chunk=esc(config.get("chunk_max_chars")),
            model=esc(config.get("embedding_model")),
            k=esc(config.get("top_k")),
            cand=esc(config.get("candidates")),
            rrf=esc(config.get("rrf_k")),
            wb=esc(config.get("bm25_weight", 1.0)),
            wd=esc(config.get("dense_weight", 1.0)),
        )
    )

    retrieval = latest.get("retrieval", {})
    out.append("<h2>Поиск (без LLM)</h2>")
    out.append('<div class="table-scroll"><table><thead><tr><th>Режим</th><th>n</th>'
               "<th>hit@1</th><th>hit@3</th><th>hit@5</th><th>MRR@10</th></tr></thead><tbody>")
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid": "Гибрид (RRF)"}
    for mode in ("bm25", "dense", "hybrid"):
        m = retrieval.get(mode)
        if not m:
            continue
        out.append(
            "<tr><td>{label}</td><td>{n}</td><td>{h1}</td><td>{h3}</td><td>{h5}</td><td>{mrr}</td></tr>".format(
                label=esc(labels[mode]),
                n=esc(m.get("n")),
                h1=_pct(m.get("hit@1")),
                h3=_pct(m.get("hit@3")),
                h5=_pct(m.get("hit@5")),
                mrr=_num(m.get("mrr@10")),
            )
        )
    out.append("</tbody></table></div>")

    by_type = retrieval.get("hybrid_by_type") or {}
    if by_type:
        out.append("<h3>Гибрид по типам вопросов</h3>")
        out.append('<div class="table-scroll"><table><thead><tr><th>Тип</th><th>n</th>'
                   "<th>hit@1</th><th>hit@5</th><th>MRR@10</th></tr></thead><tbody>")
        for qtype, m in sorted(by_type.items()):
            out.append(
                "<tr><td>{t}</td><td>{n}</td><td>{h1}</td><td>{h5}</td><td>{mrr}</td></tr>".format(
                    t=esc(qtype), n=esc(m.get("n")), h1=_pct(m.get("hit@1")),
                    h5=_pct(m.get("hit@5")), mrr=_num(m.get("mrr@10")),
                )
            )
        out.append("</tbody></table></div>")

    by_split = retrieval.get("hybrid_by_split") or {}
    if len(by_split) > 1:
        out.append("<h3>Гибрид на dev и test</h3>")
        out.append(
            '<p class="muted">Размер чанка и веса RRF подбирались только на dev; test в подборе не участвовал, '
            "поэтому честная оценка — строка test. Протокол и все итерации — <code>evals/tuning.md</code>.</p>"
        )
        out.append('<div class="table-scroll"><table><thead><tr><th>Часть</th><th>n</th>'
                   "<th>hit@1</th><th>hit@5</th><th>MRR@10</th></tr></thead><tbody>")
        for name, m in sorted(by_split.items()):
            out.append(
                "<tr><td>{t}</td><td>{n}</td><td>{h1}</td><td>{h5}</td><td>{mrr}</td></tr>".format(
                    t=esc(name), n=esc(m.get("n")), h1=_pct(m.get("hit@1")),
                    h5=_pct(m.get("hit@5")), mrr=_num(m.get("mrr@10")),
                )
            )
        out.append("</tbody></table></div>")

    misses = retrieval.get("misses") or []
    if misses:
        out.append(f"<details><summary>Промахи гибрида в top-5 ({len(misses)})</summary><ul class=\"misses\">")
        for miss in misses:
            out.append(
                "<li>{q} <span class=\"muted\">ожидалось {exp}, найдено {got}</span></li>".format(
                    q=esc(miss.get("q")),
                    exp=esc(", ".join(miss.get("expected_ids", []))),
                    got=esc(", ".join(miss.get("top", []))),
                )
            )
        out.append("</ul></details>")

    answers = latest.get("answers") or {}
    out.append("<h2>Ответы модели</h2>")
    if answers.get("status") in ("ok", "partial"):
        rows = [
            ("Модель", esc(answers.get("model"))),
            ("Вопросов прогнано", esc(answers.get("n"))),
            ("Цитата попадает в ожидаемую статью", _pct(answers.get("citation_hit_rate"))),
            ("Все цитаты ведут на найденные фрагменты", _pct(answers.get("citation_valid_rate"))),
            ("Ложный отказ на вопросах с ответом", _pct(answers.get("false_no_answer_rate"))),
            ("Корректный отказ на negative", _pct(answers.get("negative_refusal_rate"))),
            ("Средняя стоимость ответа", "$" + _num(answers.get("avg_cost_usd"), 5)),
            ("Средняя латентность", _num(answers.get("avg_latency_s"), 2) + " с"),
            ("p95 латентности", _num(answers.get("p95_latency_s"), 2) + " с"),
        ]
        out.append('<div class="table-scroll"><table><tbody>')
        for label, value in rows:
            out.append(f"<tr><th>{esc(label)}</th><td>{value}</td></tr>")
        out.append("</tbody></table></div>")
        if answers.get("status") == "partial":
            out.append(f'<p class="notice">Прогон неполный: {esc(answers.get("reason"))}</p>')
    else:
        out.append(
            '<p class="notice">Не прогонялось: {reason}.</p>'.format(
                reason=esc(answers.get("reason") or "нужен ключ ANTHROPIC_API_KEY")
            )
        )
    out.append(_PAGE_TAIL)
    return "".join(out)
