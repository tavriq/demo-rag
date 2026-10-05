"""Server-side HTML rendering. Every piece of user, model or file text is escaped first."""

from __future__ import annotations

import html
import re

from app.fmt import dec, num, pct, plural_ru
from app.llm import ARTICLE_NUMBER_RE, CITATION_RE

_ANCHOR_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def anchor_for(chunk_id: str) -> str:
    return "frag-" + _ANCHOR_UNSAFE_RE.sub("-", chunk_id)


_SECTION_RE = re.compile(r"^(Коротко|Подробно|Исключения и сроки)\s*:\s*(.*)$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[-•–—*]\s+(.*)$")
_CLAIM_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_REFERENCE_BEFORE_RE = re.compile(r"(?:\bст\.?|\bстат[а-яё]*|\bглав[а-яё]*|\bфз|№)\s*$", re.IGNORECASE)


def _mark_numbers(segment: str, missing: set[str]) -> str:
    """Escape a citation-free piece of text; wrap numbers not found in the cited articles."""
    if not missing:
        return html.escape(segment, quote=False)
    out: list[str] = []
    pos = 0
    for m in _CLAIM_NUMBER_RE.finditer(segment):
        number = m.group(0).replace(",", ".")
        if number not in missing or _REFERENCE_BEFORE_RE.search(segment[: m.start()]):
            continue
        out.append(html.escape(segment[pos : m.start()], quote=False))
        out.append(
            '<span class="num-unverified" title="Этого числа нет в тексте статей, на которые ссылается строка">'
            f"{html.escape(m.group(0), quote=False)}</span>"
        )
        pos = m.end()
    out.append(html.escape(segment[pos:], quote=False))
    return "".join(out)


def _cite_html(number: str, articles: dict[str, dict]) -> str:
    info = articles.get(number)
    if not info:
        return f'<span class="cite cite-missing" title="Этой статьи нет среди прочитанных моделью">ст. {esc(number)}</span>'
    attrs = f'class="cite" data-article="{esc(number)}"'
    if info.get("anchor"):
        attrs += f' data-anchor="{esc(info["anchor"])}"'
    href = info.get("source_url") or (f"#{info['anchor']}" if info.get("anchor") else "#")
    external = ' target="_blank" rel="noopener noreferrer"' if info.get("source_url") else ""
    title = f' title="{esc(info["header"])}"' if info.get("header") else ""
    return f'<a {attrs} href="{esc(href)}"{external}{title}>ст.&nbsp;{esc(number)}</a>'


def _inline(line: str, articles: dict[str, dict], missing: set[str]) -> str:
    out: list[str] = []
    pos = 0
    for m in CITATION_RE.finditer(line):
        out.append(_mark_numbers(line[pos : m.start()], missing))
        numbers = ARTICLE_NUMBER_RE.findall(m.group(1))
        out.append(" ".join(_cite_html(n, articles) for n in numbers) if numbers else html.escape(m.group(0), quote=False))
        pos = m.end()
    out.append(_mark_numbers(line[pos:], missing))
    return "".join(out)


def render_answer_html(text: str, articles: dict[str, dict], missing_by_line: dict | None = None) -> str:
    """Escape the answer and lay it out: sections, bullet lists, citation links.

    ``articles`` maps an article number to {header, source_url, anchor}. A citation of
    an article the model did not read is shown as a highlighted span, not a link.
    ``missing_by_line`` ({line index: [numbers]}, from app/checks.py) marks numbers not
    found in the articles cited on that line.
    """
    missing_by_line = {int(k): set(v) for k, v in (missing_by_line or {}).items()}
    out: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for i, raw in enumerate(text.split("\n")):
        line = raw.strip()
        missing = missing_by_line.get(i, set())
        if not line:
            close_list()
            continue
        section = _SECTION_RE.match(line)
        if section:
            close_list()
            out.append(f'<h3 class="ans-h">{esc(section.group(1).capitalize())}</h3>')
            rest = section.group(2)
            if rest:
                rest = rest[0].upper() + rest[1:]
                out.append(f"<p>{_inline(rest, articles, missing)}</p>")
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(bullet.group(1), articles, missing)}</li>")
            continue
        close_list()
        cls = ' class="disclaimer"' if "не юридическая консультация" in line.lower() else ""
        out.append(f"<p{cls}>{_inline(line, articles, missing)}</p>")
    close_list()
    return "\n".join(out)


def _pct(value) -> str:
    return pct(value)


def _num(value, digits: int = 3) -> str:
    return dec(value, digits)


def _tokens(value) -> str:
    return "—" if value is None else esc(f"{value:,.0f}".replace(",", "\u202f"))


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
    n_docs = corpus.get("n_docs") or 0
    split = evals.get("split", "all")
    split_note = "" if split == "all" else f" Только часть набора: <strong>{esc(split)}</strong>."
    if evals.get("limit"):
        split_note += f" Только первые {esc(evals.get('limit'))} вопросов."
    out.append(
        '<p class="lead">Прогон от <strong>{date}</strong>. Корпус: {docs} {docs_word}, {chunks} фрагментов '
        "(<code>{cpath}</code>). Вопросов: {total}, из них с ожидаемой статьёй {pos}, "
        "без ответа в базе (negative) {neg}.{split_note}</p>".format(
            date=esc(latest.get("date")),
            docs=esc(n_docs),
            docs_word=plural_ru(n_docs, "статья", "статьи", "статей"),
            chunks=esc(corpus.get("n_chunks")),
            cpath=esc(corpus.get("path")),
            total=esc(evals.get("n_total")),
            pos=esc(evals.get("n_positive")),
            neg=esc(evals.get("n_negative")),
            split_note=split_note,
        )
    )
    retrieval = latest.get("retrieval", {})
    main = retrieval.get("main_mode") or config.get("search_mode") or "hybrid"
    labels = {"bm25": "BM25", "dense": "Dense (e5-small)", "hybrid": "Гибрид (RRF)"}
    router = config.get("article_router")
    out.append(
        "<p class=\"muted\">Фрагмент до {chunk} символов. Эмбеддинги: <code>{model}</code>, в модель уходит {k} "
        "фрагментов, кандидатов на каждый поиск {cand}, RRF k={rrf}, веса BM25 : dense {wb} : {wd}. "
        "Режим поиска демо: <strong>{main}</strong>{router}. "
        "Платформа прогона: {plat}. Negative-вопросы в hit@k не входят. Статья засчитывается по лучшему "
        "из своих фрагментов; hit@5 — нужная статья среди первых пяти разных статей.</p>".format(
            plat=esc((latest.get("runtime") or {}).get("platform", "—")),
            chunk=esc(config.get("chunk_max_chars")),
            model=esc(config.get("embedding_model")),
            k=esc(config.get("top_k")),
            cand=esc(config.get("candidates")),
            rrf=esc(config.get("rrf_k")),
            wb=esc(num(config.get("bm25_weight", 1.0))),
            wd=esc(num(config.get("dense_weight", 1.0))),
            main=esc(labels.get(main, main)),
            router="" if router is None else (", роутер номеров статей " + ("включён" if router else "выключен")),
        )
    )

    out.append("<h2>Поиск (без LLM)</h2>")
    out.append('<div class="table-scroll"><table><thead><tr><th>Режим</th><th>n</th>'
               "<th>hit@1</th><th>hit@3</th><th>hit@5</th><th>MRR@10</th></tr></thead><tbody>")
    for mode in ("bm25", "dense", "hybrid"):
        m = retrieval.get(mode)
        if not m:
            continue
        out.append(
            "<tr><td>{label}</td><td>{n}</td><td>{h1}</td><td>{h3}</td><td>{h5}</td><td>{mrr}</td></tr>".format(
                label=esc(labels[mode] + (", режим демо" if mode == main else "")),
                n=esc(m.get("n")),
                h1=_pct(m.get("hit@1")),
                h3=_pct(m.get("hit@3")),
                h5=_pct(m.get("hit@5")),
                mrr=_num(m.get("mrr@10")),
            )
        )
    out.append("</tbody></table></div>")

    main_label = esc(labels.get(main, main))
    by_type = retrieval.get("main_by_type") or {}
    if by_type:
        out.append(f"<h3>{main_label} по типам вопросов</h3>")
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

    by_split = retrieval.get("main_by_split") or {}
    if len(by_split) > 1:
        out.append(f"<h3>{main_label} на dev и test</h3>")
        out.append(
            '<p class="muted">Размер фрагмента, веса и режим поиска выбирались только по dev; test в выборе не '
            "участвовал, поэтому честная оценка — строка test. Протокол и все итерации — "
            "<code>evals/tuning.md</code>.</p>"
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

    ctx = retrieval.get("context")
    multi = retrieval.get("multi_article")
    if ctx or (multi and multi.get("n")):
        out.append("<h3>Что реально получает модель</h3><ul>")
        if ctx:
            out.append(
                "<li>Нужная статья среди {k} фрагментов, которые уходят в модель: <strong>{hit}</strong> "
                "(n={n}). Эти фрагменты взяты в среднем из {avg} статьи, минимум из {mn}.</li>".format(
                    k=esc(ctx.get("top_k")), hit=_pct(ctx.get("hit")), n=esc(ctx.get("n")),
                    avg=esc(_num(ctx.get("distinct_articles_avg"), 1)), mn=esc(ctx.get("distinct_articles_min")),
                )
            )
        if multi and multi.get("n"):
            out.append(
                "<li>Вопросы на две статьи (n={n}): хотя бы одна в top-5 статей — {any5}, обе в top-5 статей — "
                "<strong>{all5}</strong>, обе среди фрагментов для модели — <strong>{allctx}</strong>.</li>".format(
                    n=esc(multi.get("n")), any5=_pct(multi.get("any@5")), all5=_pct(multi.get("all@5")),
                    allctx=_pct(multi.get("all_in_context")),
                )
            )
        out.append("</ul>")

    misses = retrieval.get("misses") or []
    if misses:
        out.append(f"<details><summary>Промахи режима демо в top-5 ({len(misses)})</summary><ul class=\"misses\">")
        for miss in misses:
            out.append(
                "<li>{q} <span class=\"muted\">ожидалось {exp}, найдено {got}</span></li>".format(
                    q=esc(miss.get("q")),
                    exp=esc(", ".join(miss.get("expected_ids", []))),
                    got=esc(", ".join(miss.get("top", []))),
                )
            )
        out.append("</ul></details>")

    spots = latest.get("spot_checks") or []
    if spots:
        out.append("<h2>Точечные проверки</h2>")
        out.append(
            '<p class="muted">Не входят в метрики: примеры со страницы и из README, запросы по номеру статьи. '
            "Ранг нужной статьи; «—» — нет в top-10. Роутер номеров статей действует во всех трёх режимах.</p>"
        )
        out.append('<div class="table-scroll"><table><thead><tr><th>Запрос</th><th>Тип</th><th>Ожидалась</th>'
                   "<th>BM25</th><th>dense</th><th>гибрид</th></tr></thead><tbody>")
        for row in spots:
            r = row.get("rank") or {}
            out.append(
                "<tr><td>{q}</td><td>{t}</td><td>{e}</td><td>{b}</td><td>{d}</td><td>{h}</td></tr>".format(
                    q=esc(row.get("q")), t=esc(row.get("type")), e=esc(row.get("expected_id")),
                    b=esc(r.get("bm25") or "—"), d=esc(r.get("dense") or "—"), h=esc(r.get("hybrid") or "—"),
                )
            )
        out.append("</tbody></table></div>")

    answers = latest.get("answers") or {}
    out.append("<h2>Ответы модели</h2>")
    if answers.get("status") in ("ok", "partial"):
        model = esc(answers.get("model"))
        if answers.get("reasoning_effort"):
            model += f", reasoning_effort {esc(answers.get('reasoning_effort'))}"
        if answers.get("gateway"):
            model += f", шлюз {esc(answers.get('gateway'))}"
        tokens = "вход {i}, выход {o}, из них рассуждения {r}".format(
            i=_tokens(answers.get("avg_input_tokens")),
            o=_tokens(answers.get("avg_output_tokens")),
            r=_tokens(answers.get("avg_reasoning_tokens")),
        )
        rows = [
            ("Модель", model),
            ("Вопросов", "{n} (с ответом в кодексе {p}, без ответа {g})".format(
                n=esc(answers.get("n")), p=esc(answers.get("n_positive")), g=esc(answers.get("n_negative")))),
            ("Ответ ссылается на ожидаемую статью", _pct(answers.get("citation_hit_rate"))),
            ("Ложный отказ, хотя ответ в кодексе есть", _pct(answers.get("false_no_answer_rate"))),
            ("Корректный отказ на вопросах без ответа", _pct(answers.get("negative_refusal_rate"))),
            ("Ответы без единой ссылки (отказы не считаются)", _pct(answers.get("uncited_answer_rate"))),
            ("Все ссылки ведут на полученные моделью фрагменты", _pct(answers.get("citation_valid_rate"))),
            ("Токены на ответ в среднем", tokens),
            ("Всего токенов за прогон", _tokens(answers.get("total_tokens"))),
            ("Задержка ответа модели, p50 / p95", "{a} с / {b} с".format(
                a=_num(answers.get("p50_latency_s"), 2), b=_num(answers.get("p95_latency_s"), 2))),
        ]
        if answers.get("total_cost_rub") is not None:
            rows.append(("Стоимость прогона", "≈ " + _num(answers.get("total_cost_rub"), 2) + " ₽"))
        out.append('<div class="table-scroll"><table><tbody>')
        for label, value in rows:
            out.append(f"<tr><th>{esc(label)}</th><td>{value}</td></tr>")
        out.append("</tbody></table></div>")
        if answers.get("status") == "partial":
            out.append(f'<p class="notice">Прогон неполный: {esc(answers.get("reason"))}</p>')
    else:
        out.append(
            '<p class="notice">Не прогонялось: {reason}.</p>'.format(
                reason=esc(answers.get("reason") or "нужен ключ API")
            )
        )
    out.append(_PAGE_TAIL)
    return "".join(out)
