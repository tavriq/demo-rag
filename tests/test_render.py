from app.render import anchor_for, render_answer_html, render_evals_page

A80 = {"80": {"header": "Статья 80. Расторжение трудового договора по инициативе работника",
              "source_url": "https://www.consultant.ru/document/cons_doc_LAW_34683/abc/", "anchor": "frag-TK-80-0"}}


def test_model_html_is_escaped():
    out = render_answer_html('<script>alert(1)</script><img src=x onerror="x()"> [ст. 80]', A80)
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert 'class="cite" data-article="80" data-anchor="frag-TK-80-0"' in out
    assert 'href="https://www.consultant.ru/document/cons_doc_LAW_34683/abc/" target="_blank"' in out
    assert ">ст.&nbsp;80</a>" in out


def test_citations_link_only_to_read_articles():
    out = render_answer_html("Норма [ст. 80], выдумка [ст. 999].", A80)
    assert 'data-article="80"' in out
    assert 'class="cite cite-missing"' in out and "ст. 999</span>" in out
    assert 'data-article="999"' not in out


def test_citation_without_source_url_links_to_fragment():
    out = render_answer_html("Норма [ст. 80].", {"80": {"header": "Статья 80", "source_url": "", "anchor": "frag-TK-80-0"}})
    assert 'href="#frag-TK-80-0"' in out and "target=" not in out


def test_injection_inside_citation_brackets_is_inert():
    out = render_answer_html('[ст. 80 "><script>x</script>]', A80)
    # only the article number survives: the rest of the bracket is dropped, never echoed
    assert "<script>" not in out and "script" not in out
    assert 'data-article="80"' in out


def test_sections_lists_and_disclaimer():
    text = ("Коротко: можно [ст. 80].\nПодробно:\n- первое [ст. 80]\n- второе 'в кавычках' \"x\"\n"
            "Исключения и сроки:\n- за 14 дней [ст. 80]\nЭто не юридическая консультация.")
    out = render_answer_html(text, A80)
    assert out.count('<h3 class="ans-h">') == 3 and "<h3 class=\"ans-h\">Исключения и сроки</h3>" in out
    assert out.count("<ul>") == 2 and out.count("<li>") == 3
    assert "второе 'в кавычках' \"x\"" in out
    assert '<p class="disclaimer">Это не юридическая консультация.</p>' in out


def test_unverified_numbers_are_marked_on_their_line_only():
    text = "- за 99 дней [ст. 80]\n- за 99 рублей и 14 дней, ст. 81 [ст. 80]"
    out = render_answer_html(text, A80, {"0": ["99"]})
    first, second = out.split("</li>")[:2]
    assert '<span class="num-unverified"' in first and ">99</span>" in first
    assert "num-unverified" not in second


def test_article_reference_numbers_are_not_marked():
    out = render_answer_html("по ст. 81 и статье 99 [ст. 80]", A80, {"0": ["81", "99"]})
    assert "num-unverified" not in out


def test_anchor_is_safe_identifier():
    assert anchor_for("TK-312.1#0") == "frag-TK-312-1-0"
    assert anchor_for('x"><script>') == "frag-x---script-"


def test_evals_page_escapes_file_content():
    latest = {
        "date": "2026-10-05",
        "corpus": {"path": "<script>bad()</script>", "n_docs": 1, "n_chunks": 2},
        "evals": {"n_total": 1, "n_positive": 1, "n_negative": 0},
        "config": {"embedding_model": "m", "top_k": 5},
        "retrieval": {
            "bm25": {"n": 1, "hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "mrr@10": 1.0},
            "main_mode": "dense",
            "main_by_split": {
                "dev": {"n": 1, "hit@1": 0.0, "hit@5": 1.0, "mrr@10": 0.5},
                "<i>test</i>": {"n": 1, "hit@1": 1.0, "hit@5": 1.0, "mrr@10": 1.0},
            },
            "misses": [{"q": "<b>q</b>", "expected_ids": ["TK-1"], "top": ["TK-2"]}],
        },
        "answers": {"status": "not_run", "reason": "нужен ключ <API>"},
    }
    page = render_evals_page(latest)
    assert "<script>bad()" not in page and "&lt;script&gt;bad()" in page
    assert "<b>q</b>" not in page
    assert "Не прогонялось: нужен ключ &lt;API&gt;" in page
    assert "100,0%" in page and "100.0%" not in page
    assert "Dense (e5-small) на dev и test" in page and "<i>test</i>" not in page
    assert "1 статья," in page


def test_evals_page_shows_partial_split_and_spot_checks():
    latest = {
        "date": "2026-10-05",
        "corpus": {"path": "c", "n_docs": 534, "n_chunks": 3200},
        "evals": {"n_total": 22, "n_positive": 22, "n_negative": 0, "split": "dev"},
        "config": {"embedding_model": "m", "top_k": 5, "bm25_weight": 1.0, "dense_weight": 1.5},
        "retrieval": {
            "context": {"top_k": 5, "n": 22, "hit": 0.75, "distinct_articles_avg": 3.5, "distinct_articles_min": 1},
            "multi_article": {"n": 4, "any@5": 0.75, "all@5": 0.25, "all_in_context": 0.0},
        },
        "spot_checks": [{"q": "<статья 1>", "type": "by_number", "expected_id": "TK-1",
                         "rank": {"bm25": 1, "dense": None, "hybrid": None}}],
        "answers": {"status": "not_run", "reason": "x"},
    }
    page = render_evals_page(latest)
    assert "534 статьи" in page and "Только часть набора: <strong>dev</strong>" in page
    assert "1 : 1,5" in page
    assert "75,0%" in page and "3,5" in page
    assert "&lt;статья 1&gt;" in page and "Точечные проверки" in page


def test_evals_page_shows_live_answer_metrics_escaped():
    latest = {
        "date": "2026-10-05",
        "corpus": {"path": "c", "n_docs": 534, "n_chunks": 3200},
        "evals": {"n_total": 50, "n_positive": 43, "n_negative": 7},
        "config": {"embedding_model": "m", "top_k": 5, "search_mode": "dense", "article_router": True},
        "retrieval": {"main_mode": "dense"},
        "answers": {
            "status": "ok", "model": "<b>m</b>", "gateway": "api.example.ai", "reasoning_effort": None,
            "n": 50, "n_positive": 43, "n_negative": 7, "citation_hit_rate": 0.8, "false_no_answer_rate": 0.05,
            "negative_refusal_rate": 1.0, "uncited_answer_rate": 0.0, "citation_valid_rate": 1.0,
            "avg_input_tokens": 1520.4, "avg_output_tokens": 140.2, "avg_reasoning_tokens": 0,
            "total_tokens": 83000, "p50_latency_s": 1.5, "p95_latency_s": 3.25, "total_cost_rub": None,
        },
    }
    page = render_evals_page(latest)
    assert "<b>m</b>" not in page and "&lt;b&gt;m&lt;/b&gt;" in page
    assert "шлюз api.example.ai" in page
    assert "1,50 с / 3,25 с" in page and "83\u202f000" in page
    assert "Режим поиска демо: <strong>Dense (e5-small)</strong>, роутер номеров статей включён" in page
    assert "₽" not in page  # no prices configured: no rubles shown


def test_evals_page_without_runs():
    page = render_evals_page(None)
    assert "Прогонов ещё не было" in page


def test_russian_number_formatting():
    from app.fmt import dec, num, pct, plural_ru

    assert [plural_ru(n, "статья", "статьи", "статей") for n in (1, 3, 5, 11, 21, 534, 112)] == [
        "статья", "статьи", "статей", "статей", "статья", "статьи", "статей"
    ]
    assert pct(0.8139) == "81,4%" and pct(None) == "—"
    assert dec(0.6734) == "0,673"
    assert num(1.5) == "1,5" and num(1.0) == "1"
