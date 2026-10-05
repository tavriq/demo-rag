from app.render import anchor_for, render_answer_html, render_evals_page


def test_model_html_is_escaped():
    out = render_answer_html('<script>alert(1)</script><img src=x onerror="x()"> [ст. 80]', {"80": "frag-TK-80-0"})
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
    assert '<a class="cite" href="#frag-TK-80-0" data-anchor="frag-TK-80-0">80</a>' in out


def test_citations_link_only_to_retrieved_fragments():
    out = render_answer_html("Норма [ст. 115], выдумка [ст. 999].", {"115": "frag-TK-115-0"})
    assert 'href="#frag-TK-115-0"' in out
    assert 'class="cite cite-missing"' in out and ">999</span>" in out
    assert "#frag-TK-999" not in out


def test_injection_inside_citation_brackets_is_inert():
    out = render_answer_html('[ст. 80 "><script>x</script>]', {"80": "frag-TK-80-0"})
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_newlines_become_breaks_and_quotes_survive():
    assert render_answer_html("a\nb 'c' \"d\"", {}) == "a<br>b 'c' \"d\""


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
            "hybrid_by_split": {
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
    assert "Гибрид на dev и test" in page and "<i>test</i>" not in page
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
