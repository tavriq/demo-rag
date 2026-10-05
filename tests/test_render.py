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
            "misses": [{"q": "<b>q</b>", "expected_ids": ["TK-1"], "top": ["TK-2"]}],
        },
        "answers": {"status": "not_run", "reason": "нужен ключ <API>"},
    }
    page = render_evals_page(latest)
    assert "<script>bad()" not in page and "&lt;script&gt;bad()" in page
    assert "<b>q</b>" not in page
    assert "Не прогонялось: нужен ключ &lt;API&gt;" in page
    assert "100.0%" in page


def test_evals_page_without_runs():
    page = render_evals_page(None)
    assert "Прогонов ещё не было" in page
