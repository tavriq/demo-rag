from app.context import GAP, ArticleStore, format_ranges
from app.retrieval import SearchHit


def _hit(chunk):
    return SearchHit(chunk, 1.0, None, 1, None, 0.9)


def test_short_article_goes_in_full(store, retriever):
    hits = retriever.search("статья 80", top_k=1)
    context = store.build_context(hits)
    a80 = next(a for a in context if a.article == "80")
    assert a80.full and a80.coverage == "целиком"
    assert a80.text == store.full_text("80")
    assert a80.header.startswith("Статья 80.")


def test_long_article_is_a_window_around_the_hit(store):
    chunks = store.chunks("81")  # 5 parts in the test corpus
    context = store.build_context([_hit(chunks[2])], full_article_chars=500)
    a81 = context[0]
    assert not a81.full and a81.parts == (2, 3, 4) and a81.coverage == "части 2–4 из 5"
    assert a81.text.startswith(GAP) and a81.text.endswith(GAP)
    assert chunks[2].body in a81.text and chunks[0].body not in a81.text


def test_articles_follow_hit_order_and_are_not_repeated(store):
    c81, c80, c115 = store.chunks("81")[0], store.chunks("80")[0], store.chunks("115")[0]
    context = store.build_context([_hit(c81), _hit(c80), _hit(store.chunks("81")[1]), _hit(c115)])
    assert [a.article for a in context] == ["81", "80", "115"]


def test_budget_skips_what_does_not_fit_but_keeps_the_first(store):
    big, small = store.chunks("81")[0], store.chunks("115")[0]
    first_len = len(store.full_text("81"))
    context = store.build_context([_hit(big), _hit(small)], max_chars=first_len + 10)
    assert [a.article for a in context] == ["81"]
    # the first article always goes in, cut to a window if the whole one does not fit
    tiny = store.build_context([_hit(big)], max_chars=200)
    assert len(tiny) == 1 and not tiny[0].full and tiny[0].parts == (1,)


def test_to_json_has_no_text_and_says_what_was_read(store):
    data = store.article("115").to_json()
    assert "text" not in data and data["coverage"] == "целиком" and data["chars"] > 0
    assert data["source_url"].startswith("https://")


def test_format_ranges():
    assert format_ranges((1, 2, 3, 7, 9, 10)) == "1–3, 7, 9–10"
    assert format_ranges((4,)) == "4"


def test_store_is_built_from_index_chunks(retriever):
    store = ArticleStore(retriever.index.chunks)
    assert "312.1" in store and "999" not in store and len(store) == 15
