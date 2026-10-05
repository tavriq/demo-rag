import json

import pytest

from app.chunking import Article, chunk_article, chunk_corpus, load_corpus, split_text
from app.text import tokenize
from tests.conftest import CHUNK_MAX, CORPUS


def _article(text: str, title: str = "Тестовая статья", number: str = "1") -> Article:
    return Article(id=f"TK-{number}", article=number, title=title, chapter="Глава 1", text=text,
                   source_url="", edition_date="")


def test_fixture_loads_with_all_fields(articles):
    assert 10 <= len(articles) <= 15
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))
    assert "TK-312.1" in ids
    assert all(a.text and a.title and a.article for a in articles)


def test_short_article_is_one_chunk(articles):
    art = next(a for a in articles if a.id == "TK-115")
    chunks = chunk_article(art, CHUNK_MAX)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "TK-115#0"
    assert chunks[0].header == "Статья 115. Продолжительность ежегодного основного оплачиваемого отпуска"
    assert chunks[0].body == art.text


def test_long_article_split_keeps_header_and_text(articles):
    art = next(a for a in articles if a.id == "TK-81")
    assert len(art.text) > CHUNK_MAX
    chunks = chunk_article(art, CHUNK_MAX)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.header.startswith("Статья 81. Расторжение трудового договора по инициативе работодателя")
        assert f"(часть {i} из {len(chunks)})" in chunk.header
        assert chunk.display_text.startswith("Статья 81.")
        assert len(chunk.body) <= CHUNK_MAX
        assert chunk.doc_id == "TK-81"
    # nothing lost: the bodies cover the original text word for word
    assert " ".join(c.body for c in chunks).split() == art.text.split()


def test_oversized_paragraph_is_split_by_sentences_then_words():
    sentence = "Слово " * 40  # ~240 chars, no sentence punctuation
    text = "Первое предложение. " * 30 + "\n" + sentence
    parts = split_text(text, 100)
    assert all(len(p) <= 100 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_title_with_article_prefix_is_not_duplicated():
    art = _article("Текст.", title="Статья 5. Заголовок", number="5")
    assert chunk_article(art, 500)[0].header == "Статья 5. Заголовок"


def test_chunk_index_text_is_header_and_body(articles):
    chunks = chunk_corpus(articles, CHUNK_MAX)
    c = next(c for c in chunks if c.doc_id == "TK-108")
    assert c.index_text == f"{c.header}\n{c.body}"
    assert c.index_text.startswith("Статья 108.")
    assert "Глава 18" not in c.index_text


def test_duplicate_ids_rejected(tmp_path):
    line = json.dumps({"id": "TK-1", "article": "1", "title": "t", "text": "x"}, ensure_ascii=False)
    path = tmp_path / "dup.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_corpus(path)


def test_missing_fields_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "TK-1", "article": "1", "title": "t"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_corpus(path)


def test_tokenizer_stems_russian_word_forms():
    assert tokenize("отпуска")[0] == tokenize("отпуск")[0] == tokenize("отпуском")[0]
    assert tokenize("Ёлка") == tokenize("елка")
    assert "312.1" in tokenize("статья 312.1 кодекса")
    assert tokenize("и в на не") == []


def test_fixture_path_is_real():
    assert CORPUS.exists()
