import pytest

from app.checks import check_answer, claim_numbers, numbers_in_text, strip_unknown_citations


def test_number_words_become_digits():
    found = numbers_in_text("не позднее чем за две недели, в течение трех рабочих дней, "
                            "двадцать восемь календарных дней, шестимесячного срока, сорока часов")
    assert {"2", "3", "28", "6", "40"} <= found


def test_weeks_also_count_in_days():
    assert {"2", "14"} <= numbers_in_text("не позднее чем за две недели")
    assert "21" in numbers_in_text("за 3 недели")


@pytest.mark.parametrize("text", ["однако", "одновременно", "семья работника", "в пятницу"])
def test_words_that_only_look_like_numbers(text):
    assert numbers_in_text(text) == set()


def test_claim_numbers_skip_citations_and_article_references():
    line = "Предупредить за 14 дней [ст. 80], по ст. 81 и статье 312.1 — 3 дня; 0,5 ставки"
    assert claim_numbers(line) == ["14", "3", "0.5"]


def test_unknown_citations_are_removed():
    text, removed = strip_unknown_citations("Норма [ст. 80] [ст. 999]. Ещё [ст. 80, 998].", {"80"})
    assert removed == ["999", "998"]
    assert text == "Норма [ст. 80]. Ещё [ст. 80]."


def test_check_answer_against_cited_articles(ctx):
    context = ctx("статья 80", 1)
    assert context[0].article == "80" and "две недели" in context[0].text
    answer = ("Коротко: за 2 недели [ст. 80].\n"
              "- через 99 дней [ст. 80]\n"
              "- выдумка [ст. 999]\n"
              "Это не юридическая консультация.")
    text, checks = check_answer(answer, context)
    assert "[ст. 999]" not in text and checks.removed == ["999"]
    assert checks.citations_total == 3
    assert checks.numbers_total == 2 and checks.numbers_missing == ["99"]
    assert checks.missing_by_line == {1: ["99"]}
    data = checks.to_json()
    assert data["citations_valid"] == 2 and data["numbers_found"] == 1 and data["missing_by_line"] == {"1": ["99"]}
