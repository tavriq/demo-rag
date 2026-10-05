"""Checks of a model answer against the articles it was given. Code, not a second model call.

1. Citations: [ст. N] of an article the model did not read is removed from the text.
2. Numbers: every number written in digits (days, months, percent, parts) is looked up
   in the text of the articles cited on the same line; a line without citations is
   checked against all articles read. The law often writes numbers in words ("двух
   недель"), so article text is normalised to a set of numbers first.

A found number does not prove the claim is right, a missing one is a strong signal
that it is not: the answer shows such numbers highlighted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.context import ContextArticle
from app.llm import ARTICLE_NUMBER_RE, CITATION_RE

# Longest prefix wins: "двадцат" before "два", "пятьдесят" before "пять".
_NUMBER_WORDS = sorted(
    [
        ("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14),
        ("пятнадцат", 15), ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18),
        ("девятнадцат", 19), ("двадцат", 20), ("тридцат", 30), ("сорок", 40),
        ("пятьдесят", 50), ("пятидесят", 50), ("шестьдесят", 60), ("шестидесят", 60),
        ("семьдесят", 70), ("семидесят", 70), ("восемьдесят", 80), ("восьмидесят", 80),
        ("девяност", 90), ("двест", 200), ("двухсот", 200), ("трист", 300), ("трехсот", 300),
        ("одн", 1), ("один", 1), ("двух", 2), ("два", 2), ("две", 2), ("двум", 2),
        ("трех", 3), ("трёх", 3), ("три", 3), ("трем", 3), ("трём", 3), ("четыр", 4),
        ("пят", 5), ("шест", 6), ("сем", 7), ("восем", 8), ("восьм", 8), ("девят", 9),
        ("десят", 10), ("полугод", 6),
    ],
    key=lambda kv: -len(kv[0]),
)
_EXACT_WORDS = {"сто": 100, "ста": 100}
_WORD_RE = re.compile(r"[а-яё]+")
_DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Words that start with a number stem but are not numbers ("однако", "семья", "пятница").
_NOT_NUMBER_RE = re.compile(
    r"^(?:одни|одних|одним|одними|однак\w*|одновремен\w*|однород\w*|однократ\w*|одинаков\w*"
    r"|семь[яиеюё]\w*|семей\w*|пятн\w*|трикотаж\w*)$"
)
# A number right after these words is a reference, not a quantity: "ст. 81", "статьи 80".
_REFERENCE_BEFORE_RE = re.compile(r"(?:\bст\.?|\bстат[а-яё]*|\bглав[а-яё]*|\bфз|№)\s*$", re.IGNORECASE)


_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[а-яё]+")


def numbers_in_text(text: str) -> set[str]:
    """Numbers in digits plus number words turned into digits.

    Weeks also count in days ("две недели" -> 2 and 14): the model is told to write
    deadlines in digits and often turns weeks into days.
    """
    found = {_norm(m.group(0)) for m in _DIGITS_RE.finditer(text)}
    tokens = _TOKEN_RE.findall(text.lower())
    values: list[float | None] = []
    for token in tokens:
        if token[0].isdigit():
            values.append(float(_norm(token)))
        elif _NOT_NUMBER_RE.match(token):
            values.append(None)
        else:
            word = _word_value(token)
            values.append(None if word is None else float(word))
    for i, value in enumerate(values):
        if value is None:
            continue
        nxt = values[i + 1] if i + 1 < len(values) else None
        if not tokens[i][0].isdigit():
            found.add(_fmt(value))
            # "двадцать восемь" -> 28
            if value % 10 == 0 and value >= 20 and nxt and nxt < 10:
                found.add(_fmt(value + nxt))
        if i + 1 < len(tokens) and tokens[i + 1].startswith("недел"):
            found.add(_fmt(value * 7))
    return found


def _fmt(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _word_value(word: str) -> int | None:
    if word in _EXACT_WORDS:
        return _EXACT_WORDS[word]
    for prefix, value in _NUMBER_WORDS:
        if word.startswith(prefix):
            return value
    return None


def _norm(number: str) -> str:
    return number.replace(",", ".")


@dataclass
class AnswerChecks:
    citations_total: int = 0
    removed: list[str] = field(default_factory=list)  # article numbers cited but not read
    numbers_total: int = 0
    numbers_missing: list[str] = field(default_factory=list)
    missing_by_line: dict[int, list[str]] = field(default_factory=dict)  # line index -> numbers

    @property
    def numbers_found(self) -> int:
        return self.numbers_total - len(self.numbers_missing)

    def to_json(self) -> dict:
        return {
            "citations_total": self.citations_total,
            "citations_valid": self.citations_total - len(self.removed),
            "removed": self.removed,
            "numbers_total": self.numbers_total,
            "numbers_found": self.numbers_found,
            "numbers_missing": self.numbers_missing,
            "missing_by_line": {str(k): v for k, v in self.missing_by_line.items()},
        }


def strip_unknown_citations(text: str, known: set[str]) -> tuple[str, list[str]]:
    """Drop article numbers the model did not read; drop the bracket if nothing is left."""
    removed: list[str] = []

    def fix(match: re.Match) -> str:
        numbers = ARTICLE_NUMBER_RE.findall(match.group(1))
        kept = [n for n in numbers if n in known]
        removed.extend(n for n in numbers if n not in known and n not in removed)
        if not numbers or kept == numbers:
            return match.group(0)
        return " ".join(f"[ст. {n}]" for n in kept)

    cleaned = CITATION_RE.sub(fix, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, removed


def claim_numbers(line: str) -> list[str]:
    """Numbers in digits outside citations and article references, in order, unique."""
    without_citations = CITATION_RE.sub(lambda m: " " * len(m.group(0)), line)
    out: list[str] = []
    for m in _DIGITS_RE.finditer(without_citations):
        if _REFERENCE_BEFORE_RE.search(without_citations[: m.start()]):
            continue
        n = _norm(m.group(0))
        if n not in out:
            out.append(n)
    return out


def check_answer(text: str, context: list[ContextArticle]) -> tuple[str, AnswerChecks]:
    """Return the cleaned text and what was checked. The disclaimer line has no numbers to check."""
    known = {a.article for a in context}
    checks = AnswerChecks()
    for m in CITATION_RE.finditer(text):
        checks.citations_total += len(ARTICLE_NUMBER_RE.findall(m.group(1)))
    text, checks.removed = strip_unknown_citations(text, known)

    numbers_by_article = {a.article: numbers_in_text(a.text) for a in context}
    all_numbers: set[str] = set().union(*numbers_by_article.values()) if numbers_by_article else set()
    for i, line in enumerate(text.split("\n")):
        cited = [n for m in CITATION_RE.finditer(line) for n in ARTICLE_NUMBER_RE.findall(m.group(1))]
        pool = set().union(*(numbers_by_article.get(a, set()) for a in cited)) if cited else all_numbers
        for n in claim_numbers(line):
            checks.numbers_total += 1
            if n not in pool:
                checks.missing_by_line.setdefault(i, []).append(n)
                if n not in checks.numbers_missing:
                    checks.numbers_missing.append(n)
    return text, checks
