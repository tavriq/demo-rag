"""Reproducible checks of data/evals.jsonl against data/corpus.jsonl.

    python3 data/check_evals.py [--corpus data/corpus.jsonl] [--evals data/evals.jsonl]

Only the standard library. Exit code 1 if a check fails.

1. Every expected_id exists in the corpus.
2. No question copies text from its expected article: the longest run of
   consecutive words shared by the question and the article (title + text) must
   be shorter than 5 words. Words are lowercased, ё→е, no stemming.
3. Negative questions really have no answer in the corpus: regular expressions
   for tax rates, the minimum wage in roubles, fine amounts, patent prices, sick
   pay percentages, mortgages and tax deductions find nothing in any article.

Not automated: that each expected article actually contains the answer. That
was checked by reading the articles (see data/README.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_SHARED_WORDS = 4  # a shared run of 5+ consecutive words fails the check
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# What a corpus answer to each negative question would have to look like.
NEGATIVE_PATTERNS = {
    "ставка НДФЛ": r"(?:ндфл|налог\w* на доходы)[^.]{0,80}\d+\s*(?:%|процент)",
    "МРОТ в рублях": r"(?:мрот|минимальн\w+ размер\w* оплаты труда)[^.]{0,80}\d[\d\s]*\s*руб",
    "процент больничного": r"(?:нетрудоспособност\w*|больничн\w*)[^.]{0,80}\d+\s*(?:%|процент)",
    "штраф в рублях": r"штраф\w*[^.]{0,80}\d[\d\s]*\s*(?:руб|тыс)",
    "стоимость патента": r"патент\w*[^.]{0,80}\d[\d\s]*\s*руб",
    "ипотека": r"ипотек",
    "налоговый вычет": r"налогов\w+ вычет",
}


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("ё", "е"))


def longest_shared_run(a: list[str], b: list[str]) -> tuple[int, list[str]]:
    """Longest common run of consecutive words (dynamic programming over word pairs)."""
    best, best_end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, best_end = cur[j], i
        prev = cur
    return best, a[best_end - best : best_end]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--evals", default="data/evals.jsonl")
    args = parser.parse_args(argv)

    corpus = {row["id"]: row for row in load_jsonl(Path(args.corpus))}
    items = load_jsonl(Path(args.evals))
    failures = 0

    missing = [(it["q"], i) for it in items for i in it.get("expected_ids", []) if i not in corpus]
    print(f"1. expected_ids в корпусе: {'ok' if not missing else 'НЕТ: ' + str(missing)}")
    failures += len(missing)

    runs = []
    for it in items:
        q_words = words(it["q"])
        for doc_id in it.get("expected_ids", []):
            if doc_id in corpus:
                article = corpus[doc_id]
                length, run = longest_shared_run(q_words, words(f"{article['title']} {article['text']}"))
                runs.append((length, doc_id, " ".join(run), it["q"]))
    runs.sort(reverse=True)
    worst = runs[0][0] if runs else 0
    too_long = [r for r in runs if r[0] > MAX_SHARED_WORDS]
    print(
        f"2. Самый длинный общий кусок вопроса и ожидаемой статьи: {worst} слов(а) "
        f"(порог: меньше {MAX_SHARED_WORDS + 1}). Пар вопрос-статья: {len(runs)}."
    )
    for length, doc_id, run, q in runs[:5]:
        print(f"   {length}: «{run}» — {doc_id} — {q}")
    failures += len(too_long)

    negatives = [it for it in items if it.get("type") == "negative" or not it.get("expected_ids")]
    print(f"3. Negative-вопросов: {len(negatives)}. Поиск ответа на них по всему корпусу:")
    for name, pattern in NEGATIVE_PATTERNS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        found = [doc_id for doc_id, a in corpus.items() if rx.search(f"{a['title']} {a['text']}".replace("ё", "е"))]
        print(f"   {name}: {'совпадений нет' if not found else 'НАЙДЕНО в ' + ', '.join(found[:10])}")
        failures += len(found)

    print("Итог:", "все проверки прошли" if not failures else f"провалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
