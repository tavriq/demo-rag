"""Russian tokenization for BM25: lowercase, ё→е, stop words, Snowball stemming."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

# Words or numbers; article numbers like "312.1" stay one token.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)*|[^\W\d_]+", re.UNICODE)

STOP_WORDS = frozenset(
    """
    а без более бы был была были было быть в вам вас ведь во вот все всего всех вы где да
    даже для до его ее ей ему если есть еще же за здесь и из или им их к как какая какой
    когда кто ли либо мне может можно мы на над надо нас не него нее нет ни них но ну о об
    однако он она они оно от по под при про с со так также такой там те тем то того тоже
    той только том ты у уже чем что чтобы эта эти это этого этой этом этот я
    """.split()
)

_stemmer = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    if word[0].isdigit():
        return word
    return _stemmer.stemWord(word)


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: normalized, stop words removed, stemmed."""
    tokens = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        if raw in STOP_WORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        tokens.append(stem(raw))
    return tokens
