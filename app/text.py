"""Russian tokenization for BM25: lowercase, ё→е, stop words, Snowball stemming."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

# Words or numbers; article numbers like "312.1" stay one token.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)*|[^\W\d_]+", re.UNICODE)

# Based on the common NLTK Russian list, numerals removed (они значимы: «три месяца»),
# plus forms frequent in user questions (меня, какие, сколько, который…).
STOP_WORDS = frozenset(
    """
    а без более бы был была были было быть в вам вами вас ваш ваша ваше ваши вдруг ведь во вот впрочем
    все всегда всего всех всю вы где да даже для до другой его ее ей ему если есть еще ж же за зачем
    здесь и из или им ими иногда их к каждый как какая какие каким каких какое каком какой какую ко
    когда кто куда ли либо лучше между меня мне мной много может можно мой моя мое мои мою мы на
    над надо наконец нам нами нас наш наша наше наши не него нее ней нельзя нет ни нибудь никогда ним
    них ничего но ну о об однако он она они оно опять от перед по под после потом потому почти при про
    раз разве с сам свое свои свой свою своя себе себя сейчас сколько со совсем так также такой там
    тебя тем теперь то тогда того тоже той только том тот ту тут ты у уж уже хоть чего чем через что
    чтоб чтобы чуть эта эти этим этих это этого этой этом этот эту я который которая которое которые
    которых которым которой котором
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
