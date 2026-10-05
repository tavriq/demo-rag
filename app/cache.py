"""Answer cache in SQLite: a repeated question is answered for free, without a model call.

The key covers everything that changes the answer: the normalized question,
top_k and a fingerprint of the model, prompt, index and retriever settings.
The cached entry keeps the fragments the answer was generated from, so the
citations always point to the fragments shown next to the answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY,
    created_ts REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS answer_cache_ts ON answer_cache (created_ts);
"""

_SPACES_RE = re.compile(r"\s+")
_TRAILING_RE = re.compile(r"[\s?!.…]+$")


def normalize_question(question: str) -> str:
    text = _SPACES_RE.sub(" ", question.casefold().replace("ё", "е")).strip()
    return _TRAILING_RE.sub("", text)


def fingerprint(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


class AnswerCache:
    def __init__(self, db_path: str | Path, max_entries: int = 2000, clock: Callable[[], float] = time.time):
        self.db_path = Path(db_path)
        self.max_entries = max_entries
        self.clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def make_key(question: str, top_k: int, config_fingerprint: str) -> str:
        raw = f"{config_fingerprint}\n{top_k}\n{normalize_question(question)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM answer_cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, payload: dict) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO answer_cache (key, created_ts, payload) VALUES (?, ?, ?)",
                (key, self.clock(), json.dumps(payload, ensure_ascii=False)),
            )
            # Bounded size: every entry was paid for, so the daily budget already limits growth.
            conn.execute(
                "DELETE FROM answer_cache WHERE key NOT IN "
                "(SELECT key FROM answer_cache ORDER BY created_ts DESC LIMIT ?)",
                (self.max_entries,),
            )

    def __len__(self) -> int:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
