"""Cost guard: hard daily USD budget + per-IP hourly rate limit, stored in SQLite.

Flow for a paid request:
  1. ``check_and_reserve`` — inside one write transaction: check the IP rate
     limit, check that spent_today + worst-case estimate <= budget, record the
     request and reserve the estimate in the ledger.
  2. after the API call, ``settle`` replaces the reservation with the real cost
     computed from ``usage``. A failed call settles at its real cost (0 if the
     API returned nothing).
Reserving the worst case before the call keeps concurrent requests from
overshooting the budget together.

IP addresses are stored only as salted HMAC hashes and only for one hour.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

WINDOW_SECONDS = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests (
    ip_hash TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_ip_ts ON requests (ip_hash, ts);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    ts REAL NOT NULL,
    usd REAL NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS ledger_day ON ledger (day);
"""


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    code: str | None = None
    message: str | None = None
    reservation_id: int | None = None
    retry_after_s: int | None = None


class CostGuard:
    def __init__(
        self,
        db_path: str | Path,
        daily_budget_usd: float,
        rate_limit_per_hour: int,
        clock: Callable[[], float] = time.time,
    ):
        self.db_path = Path(db_path)
        self.daily_budget_usd = daily_budget_usd
        self.rate_limit_per_hour = rate_limit_per_hour
        self.clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key = 'ip_salt'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('ip_salt', ?)", (secrets.token_hex(16),))
                row = conn.execute("SELECT value FROM meta WHERE key = 'ip_salt'").fetchone()
        self._salt = bytes.fromhex(row[0])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _day(self, ts: float | None = None) -> str:
        ts = self.clock() if ts is None else ts
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def hash_ip(self, ip: str) -> str:
        return hmac.new(self._salt, ip.encode("utf-8"), hashlib.sha256).hexdigest()

    def check_and_reserve(self, ip: str, estimated_usd: float, charge: bool = True) -> GuardDecision:
        """Apply the rate limit (always) and the budget (when ``charge``); reserve on success."""
        now = self.clock()
        day = self._day(now)
        ip_hash = self.hash_ip(ip)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM requests WHERE ts <= ?", (now - WINDOW_SECONDS,))
            count, oldest = conn.execute(
                "SELECT COUNT(*), MIN(ts) FROM requests WHERE ip_hash = ? AND ts > ?",
                (ip_hash, now - WINDOW_SECONDS),
            ).fetchone()
            if count >= self.rate_limit_per_hour:
                conn.execute("COMMIT")
                retry = max(1, int(oldest + WINDOW_SECONDS - now) + 1)
                minutes = max(1, (retry + 59) // 60)
                return GuardDecision(
                    allowed=False,
                    code="rate_limited",
                    message=(
                        f"Лимит демо: не больше {self.rate_limit_per_hour} вопросов в час с одного адреса. "
                        f"Попробуйте через {minutes} мин."
                    ),
                    retry_after_s=retry,
                )
            # Every processed question counts toward the rate limit, paid or not.
            conn.execute("INSERT INTO requests (ip_hash, ts) VALUES (?, ?)", (ip_hash, now))
            reservation_id = None
            if charge:
                spent = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE day = ?", (day,)).fetchone()[0]
                if spent + estimated_usd > self.daily_budget_usd:
                    conn.execute("COMMIT")
                    return GuardDecision(
                        allowed=False,
                        code="budget_exhausted",
                        message=(
                            f"Дневной бюджет демо (${self.daily_budget_usd:.2f}) исчерпан, ответ модели "
                            "недоступен до 00:00 UTC. Найденные фрагменты показаны ниже."
                        ),
                    )
                cur = conn.execute(
                    "INSERT INTO ledger (day, ts, usd, status) VALUES (?, ?, ?, 'reserved')",
                    (day, now, estimated_usd),
                )
                reservation_id = cur.lastrowid
            conn.execute("COMMIT")
            return GuardDecision(allowed=True, reservation_id=reservation_id)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def settle(
        self,
        reservation_id: int,
        actual_usd: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE ledger SET usd = ?, status = 'settled', input_tokens = ?, output_tokens = ? WHERE id = ?",
                (actual_usd, input_tokens, output_tokens, reservation_id),
            )

    def spent_today(self) -> float:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM ledger WHERE day = ?", (self._day(),)).fetchone()
        return float(row[0])

    def status(self) -> dict:
        return {
            "day_utc": self._day(),
            "spent_usd": round(self.spent_today(), 6),
            "limit_usd": self.daily_budget_usd,
            "rate_limit_per_hour": self.rate_limit_per_hour,
        }
