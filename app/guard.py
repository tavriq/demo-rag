"""Cost guard: daily and hourly token budgets for all visitors and a per-client
hourly rate limit, stored in SQLite.

Flow for a paid request:
  1. ``check_and_reserve`` — inside one write transaction: check the client rate
     limit, that tokens used in the last hour + the worst-case estimate fit the
     hourly budget and that tokens used today + the estimate fit the daily
     budget, then record the request and reserve the estimate in the ledger.
  2. after the API call, ``settle`` replaces the reservation with the real
     token count from ``usage``; ``release`` drops it when the gateway rejected
     the request.
Reserving the worst case before the call keeps concurrent requests from
overshooting the budget together. The real count can still exceed the estimate
if the characters-per-token guess is off, so the limit holds to within the
requests that run at the same time, not to the exact token.

Client keys (IPv4 address or IPv6 /64) are stored only as HMAC hashes. The salt
lives in process memory and is regenerated on every start, so the hashes cannot
be reversed from the database file alone. Rows older than an hour are deleted
on start, on every request and on every status read.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.pricing import Prices

WINDOW_SECONDS = 3600

# The dollar ledger of earlier versions ("ledger") is left as is and no longer read.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests (
    ip_hash TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_ip_ts ON requests (ip_hash, ts);
CREATE TABLE IF NOT EXISTS token_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    ts REAL NOT NULL,
    tokens INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS token_ledger_day ON token_ledger (day);
CREATE INDEX IF NOT EXISTS token_ledger_ts ON token_ledger (ts);
"""


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    code: str | None = None
    message: str | None = None
    reservation_id: int | None = None
    retry_after_s: int | None = None


def _minutes(seconds: int) -> int:
    return max(1, (seconds + 59) // 60)


def _int_ru(value: int) -> str:
    """300000 -> "300 000" (narrow no-break space, as in Russian typography)."""
    return f"{value:,}".replace(",", " ")


class CostGuard:
    def __init__(
        self,
        db_path: str | Path,
        daily_token_budget: int,
        rate_limit_per_hour: int,
        clock: Callable[[], float] = time.time,
        hourly_token_budget: int = 50_000,
        prices: Prices | None = None,
    ):
        self.db_path = Path(db_path)
        self.daily_token_budget = daily_token_budget
        self.hourly_token_budget = hourly_token_budget
        self.rate_limit_per_hour = rate_limit_per_hour
        self.prices = prices or Prices()
        self.clock = clock
        self._salt = secrets.token_bytes(16)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            # Hashes made with a previous process's salt are useless and only keep personal data around.
            conn.execute("DELETE FROM requests")
            conn.execute("DELETE FROM meta WHERE key = 'ip_salt'")  # left by older versions

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _day(self, ts: float | None = None) -> str:
        ts = self.clock() if ts is None else ts
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    def hash_ip(self, ip: str) -> str:
        return hmac.new(self._salt, ip.encode("utf-8"), hashlib.sha256).hexdigest()

    def _hourly_retry_after(self, conn: sqlite3.Connection, now: float, estimated: int) -> int:
        """Seconds until enough of the last hour's tokens leave the window for this request to fit."""
        rows = conn.execute(
            "SELECT ts, tokens FROM token_ledger WHERE ts > ? AND status != 'released' ORDER BY ts",
            (now - WINDOW_SECONDS,),
        ).fetchall()
        used = sum(tokens for _, tokens in rows)
        for ts, tokens in rows:
            used -= tokens
            if used + estimated <= self.hourly_token_budget:
                return max(1, math.ceil(ts + WINDOW_SECONDS - now))
        return WINDOW_SECONDS

    def check_and_reserve(self, ip: str | None, estimated_tokens: int, charge: bool = True) -> GuardDecision:
        """Apply the client rate limit (unless ``ip`` is None) and, when ``charge``,
        the hourly and daily token budgets; reserve the estimate on success."""
        now = self.clock()
        day = self._day(now)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM requests WHERE ts <= ?", (now - WINDOW_SECONDS,))
            if ip is not None:
                ip_hash = self.hash_ip(ip)
                count, oldest = conn.execute(
                    "SELECT COUNT(*), MIN(ts) FROM requests WHERE ip_hash = ? AND ts > ?",
                    (ip_hash, now - WINDOW_SECONDS),
                ).fetchone()
                if count >= self.rate_limit_per_hour:
                    conn.execute("COMMIT")
                    retry = max(1, math.ceil(oldest + WINDOW_SECONDS - now))
                    return GuardDecision(
                        allowed=False,
                        code="rate_limited",
                        message=(
                            f"Лимит демо: не больше {self.rate_limit_per_hour} вопросов в час с одного адреса. "
                            f"Попробуйте через {_minutes(retry)} мин."
                        ),
                        retry_after_s=retry,
                    )
                # Every processed question counts toward the rate limit, paid or not.
                conn.execute("INSERT INTO requests (ip_hash, ts) VALUES (?, ?)", (ip_hash, now))
            reservation_id = None
            if charge:
                last_hour = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger WHERE ts > ? AND status != 'released'",
                    (now - WINDOW_SECONDS,),
                ).fetchone()[0]
                if last_hour + estimated_tokens > self.hourly_token_budget:
                    retry = self._hourly_retry_after(conn, now, estimated_tokens)
                    conn.execute("COMMIT")
                    return GuardDecision(
                        allowed=False,
                        code="global_limited",
                        message=(
                            f"Демо тратит на ответы модели не больше {_int_ru(self.hourly_token_budget)} токенов "
                            f"в час на всех посетителей. Попробуйте через {_minutes(retry)} мин. Примеры со "
                            "страницы работают и сейчас. Найденные фрагменты показаны ниже."
                        ),
                        retry_after_s=retry,
                    )
                today = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger WHERE day = ?", (day,)
                ).fetchone()[0]
                if today + estimated_tokens > self.daily_token_budget:
                    conn.execute("COMMIT")
                    return GuardDecision(
                        allowed=False,
                        code="budget_exhausted",
                        message=(
                            f"Дневной лимит демо ({_int_ru(self.daily_token_budget)} токенов) исчерпан, ответ "
                            "модели на новые вопросы недоступен до 00:00 UTC. Примеры со страницы работают и "
                            "сейчас. Найденные фрагменты показаны ниже."
                        ),
                    )
                cur = conn.execute(
                    "INSERT INTO token_ledger (day, ts, tokens, status) VALUES (?, ?, ?, 'reserved')",
                    (day, now, int(estimated_tokens)),
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

    def settle(self, reservation_id: int, input_tokens: int, output_tokens: int) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE token_ledger SET tokens = ?, status = 'settled', input_tokens = ?, output_tokens = ? "
                "WHERE id = ?",
                (int(input_tokens) + int(output_tokens), input_tokens, output_tokens, reservation_id),
            )

    def release(self, reservation_id: int) -> None:
        """The gateway rejected the request (4xx): such a request is not billed."""
        with closing(self._connect()) as conn:
            conn.execute("UPDATE token_ledger SET tokens = 0, status = 'released' WHERE id = ?", (reservation_id,))

    def purge_expired(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM requests WHERE ts <= ?", (self.clock() - WINDOW_SECONDS,))

    def tokens_today(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger WHERE day = ?", (self._day(),)
            ).fetchone()
        return int(row[0])

    def _rub_today(self) -> float | None:
        if not self.prices.known:
            return None
        with closing(self._connect()) as conn:
            tokens_in, tokens_out = conn.execute(
                "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) FROM token_ledger "
                "WHERE day = ? AND status = 'settled'",
                (self._day(),),
            ).fetchone()
        return self.prices.cost_rub(tokens_in, tokens_out)

    def status(self) -> dict:
        self.purge_expired()
        now = self.clock()
        with closing(self._connect()) as conn:
            last_hour = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger WHERE ts > ? AND status != 'released'",
                (now - WINDOW_SECONDS,),
            ).fetchone()[0]
        rub = self._rub_today()
        return {
            "day_utc": self._day(),
            "tokens_today": self.tokens_today(),
            "daily_token_budget": self.daily_token_budget,
            "tokens_last_hour": int(last_hour),
            "hourly_token_budget": self.hourly_token_budget,
            "rub_today": None if rub is None else round(rub, 2),
            "rate_limit_per_hour": self.rate_limit_per_hour,
        }
