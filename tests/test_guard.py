import sqlite3

import pytest

from app.guard import CostGuard
from app.netutil import client_ip, rate_limit_key
from app.config import Settings, parse_trusted_proxies
from app.pricing import Prices, estimate_input_tokens, estimate_max_tokens


def make_guard(tmp_path, clock, budget=300_000, per_hour=10, hourly=10**9, prices=None):
    return CostGuard(
        tmp_path / "g.sqlite",
        daily_token_budget=budget,
        rate_limit_per_hour=per_hour,
        clock=clock,
        hourly_token_budget=hourly,
        prices=prices,
    )


def test_daily_budget_blocks_when_reservation_would_exceed(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=10_000, per_hour=1000)
    first = guard.check_and_reserve("1.1.1.1", 6000)
    assert first.allowed
    second = guard.check_and_reserve("2.2.2.2", 6000)  # 6000 reserved + 6000 > 10000
    assert not second.allowed
    assert second.code == "budget_exhausted"
    assert "10\u202f000 токенов" in second.message and "исчерпан" in second.message


def test_settle_replaces_estimate_with_actual_tokens(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=10_000, per_hour=1000)
    d = guard.check_and_reserve("1.1.1.1", 6000)
    assert guard.tokens_today() == 6000
    guard.settle(d.reservation_id, input_tokens=1500, output_tokens=200)
    assert guard.tokens_today() == 1700
    assert guard.check_and_reserve("2.2.2.2", 6000).allowed


def test_budget_resets_on_new_utc_day(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=5000, per_hour=1000)
    d = guard.check_and_reserve("1.1.1.1", 5000)
    guard.settle(d.reservation_id, 4000, 1000)
    assert not guard.check_and_reserve("1.1.1.1", 100).allowed
    clock.advance(24 * 3600)
    assert guard.tokens_today() == 0
    assert guard.check_and_reserve("1.1.1.1", 100).allowed


def test_rate_limit_per_ip_with_sliding_hour(tmp_path, clock):
    guard = make_guard(tmp_path, clock, per_hour=3)
    for _ in range(3):
        assert guard.check_and_reserve("10.0.0.1", 0.0, charge=False).allowed
        clock.advance(60)
    blocked = guard.check_and_reserve("10.0.0.1", 0.0, charge=False)
    assert not blocked.allowed
    assert blocked.code == "rate_limited"
    assert 0 < blocked.retry_after_s <= 3600
    assert "3 вопросов в час" in blocked.message
    # another address is not affected
    assert guard.check_and_reserve("10.0.0.2", 0.0, charge=False).allowed
    # once the first request leaves the one-hour window, one more is allowed
    clock.advance(3600 - 3 * 60 + 1)
    assert guard.check_and_reserve("10.0.0.1", 0.0, charge=False).allowed
    assert not guard.check_and_reserve("10.0.0.1", 0.0, charge=False).allowed


def test_mock_requests_do_not_spend_but_count_for_rate_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=1000, per_hour=2)
    assert guard.check_and_reserve("1.1.1.1", 500, charge=False).reservation_id is None
    assert guard.tokens_today() == 0
    assert guard.check_and_reserve("1.1.1.1", 500, charge=False).allowed
    assert not guard.check_and_reserve("1.1.1.1", 500, charge=False).allowed


def test_budget_denied_request_still_counts_for_rate_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=0, per_hour=2)
    assert guard.check_and_reserve("1.1.1.1", 10).code == "budget_exhausted"
    assert guard.check_and_reserve("1.1.1.1", 10).code == "budget_exhausted"
    assert guard.check_and_reserve("1.1.1.1", 10).code == "rate_limited"


def test_ip_is_stored_hashed_only(tmp_path, clock):
    guard = make_guard(tmp_path, clock)
    guard.check_and_reserve("203.0.113.77", 0.0, charge=False)
    with sqlite3.connect(tmp_path / "g.sqlite") as conn:
        dump = "\n".join(conn.iterdump())
    assert "203.0.113.77" not in dump


def test_status_shape(tmp_path, clock):
    status = make_guard(tmp_path, clock, budget=300_000, per_hour=10, hourly=50_000).status()
    assert status == {
        "day_utc": "2026-09-21",
        "tokens_today": 0,
        "daily_token_budget": 300_000,
        "tokens_last_hour": 0,
        "hourly_token_budget": 50_000,
        "rub_today": None,
        "rate_limit_per_hour": 10,
    }


def test_token_estimate_is_input_plus_full_completion():
    # 8000 chars / 2 chars per token + 50 tokens of chat overhead, then the whole completion limit
    assert estimate_input_tokens(8000) == 4050
    assert estimate_max_tokens(8000, 600) == 4650


def test_prices_are_optional_and_never_guessed():
    assert Prices().known is False and Prices().cost_rub(1000, 1000) is None
    assert Prices(100.0, None).cost_rub(1000, 1000) is None
    assert Prices(100.0, 400.0).cost_rub(1_000_000, 500_000) == pytest.approx(300.0)


def test_rubles_today_only_with_prices(tmp_path, clock):
    guard = make_guard(tmp_path, clock, prices=Prices(100.0, 400.0))
    d = guard.check_and_reserve("1.1.1.1", 5000)
    assert guard.status()["rub_today"] == 0  # a reservation is not a cost yet
    guard.settle(d.reservation_id, 10_000, 1000)
    assert guard.status()["rub_today"] == pytest.approx(1.4)  # 10k * 100/M + 1k * 400/M
    assert make_guard(tmp_path, clock).status()["rub_today"] is None


def test_token_settings_from_env():
    s = Settings.from_env({})
    assert s.daily_token_budget == 300_000 and s.hourly_token_budget == 50_000
    assert s.price_rub_per_1m_input is None and s.price_rub_per_1m_output is None
    s = Settings.from_env({"DAILY_TOKEN_BUDGET": "1000", "PRICE_RUB_PER_1M_INPUT": "120",
                           "PRICE_RUB_PER_1M_OUTPUT": "480", "LLM_TEMPERATURE": "none",
                           "LLM_REASONING_EFFORT": "minimal"})
    assert s.daily_token_budget == 1000 and s.price_rub_per_1m_output == 480.0
    assert s.llm_temperature is None and s.llm_reasoning_effort == "minimal"
    with pytest.raises(ValueError):
        Settings.from_env({"LLM_REASONING_EFFORT": "maximal"})


@pytest.mark.parametrize(
    "peer, xff, trusted, expected",
    [
        ("198.51.100.5", "1.2.3.4", "", "198.51.100.5"),  # no proxy trusted: header ignored
        ("198.51.100.5", "1.2.3.4", "10.0.0.1", "198.51.100.5"),  # peer is not the trusted proxy
        ("10.0.0.1", "1.2.3.4", "10.0.0.1", "1.2.3.4"),
        ("10.0.0.1", "6.6.6.6, 1.2.3.4", "10.0.0.1", "1.2.3.4"),  # spoofed left part ignored
        ("172.18.0.1", "1.2.3.4, 172.18.0.5", "172.16.0.0/12", "1.2.3.4"),  # CIDR, chain of proxies
        ("10.0.0.1", "not-an-ip", "10.0.0.1", "10.0.0.1"),
        ("10.0.0.1", None, "10.0.0.1", "10.0.0.1"),
        (None, None, "", "unknown"),
    ],
)
def test_client_ip(peer, xff, trusted, expected):
    assert client_ip(peer, xff, parse_trusted_proxies(trusted)) == expected


def test_hourly_token_budget_for_all_visitors(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=10**9, per_hour=1000, hourly=10_000)
    for i in range(3):
        assert guard.check_and_reserve(f"10.0.0.{i}", 3000).allowed
        clock.advance(60)
    blocked = guard.check_and_reserve("10.0.0.99", 3000)  # 9000 + 3000 > 10000
    assert not blocked.allowed and blocked.code == "global_limited"
    assert "10\u202f000 токенов в час" in blocked.message
    # the first reservation leaves the window 3600 s after it was made, 180 s ago
    assert blocked.retry_after_s == 3600 - 180
    # free (mock or cached) answers are not capped
    assert guard.check_and_reserve("10.0.0.99", 3000, charge=False).allowed
    clock.advance(3600 - 180 + 1)
    assert guard.check_and_reserve("10.0.0.99", 3000).allowed


def test_released_reservations_do_not_count_toward_hourly_budget(tmp_path, clock):
    guard = make_guard(tmp_path, clock, per_hour=1000, hourly=5000)
    first = guard.check_and_reserve("10.0.0.1", 5000)
    guard.release(first.reservation_id)
    assert guard.tokens_today() == 0
    assert guard.check_and_reserve("10.0.0.2", 5000).allowed


def test_reserve_without_client_skips_rate_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, per_hour=1)
    for _ in range(3):
        assert guard.check_and_reserve(None, 10).allowed


@pytest.mark.parametrize(
    "ip, key",
    [
        ("203.0.113.7", "203.0.113.7"),
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2::/64"),
        ("2001:db8:1:3::1", "2001:db8:1:3::/64"),
        ("::ffff:198.51.100.4", "198.51.100.4"),
        ("unknown", "unknown"),
    ],
)
def test_rate_limit_key(ip, key):
    assert rate_limit_key(ip) == key


def test_ipv6_rotation_inside_a_64_hits_the_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, per_hour=10)
    allowed = [guard.check_and_reserve(rate_limit_key(f"2001:db8::{i:x}"), 0.0, charge=False).allowed for i in range(1, 51)]
    assert allowed.count(True) == 10


def test_salt_is_not_stored_and_old_hashes_are_dropped_on_start(tmp_path, clock):
    guard = make_guard(tmp_path, clock)
    guard.check_and_reserve("203.0.113.77", 0.0, charge=False)
    with sqlite3.connect(tmp_path / "g.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM meta WHERE key = 'ip_salt'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
    make_guard(tmp_path, clock)  # restart: new salt, old hashes are useless and removed
    with sqlite3.connect(tmp_path / "g.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


def test_status_purges_rows_older_than_an_hour(tmp_path, clock):
    guard = make_guard(tmp_path, clock)
    guard.check_and_reserve("203.0.113.77", 0.0, charge=False)
    clock.advance(3601)
    guard.status()
    with sqlite3.connect(tmp_path / "g.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
