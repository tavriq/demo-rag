import sqlite3

import pytest

from app.guard import CostGuard
from app.netutil import client_ip, rate_limit_key
from app.config import parse_trusted_proxies
from app.pricing import cost_usd, estimate_max_cost


def make_guard(tmp_path, clock, budget=1.0, per_hour=10):
    return CostGuard(tmp_path / "g.sqlite", daily_budget_usd=budget, rate_limit_per_hour=per_hour, clock=clock)


def test_daily_budget_blocks_when_reservation_would_exceed(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=0.10, per_hour=1000)
    first = guard.check_and_reserve("1.1.1.1", 0.06)
    assert first.allowed
    second = guard.check_and_reserve("2.2.2.2", 0.06)  # 0.06 reserved + 0.06 > 0.10
    assert not second.allowed
    assert second.code == "budget_exhausted"
    assert "бюджет" in second.message


def test_settle_replaces_estimate_with_actual_cost(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=0.10, per_hour=1000)
    d = guard.check_and_reserve("1.1.1.1", 0.06)
    assert guard.spent_today() == pytest.approx(0.06)
    guard.settle(d.reservation_id, 0.004, input_tokens=2000, output_tokens=400)
    assert guard.spent_today() == pytest.approx(0.004)
    assert guard.check_and_reserve("2.2.2.2", 0.06).allowed


def test_budget_resets_on_new_utc_day(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=0.05, per_hour=1000)
    d = guard.check_and_reserve("1.1.1.1", 0.05)
    guard.settle(d.reservation_id, 0.05)
    assert not guard.check_and_reserve("1.1.1.1", 0.01).allowed
    clock.advance(24 * 3600)
    assert guard.spent_today() == 0
    assert guard.check_and_reserve("1.1.1.1", 0.01).allowed


def test_rate_limit_per_ip_with_sliding_hour(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=100, per_hour=3)
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
    guard = make_guard(tmp_path, clock, budget=1.0, per_hour=2)
    assert guard.check_and_reserve("1.1.1.1", 0.5, charge=False).reservation_id is None
    assert guard.spent_today() == 0
    assert guard.check_and_reserve("1.1.1.1", 0.5, charge=False).allowed
    assert not guard.check_and_reserve("1.1.1.1", 0.5, charge=False).allowed


def test_budget_denied_request_still_counts_for_rate_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=0.0, per_hour=2)
    assert guard.check_and_reserve("1.1.1.1", 0.01).code == "budget_exhausted"
    assert guard.check_and_reserve("1.1.1.1", 0.01).code == "budget_exhausted"
    assert guard.check_and_reserve("1.1.1.1", 0.01).code == "rate_limited"


def test_ip_is_stored_hashed_only(tmp_path, clock):
    guard = make_guard(tmp_path, clock)
    guard.check_and_reserve("203.0.113.77", 0.0, charge=False)
    with sqlite3.connect(tmp_path / "g.sqlite") as conn:
        dump = "\n".join(conn.iterdump())
    assert "203.0.113.77" not in dump


def test_status_shape(tmp_path, clock):
    status = make_guard(tmp_path, clock, budget=1.0, per_hour=10).status()
    assert status == {
        "day_utc": "2026-09-21",
        "spent_usd": 0.0,
        "limit_usd": 1.0,
        "rate_limit_per_hour": 10,
        "global_paid_per_hour": 20,
    }


def test_cost_from_usage_haiku_prices():
    # $1 / MTok input, $5 / MTok output
    assert cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)
    assert cost_usd("claude-haiku-4-5-20251001", 1000, 200) == pytest.approx(0.002)
    assert cost_usd("claude-haiku-4-5-20251001", 0, 0, 1_000_000, 1_000_000) == pytest.approx(1.35)
    with pytest.raises(KeyError):
        cost_usd("unknown-model", 1, 1)


def test_estimate_is_an_upper_bound_for_typical_prompt():
    est = estimate_max_cost("claude-haiku-4-5-20251001", prompt_chars=8000, max_tokens=600)
    # 4050 input tokens * $1/M + 600 output * $5/M
    assert est == pytest.approx((4050 * 1 + 600 * 5) / 1_000_000)


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


def test_global_hourly_cap_on_paid_answers(tmp_path, clock):
    guard = CostGuard(tmp_path / "g.sqlite", 100.0, 1000, clock=clock, global_paid_per_hour=3)
    for i in range(3):
        assert guard.check_and_reserve(f"10.0.0.{i}", 0.001).allowed
    blocked = guard.check_and_reserve("10.0.0.99", 0.001)
    assert not blocked.allowed and blocked.code == "global_limited"
    assert 0 < blocked.retry_after_s <= 3600
    # free (mock or cached) answers are not capped
    assert guard.check_and_reserve("10.0.0.99", 0.0, charge=False).allowed
    clock.advance(3601)
    assert guard.check_and_reserve("10.0.0.99", 0.001).allowed


def test_released_reservations_do_not_count_toward_global_cap(tmp_path, clock):
    guard = CostGuard(tmp_path / "g.sqlite", 100.0, 1000, clock=clock, global_paid_per_hour=1)
    first = guard.check_and_reserve("10.0.0.1", 0.01)
    guard.release(first.reservation_id)
    assert guard.spent_today() == 0
    assert guard.check_and_reserve("10.0.0.2", 0.01).allowed


def test_reserve_without_client_skips_rate_limit(tmp_path, clock):
    guard = make_guard(tmp_path, clock, budget=1.0, per_hour=1)
    for _ in range(3):
        assert guard.check_and_reserve(None, 0.001).allowed


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
    guard = make_guard(tmp_path, clock, budget=100, per_hour=10)
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
