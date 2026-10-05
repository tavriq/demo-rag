import json
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import parse_trusted_proxies
from app.guard import CostGuard
from app.llm import ClaudeGenerator, MockGenerator
from app.main import create_app


@pytest.fixture
def make_client(settings, retriever, clock):
    def _make(generator=None, **overrides):
        s = replace(settings, **overrides)
        guard = CostGuard(s.guard_db, s.daily_budget_usd, s.rate_limit_per_hour, clock=clock)
        app = create_app(settings=s, retriever=retriever, generator=generator or MockGenerator(), guard=guard)
        return TestClient(app)

    return _make


def test_page_has_no_external_resources(make_client):
    resp = make_client().get("/")
    assert resp.status_code == 200
    html = resp.text
    assert re.search(r'(src|href)="(https?:)?//', html) is None
    assert 'src="/static/app.js"' in html
    assert "Content-Security-Policy" in resp.headers
    assert "script-src 'self'" in resp.headers["Content-Security-Policy"]
    for asset in ("/static/app.js", "/static/app.css"):
        body = make_client().get(asset).text
        assert "http://" not in body and "https://" not in body


def test_health_in_mock_mode(make_client):
    body = make_client().get("/api/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "mock"
    assert body["model"] is None
    assert body["index"]["n_docs"] == 15
    assert body["budget"]["spent_usd"] == 0
    assert body["budget"]["limit_usd"] == 1.0


def test_ask_mock_returns_fragments_and_clickable_citations(make_client):
    resp = make_client().post("/api/ask", json={"question": "Сколько дней длится ежегодный оплачиваемый отпуск?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "mock"
    assert "Демо-режим без ключа: показаны найденные фрагменты" in data["answer"]["text"]
    assert data["fragments"][0]["doc_id"] == "TK-115"
    assert data["fragments"][0]["score"] > 0
    anchors = {f["anchor"] for f in data["fragments"]}
    assert data["answer"]["citations"][0]["found"] is True
    assert data["answer"]["citations"][0]["anchor"] in anchors
    assert 'class="cite"' in data["answer"]["html"]
    assert data["cost_usd"] == 0
    assert data["budget"]["spent_usd"] == 0


def test_ask_validation_and_truncation(make_client):
    client = make_client(max_question_chars=40)
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 6000}).status_code == 422
    data = client.post("/api/ask", json={"question": "отпуск " * 30}).json()
    assert data["truncated"] is True
    assert len(data["question"]) == 40


def test_rate_limit_returns_polite_429(make_client):
    client = make_client(rate_limit_per_hour=2)
    for _ in range(2):
        assert client.post("/api/ask", json={"question": "отпуск"}).status_code == 200
    resp = client.post("/api/ask", json={"question": "отпуск"})
    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"
    assert "вопросов в час" in resp.json()["message"]
    assert int(resp.headers["Retry-After"]) > 0


def test_spoofed_forwarded_for_does_not_bypass_rate_limit(make_client):
    client = make_client(rate_limit_per_hour=1)
    assert client.post("/api/ask", json={"question": "отпуск"}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    resp = client.post("/api/ask", json={"question": "отпуск"}, headers={"X-Forwarded-For": "2.2.2.2"})
    assert resp.status_code == 429


def test_live_mode_spends_from_usage_and_stops_at_budget(make_client, fake_client_factory):
    fake = fake_client_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    client = make_client(generator=gen, daily_budget_usd=0.03, rate_limit_per_hour=100)
    first = client.post("/api/ask", json={"question": "Сколько дней отпуск?"})
    assert first.status_code == 200
    data = first.json()
    assert data["mode"] == "live"
    assert data["cost_usd"] == pytest.approx(0.0035)  # 2000 * $1/M + 300 * $5/M
    assert data["budget"]["spent_usd"] == pytest.approx(0.0035)
    assert data["usage"] == {"input_tokens": 2000, "output_tokens": 300}
    # keep asking new questions until the worst-case reservation no longer fits into the budget
    statuses = [client.post("/api/ask", json={"question": f"отпуск {i}"}).status_code for i in range(10)]
    assert 429 in statuses
    blocked = client.post("/api/ask", json={"question": "отпуск без сохранения"})
    assert blocked.json()["error"] == "budget_exhausted"
    assert blocked.json()["fragments"], "fragments are still shown when the budget is exhausted"
    assert client.get("/api/health").json()["budget"]["spent_usd"] <= 0.03
    assert len(fake.messages.calls) == statuses.index(429) + 1


def test_model_output_is_escaped_in_api(make_client, fake_client_factory):
    fake = fake_client_factory(text='<img src=x onerror="alert(1)"> [ст. 115]')
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    data = make_client(generator=gen).post("/api/ask", json={"question": "<b>отпуск</b>"}).json()
    assert "<img" not in data["answer"]["html"]
    assert "&lt;img" in data["answer"]["html"]
    sent = fake.messages.calls[0]["messages"][0]["content"]
    assert "<b>" not in sent and "&lt;b&gt;отпуск" in sent


def test_api_error_releases_reservation(make_client, retriever):
    import anthropic
    import httpx2

    class Failing:
        mode = "live"
        model = "claude-haiku-4-5-20251001"

        def generate(self, question, hits):
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            response = httpx2.Response(529, request=request)
            raise anthropic.APIStatusError("overloaded", response=response, body=None)

    client = make_client(generator=Failing())
    resp = client.post("/api/ask", json={"question": "отпуск"})
    assert resp.status_code == 502
    assert resp.json()["error"] == "llm_unavailable"
    assert resp.json()["fragments"]
    assert client.get("/api/health").json()["budget"]["spent_usd"] == 0


def test_evals_page_and_json(make_client, settings):
    client = make_client()
    assert "Прогонов ещё не было" in client.get("/evals").text
    assert client.get("/api/evals").status_code == 404
    settings.evals_dir.mkdir(parents=True, exist_ok=True)
    (settings.evals_dir / "latest.json").write_text(
        json.dumps({"date": "2026-10-05", "retrieval": {}, "answers": {"status": "not_run", "reason": "нужен ключ"}}),
        encoding="utf-8",
    )
    page = client.get("/evals").text
    assert "2026-10-05" in page and "Не прогонялось" in page
    assert client.get("/api/evals").json()["date"] == "2026-10-05"


def test_missing_index_degrades_gracefully(settings, tmp_path, clock):
    s = replace(settings, index_dir=tmp_path / "nope")
    guard = CostGuard(s.guard_db, 1.0, 10, clock=clock)
    client = TestClient(create_app(settings=s, generator=MockGenerator(), guard=guard))
    health = client.get("/api/health").json()
    assert health["status"] == "degraded" and health["index"] is None
    assert client.post("/api/ask", json={"question": "отпуск"}).status_code == 503
    assert client.get("/").status_code == 200


def test_oversized_body_is_rejected_before_parsing(make_client):
    client = make_client()
    padded = json.dumps({"question": "отпуск", "pad": "A" * 100_000})
    resp = client.post("/api/ask", content=padded, headers={"Content-Type": "application/json"})
    assert resp.status_code == 413
    assert resp.json()["error"] == "too_large"

    def chunked():  # no Content-Length: the stream is cut off by the running total
        yield b'{"question": "'
        for _ in range(40):
            yield b"A" * 1024
        yield b'"}'

    resp = client.post("/api/ask", content=chunked(), headers={"Content-Type": "application/json"})
    assert resp.status_code == 413


def test_unknown_fields_are_rejected(make_client):
    resp = make_client().post("/api/ask", json={"question": "отпуск", "pad": "x"})
    assert resp.status_code == 422


def test_body_within_limit_with_escaped_cyrillic_is_accepted(make_client):
    # 2000 Cyrillic characters sent as \uXXXX escapes: the largest legitimate body
    body = json.dumps({"question": "о" * 2000}, ensure_ascii=True)
    assert len(body) > 12_000
    resp = make_client().post("/api/ask", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["truncated"] is True


def test_repeated_question_is_answered_from_cache_for_free(make_client, fake_client_factory):
    fake = fake_client_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    client = make_client(generator=gen, rate_limit_per_hour=100)
    first = client.post("/api/ask", json={"question": "Сколько дней отпуск?"}).json()
    assert first["cached"] is False and first["cost_usd"] > 0
    again = client.post("/api/ask", json={"question": "  сколько  дней ОТПУСК "}).json()
    assert again["cached"] is True
    assert again["cost_usd"] == 0 and again["usage"] is None
    assert again["answer"]["text"] == first["answer"]["text"]
    assert again["fragments"] == first["fragments"]
    assert again["answer"]["citations"][0]["found"] is True
    assert again["budget"]["spent_usd"] == pytest.approx(first["budget"]["spent_usd"])
    assert len(fake.messages.calls) == 1
    # another top_k is another context, so another answer
    other = client.post("/api/ask", json={"question": "Сколько дней отпуск?", "top_k": 3}).json()
    assert other["cached"] is False and len(fake.messages.calls) == 2


def test_cached_answer_is_served_when_budget_is_exhausted(make_client, fake_client_factory):
    fake = fake_client_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    client = make_client(generator=gen, daily_budget_usd=0.012, rate_limit_per_hour=100)
    assert client.post("/api/ask", json={"question": "Сколько дней отпуск?"}).status_code == 200
    statuses = [client.post("/api/ask", json={"question": f"вопрос {i}"}).status_code for i in range(5)]
    assert statuses[-1] == 429
    cached = client.post("/api/ask", json={"question": "Сколько дней отпуск?"})
    assert cached.status_code == 200 and cached.json()["cached"] is True


def test_global_hourly_cap_applies_across_clients(settings, retriever, clock, fake_client_factory):
    fake = fake_client_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    s = replace(settings, trusted_proxies=parse_trusted_proxies("10.0.0.1"), rate_limit_per_hour=100)
    guard = CostGuard(s.guard_db, 1.0, 100, clock=clock, global_paid_per_hour=2)
    app = create_app(settings=s, retriever=retriever, generator=gen, guard=guard)
    client = TestClient(app, client=("10.0.0.1", 50000))  # the trusted proxy
    codes = []
    for i in range(3):
        resp = client.post("/api/ask", json={"question": f"отпуск {i}"}, headers={"X-Forwarded-For": f"203.0.113.{i}"})
        codes.append(resp.status_code)
    assert codes == [200, 200, 429]
    assert resp.json()["error"] == "global_limited"
    assert resp.json()["fragments"]
    assert int(resp.headers["Retry-After"]) > 0


def test_ipv6_clients_share_a_bucket_per_64(make_client):
    from app.netutil import rate_limit_key

    keys = {rate_limit_key(f"2001:db8:0:0::{i:x}") for i in range(1, 51)}
    assert keys == {"2001:db8::/64"}


def test_warm_up_answers_examples_once(settings, retriever, clock, fake_client_factory):
    from app.examples import ui_examples

    fake = fake_client_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    gen = ClaudeGenerator(model="claude-haiku-4-5-20251001", max_tokens=600, client=fake)
    guard = CostGuard(settings.guard_db, 1.0, 10, clock=clock)
    app = create_app(settings=settings, retriever=retriever, generator=gen, guard=guard)
    examples = ui_examples()
    assert examples
    with TestClient(app) as client:
        app.state.warmup_thread.join(timeout=30)
        assert len(fake.messages.calls) == len(examples)
        data = client.post("/api/ask", json={"question": examples[0]}).json()
        assert data["cached"] is True and len(fake.messages.calls) == len(examples)


def test_forwarded_for_without_trusted_proxy_is_flagged(make_client):
    client = make_client()
    assert client.get("/api/health").json()["proxy_headers_ignored"] is False
    client.post("/api/ask", json={"question": "отпуск"}, headers={"X-Forwarded-For": "1.2.3.4"})
    assert client.get("/api/health").json()["proxy_headers_ignored"] is True
