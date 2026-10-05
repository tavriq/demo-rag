import json
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

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
    # keep asking until the worst-case reservation no longer fits into the budget
    statuses = [client.post("/api/ask", json={"question": "отпуск"}).status_code for _ in range(10)]
    assert 429 in statuses
    blocked = client.post("/api/ask", json={"question": "отпуск"})
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
