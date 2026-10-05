import json
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import parse_trusted_proxies
from app.guard import CostGuard
from app.llm import LLMTransportError, MockGenerator
from app.main import create_app


@pytest.fixture
def make_client(settings, retriever, clock):
    def _make(generator=None, **overrides):
        s = replace(settings, **overrides)
        guard = CostGuard(
            s.guard_db, s.daily_token_budget, s.rate_limit_per_hour, clock=clock,
            hourly_token_budget=s.hourly_token_budget,
        )
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
    assert body["budget"]["tokens_today"] == 0
    assert body["budget"]["daily_token_budget"] == 300_000
    assert body["search_mode"] in ("bm25", "dense", "hybrid") and body["article_router"] is True


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
    assert data["tokens"] == 0
    assert data["budget"]["tokens_today"] == 0


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


def test_live_mode_spends_from_usage_and_stops_at_budget(make_client, fake_llm_factory):
    gen, gateway = fake_llm_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    client = make_client(generator=gen, daily_token_budget=12_000, rate_limit_per_hour=100)
    first = client.post("/api/ask", json={"question": "Сколько дней отпуск?"})
    assert first.status_code == 200
    data = first.json()
    assert data["mode"] == "live"
    assert data["tokens"] == 2300
    assert data["cost_rub"] is None  # no prices configured: tokens only
    assert data["budget"]["tokens_today"] == 2300
    assert data["usage"] == {"input_tokens": 2000, "output_tokens": 300, "reasoning_tokens": 0}
    # keep asking new questions until the worst-case reservation no longer fits into the budget
    statuses = [client.post("/api/ask", json={"question": f"отпуск {i}"}).status_code for i in range(10)]
    assert 429 in statuses
    blocked = client.post("/api/ask", json={"question": "отпуск без сохранения"})
    assert blocked.json()["error"] == "budget_exhausted"
    assert blocked.json()["fragments"], "fragments are still shown when the budget is exhausted"
    assert client.get("/api/health").json()["budget"]["tokens_today"] <= 12_000
    assert len(gateway.calls) == statuses.index(429) + 1


def test_rubles_shown_only_when_prices_are_set(settings, retriever, clock, fake_llm_factory):
    from app.pricing import Prices

    gen, _ = fake_llm_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    guard = CostGuard(settings.guard_db, 300_000, 10, clock=clock, prices=Prices(100.0, 400.0))
    s = replace(settings, price_rub_per_1m_input=100.0, price_rub_per_1m_output=400.0)
    client = TestClient(create_app(settings=s, retriever=retriever, generator=gen, guard=guard))
    data = client.post("/api/ask", json={"question": "отпуск"}).json()
    assert data["cost_rub"] == pytest.approx(0.14)  # 1000 * 100/M + 100 * 400/M
    assert data["budget"]["rub_today"] == pytest.approx(0.14)


def test_model_output_is_escaped_in_api(make_client, fake_llm_factory):
    gen, gateway = fake_llm_factory(text='<img src=x onerror="alert(1)"> [ст. 115]')
    data = make_client(generator=gen).post("/api/ask", json={"question": "<b>отпуск</b>"}).json()
    assert "<img" not in data["answer"]["html"]
    assert "&lt;img" in data["answer"]["html"]
    sent = gateway.calls[0]["messages"][1]["content"]
    assert "<b>" not in sent and "&lt;b&gt;отпуск" in sent


@pytest.mark.parametrize("status, released", [(400, True), (429, True), (503, False)])
def test_gateway_error_status(make_client, fake_llm_factory, status, released):
    gen, _ = fake_llm_factory(status=status)
    client = make_client(generator=gen)
    resp = client.post("/api/ask", json={"question": "отпуск"})
    assert resp.status_code == 502
    assert resp.json()["error"] == "llm_unavailable"
    assert resp.json()["fragments"]
    tokens = client.get("/api/health").json()["budget"]["tokens_today"]
    # a rejected request is not billed; on 5xx the worst-case reservation stays
    assert (tokens == 0) is released


def test_transport_error_keeps_reservation_and_logs_no_url(make_client, retriever, caplog):
    class Failing:
        mode = "live"
        model = "test/model"

        def generate(self, question, hits):
            raise LLMTransportError("ReadTimeout")

    client = make_client(generator=Failing())
    with caplog.at_level("INFO"):
        resp = client.post("/api/ask", json={"question": "отпуск"})
    assert resp.status_code == 502
    assert client.get("/api/health").json()["budget"]["tokens_today"] > 0
    assert "LLMTransportError" in caplog.text and "http" not in caplog.text.lower().replace("httpx", "")


def test_missing_usage_keeps_worst_case_reservation(make_client, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Ответ [ст. 115].", usage=False)
    client = make_client(generator=gen)
    assert client.post("/api/ask", json={"question": "отпуск"}).status_code == 200
    assert client.get("/api/health").json()["budget"]["tokens_today"] > 1000


def test_truncated_answer_is_not_cached(make_client, fake_llm_factory):
    gen, gateway = fake_llm_factory(text="Длинный", finish_reason="length")
    client = make_client(generator=gen, rate_limit_per_hour=100)
    client.post("/api/ask", json={"question": "отпуск"})
    again = client.post("/api/ask", json={"question": "отпуск"}).json()
    assert again["cached"] is False and len(gateway.calls) == 2


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
    guard = CostGuard(s.guard_db, 300_000, 10, clock=clock)
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


def test_repeated_question_is_answered_from_cache_for_free(make_client, fake_llm_factory):
    gen, gateway = fake_llm_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    client = make_client(generator=gen, rate_limit_per_hour=100)
    first = client.post("/api/ask", json={"question": "Сколько дней отпуск?"}).json()
    assert first["cached"] is False and first["tokens"] > 0
    again = client.post("/api/ask", json={"question": "  сколько  дней ОТПУСК "}).json()
    assert again["cached"] is True
    assert again["tokens"] == 0 and again["usage"] is None
    assert again["answer"]["text"] == first["answer"]["text"]
    assert again["fragments"] == first["fragments"]
    assert again["answer"]["citations"][0]["found"] is True
    assert again["budget"]["tokens_today"] == first["budget"]["tokens_today"]
    assert len(gateway.calls) == 1
    # another top_k is another context, so another answer
    other = client.post("/api/ask", json={"question": "Сколько дней отпуск?", "top_k": 3}).json()
    assert other["cached"] is False and len(gateway.calls) == 2


def test_cached_answer_is_served_when_budget_is_exhausted(make_client, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Отпуск 28 дней [ст. 115].", input_tokens=2000, output_tokens=300)
    client = make_client(generator=gen, daily_token_budget=15000, rate_limit_per_hour=100)
    assert client.post("/api/ask", json={"question": "Сколько дней отпуск?"}).status_code == 200
    statuses = [client.post("/api/ask", json={"question": f"вопрос {i}"}).status_code for i in range(5)]
    assert statuses[-1] == 429
    cached = client.post("/api/ask", json={"question": "Сколько дней отпуск?"})
    assert cached.status_code == 200 and cached.json()["cached"] is True


def test_hourly_token_budget_applies_across_clients(settings, retriever, clock, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Ответ [ст. 115].", input_tokens=5000, output_tokens=100)
    s = replace(settings, trusted_proxies=parse_trusted_proxies("10.0.0.1"), rate_limit_per_hour=100)
    guard = CostGuard(s.guard_db, 10**9, 100, clock=clock, hourly_token_budget=20000)
    app = create_app(settings=s, retriever=retriever, generator=gen, guard=guard)
    client = TestClient(app, client=("10.0.0.1", 50000))  # the trusted proxy
    codes = []
    for i in range(5):
        resp = client.post("/api/ask", json={"question": f"отпуск {i}"}, headers={"X-Forwarded-For": f"203.0.113.{i}"})
        codes.append(resp.status_code)
    # every visitor has a different address, so only the shared hourly budget can stop them
    assert codes[0] == 200 and 429 in codes
    assert all(code == 429 for code in codes[codes.index(429):])
    assert resp.json()["error"] == "global_limited"
    assert resp.json()["fragments"]
    assert int(resp.headers["Retry-After"]) > 0


def test_ipv6_clients_share_a_bucket_per_64(make_client):
    from app.netutil import rate_limit_key

    keys = {rate_limit_key(f"2001:db8:0:0::{i:x}") for i in range(1, 51)}
    assert keys == {"2001:db8::/64"}


def test_warm_up_answers_examples_once(settings, retriever, clock, fake_llm_factory):
    from app.examples import ui_examples

    gen, gateway = fake_llm_factory(text="Ответ [ст. 115].", input_tokens=1000, output_tokens=100)
    guard = CostGuard(settings.guard_db, 300_000, 10, clock=clock)
    app = create_app(settings=settings, retriever=retriever, generator=gen, guard=guard)
    examples = ui_examples()
    assert examples
    with TestClient(app) as client:
        app.state.warmup_thread.join(timeout=30)
        assert len(gateway.calls) == len(examples)
        data = client.post("/api/ask", json={"question": examples[0]}).json()
        assert data["cached"] is True and len(gateway.calls) == len(examples)


def test_question_with_article_number_pins_that_article(make_client):
    data = make_client().post("/api/ask", json={"question": "Что сказано в ст. 81?"}).json()
    assert data["fragments"][0]["article"] == "81" and data["fragments"][0]["pinned"] is True
    assert all(not f["pinned"] for f in data["fragments"] if f["article"] != "81")


def test_forwarded_for_without_trusted_proxy_is_flagged(make_client):
    client = make_client()
    assert client.get("/api/health").json()["proxy_headers_ignored"] is False
    client.post("/api/ask", json={"question": "отпуск"}, headers={"X-Forwarded-For": "1.2.3.4"})
    assert client.get("/api/health").json()["proxy_headers_ignored"] is True


def _sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_live_answer_has_articles_and_checks(make_client, fake_llm_factory):
    text = "Коротко: увольнение по желанию [ст. 80], предупредить за 2 недели [ст. 80], а не за 30 дней [ст. 80] [ст. 999]."
    gen, gateway = fake_llm_factory(text=text)
    data = make_client(generator=gen).post("/api/ask", json={"question": "статья 80"}).json()
    assert data["articles"][0]["article"] == "80" and data["articles"][0]["coverage"] == "целиком"
    assert data["articles"][0]["anchor"].startswith("frag-TK-80")
    checks = data["answer"]["checks"]
    assert checks["removed"] == ["999"] and checks["numbers_missing"] == ["30"]
    assert "[ст. 999]" not in data["answer"]["text"]
    assert 'class="num-unverified"' in data["answer"]["html"]
    assert data["answer"]["citations"][0]["source_url"].startswith("https://")
    # the model read whole articles, not 350-character fragments
    user_msg = gateway.calls[0]["messages"][1]["content"]
    assert '<article number="80"' in user_msg and "<fragment" not in user_msg


def test_stream_endpoint_sends_meta_deltas_and_done(make_client, fake_llm_factory):
    gen, _ = fake_llm_factory(text="Коротко: отпуск 28 календарных дней [ст. 115].", input_tokens=3000,
                              output_tokens=100)
    client = make_client(generator=gen)
    resp = client.post("/api/ask/stream", json={"question": "Сколько дней отпуск?"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-accel-buffering"] == "no"
    events = _sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and names.count("delta") > 1
    assert events[0][1]["articles"]
    streamed = "".join(data["t"] for name, data in events if name == "delta")
    done = events[-1][1]
    assert done["answer"]["text"].startswith(streamed.strip())
    assert done["tokens"] == 3100 and done["budget"]["tokens_today"] == 3100
    # the same question again: from the cache, no deltas, still meta + done
    again = _sse(client.post("/api/ask/stream", json={"question": "Сколько дней отпуск?"}).text)
    assert [n for n, _ in again] == ["meta", "done"] and again[-1][1]["cached"] is True
    assert again[-1][1]["articles"] == done["articles"]


def test_stream_endpoint_reports_errors_as_events(make_client, fake_llm_factory):
    gen, _ = fake_llm_factory(status=503)
    events = _sse(make_client(generator=gen).post("/api/ask/stream", json={"question": "отпуск"}).text)
    assert events[-1][0] == "error" and events[-1][1]["status"] == 502
    assert events[-1][1]["articles"]
    client = make_client(rate_limit_per_hour=1)
    client.post("/api/ask/stream", json={"question": "отпуск"})
    limited = _sse(client.post("/api/ask/stream", json={"question": "отпуск 2"}).text)
    assert limited == [("error", limited[0][1])] and limited[0][1]["status"] == 429
    assert client.post("/api/ask/stream", json={"question": "   "}).status_code == 400
