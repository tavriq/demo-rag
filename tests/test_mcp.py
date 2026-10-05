import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from app.llm import MockGenerator
from app.main import create_app
from app.mcp_server import build_mcp_server, normalize_article_number


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def server(retriever, store):
    return build_mcp_server(retriever, store)


def test_tools_are_listed_read_only(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in _run(go())}
    assert set(tools) == {"search_tk", "get_article"}
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True and tool.description
    assert tools["search_tk"].input_schema["properties"]["limit"]["maximum"] == 10


def test_search_returns_distinct_articles_with_links(server):
    async def go():
        async with Client(server) as client:
            return await client.call_tool("search_tk", {"query": "ежегодный оплачиваемый отпуск", "limit": 3})

    result = _run(go())
    assert not result.is_error
    data = result.structured_content
    articles = [r["article"] for r in data["results"]]
    assert len(articles) == 3 and len(set(articles)) == 3
    first = data["results"][0]
    assert first["title"].startswith("Статья ") and first["source_url"].startswith("https://")
    assert first["fragment"] and first["parts_total"] >= first["part"] >= 1


def test_get_article_returns_full_text_and_rejects_unknown(server, store):
    async def go():
        async with Client(server) as client:
            ok = await client.call_tool("get_article", {"number": "ст. 81"})
            bad = await client.call_tool("get_article", {"number": "9999"})
            return ok, bad

    ok, bad = _run(go())
    assert not ok.is_error
    assert ok.structured_content["article"] == "81"
    assert ok.structured_content["text"] == store.full_text("81")
    assert ok.structured_content["title"].startswith("Статья 81.")
    assert bad.is_error and "search_tk" in bad.content[0].text


@pytest.mark.parametrize("raw, expected", [("81", "81"), ("ст. 312.1", "312.1"), ("Статья 84.1 ТК РФ", "84.1"),
                                           ("нет номера", None)])
def test_normalize_article_number(raw, expected):
    assert normalize_article_number(raw) == expected


def _rpc(method, params=None, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_http_endpoint_answers_json_rpc(settings, retriever, clock):
    app = create_app(settings=replace(settings, mcp_allowed_hosts=("testserver", "rag.example.org")),
                     retriever=retriever, generator=MockGenerator())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["mcp"] == "/mcp"
        init = client.post("/mcp", headers=HEADERS, json=_rpc("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"}}))
        assert init.status_code == 200, init.text
        assert init.json()["result"]["serverInfo"]["name"] == "tk-rf"
        call = client.post("/mcp", headers=HEADERS, json=_rpc("tools/call", {
            "name": "get_article", "arguments": {"number": "115"}}, id_=2))
        assert call.status_code == 200, call.text
        body = call.json()["result"]
        assert body["structuredContent"]["article"] == "115"
        assert json.loads(body["content"][0]["text"])["article"] == "115"
        # DNS-rebinding protection: an unknown Host header is refused
        foreign = client.post("/mcp", headers={**HEADERS, "Host": "evil.example"}, json=_rpc("tools/list"))
        assert foreign.status_code in (403, 421)


def test_http_endpoint_can_be_switched_off(settings, retriever):
    app = create_app(settings=replace(settings, mcp_enabled=False), retriever=retriever, generator=MockGenerator())
    with TestClient(app) as client:
        assert client.get("/api/health").json()["mcp"] is None
        assert client.post("/mcp", headers=HEADERS, json=_rpc("tools/list")).status_code in (404, 405)
