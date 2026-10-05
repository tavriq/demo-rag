"""MCP server over the same index: Claude Code (or any MCP client) searches the Labour Code itself.

Two read-only tools, no language model and no tokens spent:
  * search_tk(query, limit) — semantic search, distinct articles with the matching fragment;
  * get_article(number) — the full text of one article with its title and source link.

Transports:
  * stdio, for a local checkout:  claude mcp add tk-rf -- python3 -m app.mcp_server
  * Streamable HTTP inside the web app at /mcp (stateless, JSON responses), see app/main.py:
      claude mcp add --transport http tk-rf https://rag.tavriq.ru/mcp
"""

from __future__ import annotations

import logging
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from app.context import ArticleStore, base_header
from app.llm import ARTICLE_NUMBER_RE
from app.retrieval import Retriever

log = logging.getLogger("demo_rag.mcp")

MAX_QUERY_CHARS = 500
MAX_LIMIT = 10
INSTRUCTIONS = (
    "Поиск по Трудовому кодексу РФ (534 статьи, редакция в поле edition_date). "
    "Сначала search_tk по вопросу на русском, потом get_article для статей, которые нужно прочитать целиком. "
    "Отвечая пользователю, ссылайся на номер статьи и source_url. Это справочные данные, а не юридическая консультация."
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


class SearchHitOut(BaseModel):
    article: str = Field(description="Номер статьи, например 81 или 312.1")
    title: str
    chapter: str
    fragment: str = Field(description="Найденный фрагмент статьи, до 350 символов")
    part: int = Field(description="Номер фрагмента в статье")
    parts_total: int
    similarity: float | None = Field(description="Косинусная близость фрагмента к запросу, 0–1")
    source_url: str


class SearchResult(BaseModel):
    query: str
    results: list[SearchHitOut]


class ArticleOut(BaseModel):
    article: str
    title: str
    chapter: str
    text: str
    chars: int
    source_url: str
    edition_date: str


def normalize_article_number(raw: str) -> str | None:
    """"ст. 81", "Статья 312.1 ТК", "81" -> "81" / "312.1"."""
    found = ARTICLE_NUMBER_RE.findall(raw or "")
    return found[0] if found else None


def build_mcp_server(retriever: Retriever, store: ArticleStore | None = None) -> MCPServer:
    store = store or ArticleStore(retriever.index.chunks)
    server = MCPServer(name="tk-rf", title="Трудовой кодекс РФ", instructions=INSTRUCTIONS, version="1.0.0")

    @server.tool(title="Поиск по Трудовому кодексу", annotations=READ_ONLY)
    def search_tk(
        query: Annotated[str, Field(description="Вопрос или ключевые слова на русском, например «увольнение на больничном»")],
        limit: Annotated[int, Field(ge=1, le=MAX_LIMIT, description="Сколько разных статей вернуть")] = 5,
    ) -> SearchResult:
        """Найти статьи Трудового кодекса РФ по смыслу вопроса. Возвращает разные статьи в порядке
        близости, у каждой — лучший найденный фрагмент и ссылку на КонсультантПлюс."""
        query = " ".join((query or "").split())[:MAX_QUERY_CHARS]
        if not query:
            raise ToolError("Пустой запрос: передайте вопрос на русском.")
        results: list[SearchHitOut] = []
        seen: set[str] = set()
        for hit in retriever.search(query, top_k=MAX_LIMIT * 4):
            chunk = hit.chunk
            if chunk.article in seen:
                continue
            seen.add(chunk.article)
            results.append(
                SearchHitOut(
                    article=chunk.article,
                    title=base_header(chunk),
                    chapter=chunk.chapter,
                    fragment=chunk.body,
                    part=chunk.part,
                    parts_total=chunk.parts_total,
                    similarity=None if hit.dense_score is None else round(hit.dense_score, 4),
                    source_url=chunk.source_url,
                )
            )
            if len(results) >= limit:
                break
        log.info("mcp search_tk results=%d", len(results))
        return SearchResult(query=query, results=results)

    @server.tool(title="Статья Трудового кодекса целиком", annotations=READ_ONLY)
    def get_article(
        number: Annotated[str, Field(description="Номер статьи: «81», «312.1» или «ст. 80»")],
    ) -> ArticleOut:
        """Полный текст статьи Трудового кодекса РФ по номеру, с названием, главой и ссылкой на источник."""
        normalized = normalize_article_number(number)
        article = store.article(normalized) if normalized else None
        if article is None:
            raise ToolError(f"Статьи «{number}» нет в корпусе. Найдите номер через search_tk.")
        log.info("mcp get_article article=%s", article.article)
        return ArticleOut(
            article=article.article,
            title=article.header,
            chapter=article.chapter,
            text=article.text,
            chars=len(article.text),
            source_url=article.source_url,
            edition_date=article.edition_date,
        )

    return server


def http_security(allowed_hosts: list[str]) -> TransportSecuritySettings:
    """DNS-rebinding protection: only these Host headers, and browser requests only from these origins."""
    origins = [f"https://{h}" for h in allowed_hosts if "*" not in h] + [
        f"http://{h}" for h in allowed_hosts if h.split(":")[0] in ("127.0.0.1", "localhost", "[::1]")
    ]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=allowed_hosts, allowed_origins=origins
    )


def main() -> None:
    """stdio transport for a local checkout: the index must be built (python3 -m app.build_index)."""
    from app.config import Settings
    from app.main import _load_retriever

    logging.basicConfig(level=logging.WARNING)  # stdout is the protocol channel; logs go to stderr
    build_mcp_server(_load_retriever(Settings.from_env())).run("stdio")


if __name__ == "__main__":
    main()
