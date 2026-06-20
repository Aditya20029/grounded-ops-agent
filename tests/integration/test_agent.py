"""Integration test: orchestrator drives the real MCP server + real retrieval.

A scripted fake LLM calls a real analytics tool then answers, so the full stack
(seed retrieval -> tool loop -> MCP subprocess -> Postgres -> grounded answer)
runs in CI without any API key.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.agent.orchestrator import Orchestrator
from app.agent.tool_executor import MCPToolExecutor
from app.core.db import session_scope
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import PgChunkWriter
from app.llm.embeddings import HashingEmbeddingProvider
from app.llm.types import LLMResponse, LLMUsage, ToolCall
from app.mcp_client.client import MCPClient
from app.retrieval.service import build_retrieval_service
from tests.fakes.providers import FakeLLMProvider

pytestmark = pytest.mark.integration


async def _truncate_chunks() -> None:
    async with session_scope() as session:
        await session.execute(text("TRUNCATE chunks"))


async def test_agent_drives_real_mcp(db_engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    get_settings.cache_clear()
    settings = get_settings()

    await _truncate_chunks()
    provider = HashingEmbeddingProvider(settings.embedding_dim)
    await ingest_documents(
        [
            SourceDoc(
                "DOC-PAY",
                "postmortem",
                "Payments P1",
                "# Summary\n\npayments deployment regression caused a P1 outage",
                markdown=True,
            )
        ],
        provider,
        PgChunkWriter(),
    )

    llm = FakeLLMProvider(
        responses=[
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        "a1",
                        "aggregate",
                        {
                            "table": "incidents",
                            "group_by": ["root_cause_category"],
                            "agg_fn": "count",
                        },
                    ),
                ),
                usage=LLMUsage(10, 5),
                model="fake",
            ),
            LLMResponse(
                text="The top cause is deployment [1].", usage=LLMUsage(5, 5), model="fake"
            ),
        ]
    )

    try:
        async with MCPClient(settings) as client:
            orch = Orchestrator(
                llm=llm,
                retrieval=build_retrieval_service(settings),
                executor=MCPToolExecutor(client),
                settings=settings,
            )
            result = await orch.run("top root causes for P1 incidents")

        assert result.stop_reason == "answered"
        assert any(step.tool == "aggregate" and not step.is_error for step in result.trace)
        assert result.sources  # the seed postmortem was cited
    finally:
        get_settings.cache_clear()
