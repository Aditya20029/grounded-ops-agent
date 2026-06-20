"""Integration test: full MCP stdio round-trip against the analytics server.

Spawns ``python -m app.mcp_server`` as a subprocess, lists tools, calls them,
and checks the whitelist rejects a bad table. Forces offline embeddings in the
child so no model download is needed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.core.db import session_scope
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import PgChunkWriter
from app.llm.embeddings import HashingEmbeddingProvider
from app.mcp_client.client import MCPClient

pytestmark = pytest.mark.integration

_EXPECTED_TOOLS = {
    "list_schema",
    "query_metrics",
    "aggregate",
    "get_timeseries",
    "get_record",
    "search_records",
}


async def _truncate_chunks() -> None:
    async with session_scope() as session:
        await session.execute(text("TRUNCATE chunks"))


async def test_mcp_round_trip(db_engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    # Child server inherits this env, so it uses offline embeddings.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    get_settings.cache_clear()
    settings = get_settings()

    await _truncate_chunks()
    provider = HashingEmbeddingProvider(settings.embedding_dim)
    await ingest_documents(
        [
            SourceDoc(
                "DOC-X",
                "postmortem",
                "Payments P1",
                "# Summary\n\npayments deployment regression caused a P1 outage",
                markdown=True,
            )
        ],
        provider,
        PgChunkWriter(),
    )

    try:
        async with MCPClient(settings) as client:
            names = {t.name for t in await client.list_tools()}
            assert names >= _EXPECTED_TOOLS

            schema = await client.call_tool("list_schema", {})
            assert not schema.is_error
            assert "incidents" in schema.text

            agg = await client.call_tool(
                "aggregate",
                {
                    "table": "incidents",
                    "group_by": ["root_cause_category"],
                    "agg_fn": "count",
                    "filters": {"severity": "P1"},
                },
            )
            assert not agg.is_error

            found = await client.call_tool(
                "search_records", {"query": "payments deployment regression", "top_k": 3}
            )
            assert not found.is_error
            assert "DOC-X" in found.text

            # Whitelist rejects a non-whitelisted table.
            bad_is_error = False
            try:
                bad = await client.call_tool(
                    "aggregate", {"table": "not_a_table", "group_by": [], "agg_fn": "count"}
                )
                bad_is_error = bad.is_error
            except Exception:
                bad_is_error = True  # protocol-level error is also acceptable
            assert bad_is_error
    finally:
        get_settings.cache_clear()
