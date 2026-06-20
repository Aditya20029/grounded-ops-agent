"""The MCP analytics server: read-only tools over the operational data.

Tools are parameterized, validated against the whitelist, and row-capped. They
never write and never accept non-whitelisted identifiers. ``search_records``
delegates to the single retrieval implementation so behaviour matches the
orchestrator's seed retrieval.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from app.core.db import session_scope
from app.core.settings import Settings, get_settings
from app.mcp_server import analytics
from app.retrieval.service import build_retrieval_service
from app.retrieval.types import SearchFilters
from app.security import whitelist


def build_mcp(settings: Settings | None = None) -> FastMCP:
    """Construct the FastMCP server with all analytics tools registered."""
    settings = settings or get_settings()
    cap = settings.mcp_max_result_rows
    parsed = urlparse(settings.mcp_server_url)

    mcp = FastMCP(
        "grounded-ops-analytics",
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 8848,
        streamable_http_path=parsed.path or "/mcp",
    )

    @mcp.tool()
    async def list_schema() -> dict[str, Any]:
        """List the tables, columns, group-by columns, and aggregate functions
        available to the analytics tools."""
        return whitelist.schema()

    @mcp.tool()
    async def query_metrics(
        metric: str,
        severity: str | None = None,
        service: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Compute a curated incident metric. ``metric`` is one of
        avg_resolution_minutes, p50_resolution_minutes, incident_count,
        unresolved_count. ``start``/``end`` are ISO timestamps over opened_at."""
        async with session_scope() as session:
            return await analytics.run_query_metric(
                session, metric=metric, severity=severity, service=service, start=start, end=end
            )

    @mcp.tool()
    async def aggregate(
        table: str,
        group_by: list[str],
        agg_fn: str,
        column: str | None = None,
        filters: dict[str, Any] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Grouped aggregation over a whitelisted table (e.g. top root causes:
        table='incidents', group_by=['root_cause_category'], agg_fn='count',
        filters={'severity':'P1'})."""
        async with session_scope() as session:
            return await analytics.run_aggregate(
                session,
                table=table,
                group_by=group_by,
                agg_fn=agg_fn,
                column=column,
                filters=filters,
                start=start,
                end=end,
                cap=cap,
            )

    @mcp.tool()
    async def get_timeseries(
        metric_name: str,
        service: str | None = None,
        granularity: str = "day",
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Time-bucketed series from metrics_daily. granularity in day/week/month."""
        async with session_scope() as session:
            return await analytics.run_timeseries(
                session,
                metric_name=metric_name,
                service=service,
                granularity=granularity,
                start=start,
                end=end,
                cap=cap,
            )

    @mcp.tool()
    async def get_record(table: str, id: str) -> dict[str, Any] | None:
        """Fetch a single row from a whitelisted table by id."""
        async with session_scope() as session:
            return await analytics.get_record(session, table=table, record_id=id)

    @mcp.tool()
    async def search_records(
        query: str, source_types: list[str] | None = None, top_k: int = 6
    ) -> list[dict[str, Any]]:
        """Hybrid semantic + keyword search over the unstructured corpus."""
        service = build_retrieval_service(settings)
        filters = SearchFilters(source_types=tuple(source_types)) if source_types else None
        chunks = await service.retrieve(
            query[: settings.max_query_chars], top_k=min(top_k, 20), filters=filters
        )
        return [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_type": c.source_type,
                "title": c.title,
                "snippet": " ".join(c.content.split())[:300],
                "score": round(c.score, 4),
            }
            for c in chunks
        ]

    return mcp
