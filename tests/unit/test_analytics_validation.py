"""Unit tests: analytics reject non-whitelisted identifiers before any SQL runs."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import ToolError
from app.mcp_server.analytics import run_aggregate


class _DummySession:
    """Never used: validation fails before the session is touched."""


@pytest.mark.unit
async def test_aggregate_rejects_non_whitelisted_table() -> None:
    with pytest.raises(ToolError):
        await run_aggregate(
            _DummySession(),  # type: ignore[arg-type]
            table="customers; DROP TABLE incidents",
            group_by=[],
            agg_fn="count",
            cap=10,
        )


@pytest.mark.unit
async def test_aggregate_rejects_non_whitelisted_column() -> None:
    with pytest.raises(ToolError):
        await run_aggregate(
            _DummySession(),  # type: ignore[arg-type]
            table="metrics_daily",
            group_by=[],
            agg_fn="sum",
            column="value); DROP TABLE x; --",
            cap=10,
        )


@pytest.mark.unit
async def test_aggregate_rejects_non_whitelisted_group_by() -> None:
    with pytest.raises(ToolError):
        await run_aggregate(
            _DummySession(),  # type: ignore[arg-type]
            table="incidents",
            group_by=["id"],  # not a permitted group-by column
            agg_fn="count",
            cap=10,
        )


@pytest.mark.unit
async def test_aggregate_rejects_bad_aggregate_function() -> None:
    bad_args: dict[str, Any] = {"table": "incidents", "group_by": ["severity"], "agg_fn": "drop"}
    with pytest.raises(ToolError):
        await run_aggregate(_DummySession(), cap=10, **bad_args)  # type: ignore[arg-type]
