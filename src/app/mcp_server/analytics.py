"""Read-only analytics queries over the structured tables.

Every query validates identifiers against the whitelist and binds all values as
parameters. There are no writes and no arbitrary SQL. These functions take a
session so they can be unit/integration tested directly, independent of the MCP
transport.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ToolError
from app.security import whitelist

MAX_GROUP_BY = 3
MAX_FILTERS = 8
ALLOWED_METRICS = frozenset(
    {"avg_resolution_minutes", "p50_resolution_minutes", "incident_count", "unresolved_count"}
)


def _parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"invalid datetime '{value}', expected ISO 8601") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row_to_json(mapping: dict[str, Any]) -> dict[str, Any]:
    return {k: _jsonable(v) for k, v in mapping.items()}


def _time_clause(
    table: str, start: str | None, end: str | None, where: list[str], params: dict[str, Any]
) -> None:
    if start is None and end is None:
        return
    time_column = whitelist.policy(table).time_column
    if time_column is None:
        raise ToolError(f"table '{table}' has no time column to filter on")
    if start is not None:
        where.append(f"{time_column} >= :start")
        params["start"] = _parse_dt(start)
    if end is not None:
        where.append(f"{time_column} <= :end")
        params["end"] = _parse_dt(end)


async def run_aggregate(
    session: AsyncSession,
    *,
    table: str,
    group_by: list[str],
    agg_fn: str,
    column: str | None = None,
    filters: dict[str, Any] | None = None,
    start: str | None = None,
    end: str | None = None,
    cap: int,
) -> list[dict[str, Any]]:
    """Grouped aggregation over a whitelisted table."""
    filters = filters or {}
    if len(group_by) > MAX_GROUP_BY:
        raise ToolError(f"too many group_by columns (max {MAX_GROUP_BY})")
    if len(filters) > MAX_FILTERS:
        raise ToolError(f"too many filters (max {MAX_FILTERS})")

    whitelist.policy(table)  # validates table
    whitelist.validate_agg(agg_fn)
    whitelist.validate_group_by(table, group_by)
    whitelist.validate_filter_keys(table, list(filters))

    if agg_fn == "count":
        agg_expr = "count(*)"
    else:
        whitelist.validate_numeric_column(table, column or "")
        agg_expr = f"{agg_fn}({column})"

    where: list[str] = []
    params: dict[str, Any] = {}
    for i, (key, value) in enumerate(filters.items()):
        where.append(f"{key} = :v{i}")
        params[f"v{i}"] = value
    _time_clause(table, start, end, where, params)

    select_list = ", ".join([*group_by, f"{agg_expr} AS agg_value"])
    sql = f"SELECT {select_list} FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group_by:
        sql += " GROUP BY " + ", ".join(group_by)
    sql += f" ORDER BY agg_value DESC NULLS LAST LIMIT {int(cap)}"

    result = await session.execute(text(sql), params)
    return [_row_to_json(dict(row._mapping)) for row in result]


async def run_query_metric(
    session: AsyncSession,
    *,
    metric: str,
    severity: str | None = None,
    service: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """A curated single-number metric over the incidents table."""
    if metric not in ALLOWED_METRICS:
        raise ToolError(f"metric '{metric}' is not allowed (allowed: {sorted(ALLOWED_METRICS)})")

    where: list[str] = ["1=1"]
    params: dict[str, Any] = {}
    if severity is not None:
        where.append("severity = :sev")
        params["sev"] = severity
    if service is not None:
        where.append("service = :svc")
        params["svc"] = service
    _time_clause("incidents", start, end, where, params)
    clause = " AND ".join(where)

    resolution = "extract(epoch from (resolved_at - opened_at)) / 60.0"
    if metric == "avg_resolution_minutes":
        sql = (
            f"SELECT avg({resolution}) AS value FROM incidents "
            f"WHERE resolved_at IS NOT NULL AND {clause}"
        )
    elif metric == "p50_resolution_minutes":
        sql = (
            f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {resolution}) AS value "
            f"FROM incidents WHERE resolved_at IS NOT NULL AND {clause}"
        )
    elif metric == "unresolved_count":
        sql = f"SELECT count(*) AS value FROM incidents WHERE resolved_at IS NULL AND {clause}"
    else:  # incident_count
        sql = f"SELECT count(*) AS value FROM incidents WHERE {clause}"

    row = (await session.execute(text(sql), params)).first()
    value = row.value if row is not None else None
    return {
        "metric": metric,
        "filters": {"severity": severity, "service": service, "start": start, "end": end},
        "value": float(value) if value is not None else None,
    }


async def run_timeseries(
    session: AsyncSession,
    *,
    metric_name: str,
    service: str | None = None,
    granularity: str = "day",
    start: str | None = None,
    end: str | None = None,
    cap: int,
) -> list[dict[str, Any]]:
    """A time-bucketed series from metrics_daily."""
    whitelist.validate_granularity(granularity)
    where = ["metric_name = :metric"]
    params: dict[str, Any] = {"metric": metric_name}
    if service is not None:
        where.append("service = :svc")
        params["svc"] = service
    _time_clause("metrics_daily", start, end, where, params)

    sql = (
        f"SELECT date_trunc('{granularity}', date) AS bucket, avg(value) AS value "
        f"FROM metrics_daily WHERE {' AND '.join(where)} "
        f"GROUP BY bucket ORDER BY bucket LIMIT {int(cap)}"
    )
    result = await session.execute(text(sql), params)
    return [{"bucket": _jsonable(row.bucket), "value": float(row.value)} for row in result]


async def get_record(session: AsyncSession, *, table: str, record_id: str) -> dict[str, Any] | None:
    """Fetch a single row from a whitelisted table by id."""
    whitelist.policy(table)  # validates table
    sql = f"SELECT * FROM {table} WHERE id = :id LIMIT 1"
    row = (await session.execute(text(sql), {"id": record_id})).first()
    return _row_to_json(dict(row._mapping)) if row is not None else None
