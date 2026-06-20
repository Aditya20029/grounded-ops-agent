"""Identifier whitelist for the analytics tools.

SQL identifiers (table, column, group-by, aggregate function) cannot be bound as
parameters, so they are validated against an explicit whitelist and only then
interpolated into SQL. Values are always bound as parameters. This closes the
identifier-injection vector: anything not in the whitelist is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.errors import ToolError

ALLOWED_AGG: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})
ALLOWED_GRANULARITY: frozenset[str] = frozenset({"day", "week", "month"})


@dataclass(frozen=True)
class TablePolicy:
    columns: frozenset[str]
    numeric_columns: frozenset[str]
    group_by: frozenset[str]
    filter_columns: frozenset[str]
    time_column: str | None = None
    extra: frozenset[str] = field(default_factory=frozenset)


ALLOWED: dict[str, TablePolicy] = {
    "incidents": TablePolicy(
        columns=frozenset(
            {
                "id",
                "severity",
                "service",
                "opened_at",
                "resolved_at",
                "root_cause_category",
                "customer_id",
            }
        ),
        numeric_columns=frozenset(),
        group_by=frozenset({"severity", "service", "root_cause_category"}),
        filter_columns=frozenset({"severity", "service", "root_cause_category", "customer_id"}),
        time_column="opened_at",
    ),
    "tickets": TablePolicy(
        columns=frozenset(
            {"id", "title", "priority", "status", "customer_id", "created_at", "resolved_at"}
        ),
        numeric_columns=frozenset(),
        group_by=frozenset({"priority", "status"}),
        filter_columns=frozenset({"priority", "status", "customer_id"}),
        time_column="created_at",
    ),
    "metrics_daily": TablePolicy(
        columns=frozenset({"id", "date", "service", "metric_name", "value"}),
        numeric_columns=frozenset({"value"}),
        group_by=frozenset({"service", "metric_name"}),
        filter_columns=frozenset({"service", "metric_name"}),
        time_column="date",
    ),
    "customers": TablePolicy(
        columns=frozenset({"id", "name", "plan", "region", "sla_id", "created_at"}),
        numeric_columns=frozenset(),
        group_by=frozenset({"plan", "region"}),
        filter_columns=frozenset({"plan", "region", "sla_id"}),
        time_column="created_at",
    ),
    "slas": TablePolicy(
        columns=frozenset({"id", "name", "tier", "target_resolution_minutes", "uptime_target_pct"}),
        numeric_columns=frozenset({"target_resolution_minutes", "uptime_target_pct"}),
        group_by=frozenset({"tier"}),
        filter_columns=frozenset({"tier"}),
        time_column=None,
    ),
}


def policy(table: str) -> TablePolicy:
    """Return the policy for a whitelisted table, else raise ToolError."""
    if table not in ALLOWED:
        raise ToolError(f"table '{table}' is not allowed (whitelisted: {sorted(ALLOWED)})")
    return ALLOWED[table]


def validate_agg(agg_fn: str) -> str:
    if agg_fn not in ALLOWED_AGG:
        raise ToolError(f"aggregate '{agg_fn}' is not allowed (allowed: {sorted(ALLOWED_AGG)})")
    return agg_fn


def validate_group_by(table: str, columns: list[str]) -> list[str]:
    allowed = policy(table).group_by
    for col in columns:
        if col not in allowed:
            raise ToolError(f"group_by '{col}' is not allowed on '{table}'")
    return columns


def validate_filter_keys(table: str, keys: list[str]) -> None:
    allowed = policy(table).filter_columns
    for key in keys:
        if key not in allowed:
            raise ToolError(f"filter column '{key}' is not allowed on '{table}'")


def validate_numeric_column(table: str, column: str) -> str:
    if column not in policy(table).numeric_columns:
        raise ToolError(f"column '{column}' is not an aggregatable numeric column on '{table}'")
    return column


def validate_granularity(granularity: str) -> str:
    if granularity not in ALLOWED_GRANULARITY:
        raise ToolError(
            f"granularity '{granularity}' is not allowed (allowed: {sorted(ALLOWED_GRANULARITY)})"
        )
    return granularity


def schema() -> dict[str, object]:
    """A description of the whitelisted surface, returned by ``list_schema``."""
    return {
        "tables": {
            name: {
                "columns": sorted(p.columns),
                "group_by_columns": sorted(p.group_by),
                "filter_columns": sorted(p.filter_columns),
                "numeric_columns": sorted(p.numeric_columns),
                "time_column": p.time_column,
            }
            for name, p in ALLOWED.items()
        },
        "aggregate_functions": sorted(ALLOWED_AGG),
        "granularities": sorted(ALLOWED_GRANULARITY),
    }
