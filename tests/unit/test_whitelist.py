"""Unit tests for the analytics identifier whitelist (injection defense)."""

from __future__ import annotations

import pytest

from app.core.errors import ToolError
from app.security import whitelist


@pytest.mark.unit
def test_policy_rejects_unknown_table() -> None:
    with pytest.raises(ToolError):
        whitelist.policy("users; DROP TABLE incidents")


@pytest.mark.unit
def test_validate_agg() -> None:
    assert whitelist.validate_agg("count") == "count"
    with pytest.raises(ToolError):
        whitelist.validate_agg("drop")


@pytest.mark.unit
def test_group_by_rejects_non_whitelisted_column() -> None:
    assert whitelist.validate_group_by("incidents", ["severity"]) == ["severity"]
    with pytest.raises(ToolError):
        whitelist.validate_group_by("incidents", ["id; --"])


@pytest.mark.unit
def test_filter_keys_rejected() -> None:
    with pytest.raises(ToolError):
        whitelist.validate_filter_keys("incidents", ["password"])


@pytest.mark.unit
def test_numeric_column_required_for_non_count_agg() -> None:
    assert whitelist.validate_numeric_column("metrics_daily", "value") == "value"
    with pytest.raises(ToolError):
        whitelist.validate_numeric_column("incidents", "service")


@pytest.mark.unit
def test_granularity_validation() -> None:
    assert whitelist.validate_granularity("week") == "week"
    with pytest.raises(ToolError):
        whitelist.validate_granularity("century")


@pytest.mark.unit
def test_schema_describes_surface() -> None:
    schema = whitelist.schema()
    assert "incidents" in schema["tables"]  # type: ignore[operator]
    assert "count" in schema["aggregate_functions"]  # type: ignore[operator]
