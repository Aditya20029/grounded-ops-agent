"""Unit tests for the MCP <-> provider tool adapter (both directions)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.tools.adapter import (
    mcp_tool_obj_to_spec,
    normalize_call_result,
    spec_to_anthropic,
    spec_to_openai,
)

_SCHEMA = {
    "type": "object",
    "properties": {"table": {"type": "string"}, "agg_fn": {"type": "string"}},
    "required": ["table", "agg_fn"],
}


@pytest.mark.unit
def test_mcp_to_spec_to_openai_and_anthropic() -> None:
    tool = SimpleNamespace(name="aggregate", description="grouped aggregation", inputSchema=_SCHEMA)
    spec = mcp_tool_obj_to_spec(tool)
    assert spec.name == "aggregate"
    assert spec.input_schema == _SCHEMA

    openai = spec_to_openai(spec)
    assert openai["type"] == "function"
    assert openai["function"]["name"] == "aggregate"
    assert openai["function"]["parameters"] == _SCHEMA

    anthropic = spec_to_anthropic(spec)
    assert anthropic["name"] == "aggregate"
    assert anthropic["description"] == "grouped aggregation"
    assert anthropic["input_schema"] == _SCHEMA


@pytest.mark.unit
def test_missing_description_and_schema_default() -> None:
    spec = mcp_tool_obj_to_spec(SimpleNamespace(name="t", description=None, inputSchema=None))
    assert spec.description == ""
    assert spec.input_schema == {"type": "object", "properties": {}}


@pytest.mark.unit
def test_normalize_success_result() -> None:
    block = SimpleNamespace(type="text", text="hello world")
    result = SimpleNamespace(content=[block], structuredContent={"value": 42}, isError=False)
    norm = normalize_call_result(result)
    assert norm.text == "hello world"
    assert norm.structured == {"value": 42}
    assert norm.is_error is False


@pytest.mark.unit
def test_normalize_error_result() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(text="boom")], structuredContent=None, isError=True
    )
    norm = normalize_call_result(result)
    assert norm.is_error is True
    assert norm.text == "boom"
