"""Tool adapter: MCP tool schemas <-> provider tool formats; result normalization.

MCP tools are discovered as JSON-schema definitions. This isolates every
provider difference in one place: an MCP tool becomes a ``ToolSpec`` (the
internal shape the orchestrator and providers consume), and helpers render a
``ToolSpec`` into OpenAI function-calling and Anthropic tool formats. Tool
results coming back are normalized into one internal shape. Duck-typed so it can
be unit-tested without importing the MCP SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.types import ToolSpec

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True)
class NormalizedToolResult:
    """A tool result reduced to one internal shape."""

    text: str
    structured: Any | None
    is_error: bool


def mcp_tool_to_spec(
    name: str, description: str | None, input_schema: dict[str, Any] | None
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or "",
        input_schema=input_schema or dict(_EMPTY_SCHEMA),
    )


def mcp_tool_obj_to_spec(tool: Any) -> ToolSpec:
    """Convert an MCP ``Tool`` (``.name``/``.description``/``.inputSchema``)."""
    return mcp_tool_to_spec(
        tool.name, getattr(tool, "description", None), getattr(tool, "inputSchema", None)
    )


def spec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    """OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def spec_to_anthropic(spec: ToolSpec) -> dict[str, Any]:
    """Anthropic tool format."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


def normalize_call_result(result: Any) -> NormalizedToolResult:
    """Reduce an MCP ``CallToolResult`` to ``NormalizedToolResult``."""
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    return NormalizedToolResult(
        text="\n".join(texts),
        structured=getattr(result, "structuredContent", None),
        is_error=bool(getattr(result, "isError", False)),
    )
