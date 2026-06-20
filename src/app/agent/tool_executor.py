"""Tool execution abstraction for the orchestrator.

The orchestrator depends on this protocol, not on the MCP client directly, so it
can be driven by a fake in tests and by the real MCP server in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.llm.types import ToolSpec
from app.mcp_client.client import MCPClient


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    structured: Any | None
    is_error: bool


@runtime_checkable
class ToolExecutor(Protocol):
    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(
        self, name: str, arguments: dict[str, Any], *, timeout: float
    ) -> ToolOutcome: ...


class MCPToolExecutor:
    """ToolExecutor backed by a connected MCP client."""

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def list_tools(self) -> list[ToolSpec]:
        return await self._client.list_tools()

    async def call(self, name: str, arguments: dict[str, Any], *, timeout: float) -> ToolOutcome:
        result = await self._client.call_tool(name, arguments, timeout=timeout)
        return ToolOutcome(text=result.text, structured=result.structured, is_error=result.is_error)
