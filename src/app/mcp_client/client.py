"""A thin, typed MCP client that manages the server lifecycle.

For stdio it spawns ``python -m app.mcp_server`` as a subprocess (inheriting the
environment so the child shares ``DATABASE_URL`` and embedding config) and tears
it down on exit. For streamable HTTP it connects to a running server. Tool
results are normalized; timeouts surface as ``ToolError``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from app.core.errors import ToolError
from app.core.settings import Settings
from app.llm.types import ToolSpec
from app.tools.adapter import (
    NormalizedToolResult,
    mcp_tool_obj_to_spec,
    normalize_call_result,
)


class MCPClient:
    """Connect to the analytics server, list tools, and call them."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> MCPClient:
        self._stack = AsyncExitStack()
        try:
            if self._settings.mcp_transport == "streamable-http":
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(self._settings.mcp_server_url)
                )
            else:
                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", "app.mcp_server"],
                    env=dict(os.environ),
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def list_tools(self) -> list[ToolSpec]:
        resp = await self._session.list_tools()
        return [mcp_tool_obj_to_spec(t) for t in resp.tools]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float = 15.0
    ) -> NormalizedToolResult:
        try:
            result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout)
        except TimeoutError as exc:
            raise ToolError(f"tool '{name}' timed out after {timeout}s") from exc
        return normalize_call_result(result)
