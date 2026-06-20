"""Run the MCP analytics server: ``python -m app.mcp_server``.

Transport is chosen from settings: stdio for local development (the orchestrator
spawns this as a subprocess) or streamable-http for a networked/deployed setup.
"""

from __future__ import annotations

from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.mcp_server.server import build_mcp


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    mcp = build_mcp(settings)
    if settings.mcp_transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
