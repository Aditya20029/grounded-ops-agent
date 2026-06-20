"""End-to-end smoke test of the headline query.

Asserts the response contains a non-empty sources array and (with a real LLM
provider) at least one analytics tool call in the trace. With the offline echo
provider the tool-planning step is skipped, so only the sources assertion runs.

Usage: python scripts/smoke.py   (after make up/migrate/seed/ingest)
"""

from __future__ import annotations

import asyncio
import sys

from app.agent.orchestrator import build_orchestrator, mcp_executor
from app.core.db import dispose_engine
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.mcp_client.client import MCPClient

HEADLINE = (
    "What was the average resolution time for P1 incidents last quarter, "
    "and what were the top 3 recurring root causes?"
)
_ANALYTICS_TOOLS = {"aggregate", "query_metrics", "get_timeseries", "get_record", "list_schema"}


async def _main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    real_llm = settings.llm_provider in ("anthropic", "openai")

    try:
        async with MCPClient(settings) as client:
            orch = build_orchestrator(settings, mcp_executor(client))
            result = await orch.run(HEADLINE)
    finally:
        await dispose_engine()

    print(f"\nANSWER:\n{result.answer}\n")
    print(f"sources: {len(result.sources)} | steps: {result.steps} | stop: {result.stop_reason}")
    analytics_calls = [t for t in result.trace if t.tool in _ANALYTICS_TOOLS]
    print(f"analytics tool calls: {[t.tool for t in analytics_calls]}")

    assert result.sources, "FAIL: response has no sources"
    if real_llm:
        assert analytics_calls, "FAIL: no analytics tool call in the trace"
        print("\nSMOKE PASSED (sources + analytics tool call).")
    else:
        print("\nSMOKE PASSED (sources). Set a real LLM key to also exercise tool planning.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
