"""Debug CLI for the full agent loop (spawns the MCP server, uses the real LLM).

Usage:
    python scripts/ask.py "What was the average resolution time for P1 incidents
    last quarter, and what were the top 3 recurring root causes?"
"""

from __future__ import annotations

import argparse
import asyncio

from app.agent.orchestrator import build_orchestrator, mcp_executor
from app.core.db import dispose_engine
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.mcp_client.client import MCPClient


async def _main(question: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        async with MCPClient(settings) as client:
            orch = build_orchestrator(settings, mcp_executor(client))
            result = await orch.run(question)
    finally:
        await dispose_engine()

    print("\n=== ANSWER ===\n" + result.answer)
    print("\n=== SOURCES ===")
    for src in result.sources:
        print(f"  [{src.index}] {src.source_type} {src.doc_id} :: {src.title}")
    print("\n=== TRACE ===")
    for step in result.trace:
        flag = " ERROR" if step.is_error else ""
        print(f"  step {step.step} {step.tool}({step.arguments}) {step.latency_ms:.0f}ms{flag}")
        print(f"      -> {step.result_summary}")
    print(
        f"\nmodel={result.model} steps={result.steps} stop={result.stop_reason} "
        f"cost=${result.cost_usd:.4f} tokens={result.usage.input_tokens}in/"
        f"{result.usage.output_tokens}out"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grounded agent on one question.")
    parser.add_argument("question")
    args = parser.parse_args()
    asyncio.run(_main(args.question))


if __name__ == "__main__":
    main()
