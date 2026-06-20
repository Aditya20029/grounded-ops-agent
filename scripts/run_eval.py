"""Run the evaluation harness and print the Markdown metrics table.

Usage: python scripts/run_eval.py [--top-k 5] [--no-judge]

Faithfulness (LLM judge, temperature 0) runs only when an LLM provider key is
configured; it uses EVAL_JUDGE_MODEL. Run after `make seed` and `make ingest`.
The printed numbers are reproducible (seeded data, pinned models, temperature 0)
and are pasted into the README with the models and date.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from app.core.db import dispose_engine
from app.core.logging import configure_logging
from app.core.settings import Settings, get_settings
from app.eval.harness import format_report, run_eval
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider


def _build_judge(settings: Settings, enabled: bool) -> LLMProvider | None:
    if not enabled or settings.llm_provider not in ("anthropic", "openai"):
        return None
    judge_settings = settings.model_copy(update={"llm_model": settings.eval_judge_model})
    return get_llm_provider(judge_settings)


async def _main(top_k: int, judge_enabled: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    judge = _build_judge(settings, judge_enabled)
    date = datetime.now(UTC).date().isoformat()
    try:
        report = await run_eval(settings, top_k=top_k, date=date, judge=judge)
    finally:
        await dispose_engine()
    print(format_report(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval harness.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    args = parser.parse_args()
    asyncio.run(_main(args.top_k, not args.no_judge))


if __name__ == "__main__":
    main()
