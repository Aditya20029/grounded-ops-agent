"""Debug CLI for raw hybrid retrieval.

Usage:
    python scripts/search.py "payments p1 incident root cause" --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.db import dispose_engine
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.retrieval.service import build_retrieval_service


async def _main(query: str, top_k: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    service = build_retrieval_service(settings)
    try:
        results = await service.retrieve(query, top_k=top_k)
    finally:
        await dispose_engine()

    if not results:
        print("No results. Did you run `make seed` and `make ingest`?")
        return
    for i, r in enumerate(results, start=1):
        snippet = " ".join(r.content.split())[:160]
        print(f"[{i}] {r.retriever} score={r.score:.4f} {r.source_type} {r.doc_id} :: {r.title}")
        print(f"    {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval debug search.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_main(args.query, args.top_k))


if __name__ == "__main__":
    main()
