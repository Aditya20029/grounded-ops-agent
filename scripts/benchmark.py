"""Benchmark FAISS (flat/IVF/HNSW) vs pgvector on recall@k and latency.

Builds a self-retrieval query set from a sample of stored chunks (each chunk's
own text is the query; itself is the relevant result) and times each store.
Run after `make seed` and `make ingest`. The eval harness (Phase 9) reuses this
with the gold set and writes the numbers into the README.

Usage:
    python scripts/benchmark.py --queries 50 --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import random

from app.core.db import dispose_engine
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.llm.embeddings import get_embedding_provider
from app.retrieval.benchmark import benchmark_store, format_table
from app.retrieval.faiss_store import DEFAULT_INDEX_DIR, FaissStore
from app.retrieval.pgvector_store import PgVectorStore


async def _main(n_queries: int, top_k: int, seed: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    provider = get_embedding_provider(settings)
    pg_store = PgVectorStore()

    chunks = await pg_store.load_all()
    if not chunks:
        print("No chunks in pgvector. Run `make seed` and `make ingest` first.")
        await dispose_engine()
        return

    rng = random.Random(seed)
    sample = rng.sample(chunks, min(n_queries, len(chunks)))
    queries = [(" ".join(c.content.split())[:160], c.chunk_id) for c in sample]

    rows = [await benchmark_store("pgvector", pg_store, provider, queries, top_k=top_k)]
    for index_type in ("flat", "ivf", "hnsw"):
        store = FaissStore(
            settings.embedding_dim,
            index_path=DEFAULT_INDEX_DIR / f"bench_{index_type}.faiss",
            index_type=index_type,
        )
        await store.build(chunks, provider.model_name)
        rows.append(
            await benchmark_store(f"faiss:{index_type}", store, provider, queries, top_k=top_k)
        )

    print(f"\nFAISS vs pgvector  (n={len(queries)} queries, model={provider.model_name})\n")
    print(format_table(rows, top_k))
    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS vs pgvector benchmark.")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(_main(args.queries, args.top_k, args.seed))


if __name__ == "__main__":
    main()
