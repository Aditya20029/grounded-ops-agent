"""FAISS-vs-pgvector benchmark: recall@k and p50/p95 latency.

This comparison is the reason both stores exist. Given a set of (query,
relevant_chunk_id) pairs, each store is timed and scored so the eval harness can
report a side-by-side table.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import ceil, floor
from time import perf_counter

from app.llm.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore

Query = tuple[str, str]  # (query_text, relevant_chunk_id)


@dataclass(frozen=True)
class BenchmarkRow:
    label: str
    recall_at_k: float
    mrr: float
    p50_ms: float
    p95_ms: float
    n_queries: int


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    low, high = floor(k), ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] * (high - k) + ordered[high] * (k - low)


async def benchmark_store(
    label: str,
    store: VectorStore,
    provider: EmbeddingProvider,
    queries: list[Query],
    *,
    top_k: int = 5,
) -> BenchmarkRow:
    """Time ``store`` over ``queries`` and compute recall@k and MRR."""
    latencies: list[float] = []
    recalls: list[float] = []
    rrs: list[float] = []
    for text, relevant in queries:
        query_vector = await asyncio.to_thread(provider.embed_query, text)
        start = perf_counter()
        results = await store.search(query_vector, top_k)
        latencies.append((perf_counter() - start) * 1000.0)

        ids = [r.chunk_id for r in results]
        recalls.append(1.0 if relevant in ids else 0.0)
        rr = 0.0
        for rank, chunk_id in enumerate(ids, start=1):
            if chunk_id == relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    n = len(queries)
    return BenchmarkRow(
        label=label,
        recall_at_k=sum(recalls) / n if n else 0.0,
        mrr=sum(rrs) / n if n else 0.0,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        n_queries=n,
    )


def format_table(rows: list[BenchmarkRow], top_k: int) -> str:
    """Render benchmark rows as a Markdown table."""
    lines = [
        f"| store | recall@{top_k} | MRR | p50 ms | p95 ms | queries |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.label} | {r.recall_at_k:.3f} | {r.mrr:.3f} | "
            f"{r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.n_queries} |"
        )
    return "\n".join(lines)
