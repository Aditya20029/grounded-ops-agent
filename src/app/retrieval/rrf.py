"""Reciprocal Rank Fusion for merging ranked retrieval lists.

``score(d) = sum over lists of 1 / (k + rank_list(d))`` with ``k = 60`` by
default. A higher fused score ranks higher. Documents are deduplicated by
``chunk_id``; the first-seen instance carries the merged score.
"""

from __future__ import annotations

from dataclasses import replace

from app.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists; ranks are 1-based within each list."""
    scores: dict[str, float] = {}
    representative: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            representative.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(
        representative.values(),
        key=lambda c: (scores[c.chunk_id], c.chunk_id),
        reverse=True,
    )
    fused = [replace(c, score=scores[c.chunk_id], retriever="rrf") for c in ordered]
    return fused[:top_k] if top_k is not None else fused
