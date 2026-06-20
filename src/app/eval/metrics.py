"""Evaluation metric definitions (pure functions over ranked doc ids).

Relevance is at the document level (a retrieved chunk is relevant if its
``doc_id`` is in the gold set), which is robust to chunk-boundary changes.
Rankings are expected to be deduplicated to unique doc ids in rank order.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def dedup_preserve_order(doc_ids: Iterable[str]) -> list[str]:
    """First occurrence of each doc id, preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs present in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k with binary relevance."""
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def citation_precision_recall(cited: set[str], relevant: set[str]) -> tuple[float, float]:
    """Precision/recall of cited docs against the relevant set."""
    precision = len(cited & relevant) / len(cited) if cited else 0.0
    recall = len(cited & relevant) / len(relevant) if relevant else 0.0
    return precision, recall


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] * (high - k) + ordered[high] * (k - low)
