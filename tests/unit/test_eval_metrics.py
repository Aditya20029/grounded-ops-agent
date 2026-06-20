"""Unit tests for evaluation metric definitions."""

from __future__ import annotations

import math

import pytest

from app.eval import metrics


@pytest.mark.unit
def test_dedup_preserve_order() -> None:
    assert metrics.dedup_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


@pytest.mark.unit
def test_recall_at_k() -> None:
    ranked = ["d1", "d2", "d3", "d4"]
    assert metrics.recall_at_k(ranked, {"d2", "d4"}, k=4) == 1.0
    assert metrics.recall_at_k(ranked, {"d2", "d4"}, k=2) == 0.5
    assert metrics.recall_at_k(ranked, set(), k=4) == 0.0


@pytest.mark.unit
def test_mrr() -> None:
    assert metrics.mrr(["d1", "d2", "d3"], {"d2"}) == 0.5
    assert metrics.mrr(["d1", "d2"], {"d1"}) == 1.0
    assert metrics.mrr(["d1", "d2"], {"x"}) == 0.0


@pytest.mark.unit
def test_ndcg_at_k() -> None:
    # Relevant at rank 1 -> perfect nDCG.
    assert metrics.ndcg_at_k(["d1", "d2"], {"d1"}, k=2) == 1.0
    # Relevant at rank 2 only: dcg = 1/log2(3); idcg = 1/log2(2) = 1.
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(metrics.ndcg_at_k(["d1", "d2"], {"d2"}, k=2), expected)


@pytest.mark.unit
def test_citation_precision_recall() -> None:
    precision, recall = metrics.citation_precision_recall({"d1", "d2"}, {"d1", "d3"})
    assert precision == 0.5  # d1 of {d1,d2} is relevant
    assert recall == 0.5  # d1 of {d1,d3} is cited
    assert metrics.citation_precision_recall(set(), {"d1"}) == (0.0, 0.0)


@pytest.mark.unit
def test_percentile() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert metrics.percentile(values, 0.5) == 25.0
    assert metrics.percentile([], 0.5) == 0.0
