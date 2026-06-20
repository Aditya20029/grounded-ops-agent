"""Unit tests for Reciprocal Rank Fusion on known inputs."""

from __future__ import annotations

import math

import pytest

from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d",
        source_type="ticket",
        title="t",
        content="body",
        char_start=0,
        char_end=1,
        score=0.0,
        retriever="dense",
    )


@pytest.mark.unit
def test_rrf_scores_match_formula() -> None:
    list_a = [_chunk("1"), _chunk("2"), _chunk("3")]
    list_b = [_chunk("2"), _chunk("1"), _chunk("4")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    scores = {c.chunk_id: c.score for c in fused}

    assert math.isclose(scores["1"], 1 / 61 + 1 / 62)
    assert math.isclose(scores["2"], 1 / 61 + 1 / 62)
    assert math.isclose(scores["3"], 1 / 63)
    assert math.isclose(scores["4"], 1 / 63)
    assert {c.chunk_id for c in fused} == {"1", "2", "3", "4"}
    assert all(c.retriever == "rrf" for c in fused)


@pytest.mark.unit
def test_rrf_tiebreak_is_deterministic() -> None:
    # "1" and "2" tie; the (score, chunk_id) tiebreak puts "2" first.
    fused = reciprocal_rank_fusion([[_chunk("1"), _chunk("2")], [_chunk("2"), _chunk("1")]], k=60)
    assert fused[0].chunk_id == "2"


@pytest.mark.unit
def test_rrf_single_list_preserves_order() -> None:
    fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b"), _chunk("c")]], k=60)
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


@pytest.mark.unit
def test_rrf_top_k_truncates() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]], k=60, top_k=2
    )
    assert len(fused) == 2
