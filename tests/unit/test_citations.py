"""Unit tests for citation post-validation (strip hallucinated indices)."""

from __future__ import annotations

import pytest

from app.agent.citation_registry import CitationRegistry
from app.agent.citations import validate_and_strip
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "d", "ticket", "t", "body", 0, 4, 1.0, "rrf")


@pytest.mark.unit
def test_strips_hallucinated_citations() -> None:
    registry = CitationRegistry()
    registry.add([_chunk("a"), _chunk("b")])

    cleaned, used = validate_and_strip("Alpha [1] beta [2] gamma [7].", registry)
    assert used == [1, 2]
    assert "[7]" not in cleaned
    assert "[1]" in cleaned and "[2]" in cleaned


@pytest.mark.unit
def test_no_citations_yields_empty_used() -> None:
    registry = CitationRegistry()
    registry.add([_chunk("a")])
    cleaned, used = validate_and_strip("No citations here.", registry)
    assert used == []
    assert cleaned == "No citations here."
