"""Unit tests for citation registry index stability across retrieval rounds."""

from __future__ import annotations

import pytest

from app.agent.citation_registry import CitationRegistry
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str, content: str = "some content here") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"D-{chunk_id}",
        source_type="postmortem",
        title=f"Title {chunk_id}",
        content=content,
        char_start=0,
        char_end=len(content),
        score=1.0,
        retriever="rrf",
    )


@pytest.mark.unit
def test_indices_are_stable_across_rounds() -> None:
    registry = CitationRegistry()
    round1 = registry.add([_chunk("a"), _chunk("b")])
    assert round1 == [1, 2]

    # "b" reappears in a later round and keeps index 2; "c" is new.
    round2 = registry.add([_chunk("b"), _chunk("c")])
    assert round2 == [2, 3]

    assert len(registry) == 3
    assert registry.get(2) is not None
    assert registry.get(2).chunk_id == "b"  # type: ignore[union-attr]
    assert registry.has_index(3)
    assert not registry.has_index(4)


@pytest.mark.unit
def test_render_includes_all_indices_and_fences() -> None:
    registry = CitationRegistry()
    registry.add([_chunk("a"), _chunk("b"), _chunk("c")])
    rendered = registry.render_for_prompt()
    assert "[1]" in rendered and "[2]" in rendered and "[3]" in rendered
    assert "<<<RECORD" in rendered and "RECORD>>>" in rendered
