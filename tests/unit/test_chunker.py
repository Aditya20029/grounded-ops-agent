"""Unit tests for the token-aware, header-aware chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_document, split_markdown_sections


@pytest.mark.unit
def test_char_spans_round_trip() -> None:
    text = "Lorem ipsum dolor sit amet, consectetur. " * 200
    pieces = chunk_document(text, markdown=False, target=50, overlap=10)
    assert len(pieces) > 1
    for piece in pieces:
        # The defining invariant: content equals the document slice.
        assert piece.content == text[piece.char_start : piece.char_end]


@pytest.mark.unit
def test_indices_are_monotonic_from_zero() -> None:
    text = "alpha beta gamma delta epsilon. " * 100
    pieces = chunk_document(text, markdown=False, target=30, overlap=5)
    assert [p.chunk_index for p in pieces] == list(range(len(pieces)))


@pytest.mark.unit
def test_overlap_creates_shared_tokens() -> None:
    text = "one two three four five six seven eight. " * 50
    no_overlap = chunk_document(text, markdown=False, target=40, overlap=0)
    overlap = chunk_document(text, markdown=False, target=40, overlap=20)
    # More overlap -> more (shorter-stride) chunks for the same text.
    assert len(overlap) >= len(no_overlap)


@pytest.mark.unit
def test_short_text_single_chunk() -> None:
    pieces = chunk_document("just a few words here", markdown=False)
    assert len(pieces) == 1
    assert pieces[0].content.strip() == "just a few words here"


@pytest.mark.unit
def test_markdown_is_header_aware() -> None:
    md = (
        "# Title\n\nintro paragraph\n\n"
        "## Summary\n\n" + ("summary sentence. " * 40) + "\n\n"
        "## Root cause\n\n" + ("cause sentence. " * 40)
    )
    sections = split_markdown_sections(md)
    assert len(sections) >= 3  # [title+intro], [Summary], [Root cause]

    pieces = chunk_document(md, markdown=True, target=40, overlap=8)
    assert [p.chunk_index for p in pieces] == list(range(len(pieces)))
    for piece in pieces:
        assert piece.content == md[piece.char_start : piece.char_end]
