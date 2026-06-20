"""Unit tests for single-shot grounded generation (fakes + offline echo)."""

from __future__ import annotations

import pytest

from app.agent.generation import generate_grounded_answer
from app.llm.echo_provider import EchoLLMProvider
from app.retrieval.types import RetrievedChunk
from tests.fakes.providers import FakeLLMProvider, FakeRetrieval


def _chunk(chunk_id: str, content: str, source: str = "postmortem") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"DOC-{chunk_id}",
        source_type=source,
        title=f"Title {chunk_id}",
        content=content,
        char_start=0,
        char_end=len(content),
        score=1.0,
        retriever="rrf",
    )


@pytest.mark.unit
async def test_single_shot_answer_has_valid_citations() -> None:
    chunks = [
        _chunk("c1", "Average P1 resolution last quarter was about three hours."),
        _chunk("c2", "The top recurring root cause was a deployment regression."),
    ]
    llm = FakeLLMProvider(
        answer_text="Average P1 resolution was ~3h [1]. Top cause was deployment [2]. [9]"
    )
    result = await generate_grounded_answer("q", FakeRetrieval(chunks), llm, top_k=8)

    assert result.used_indices == [1, 2]
    assert len(result.sources) == 2
    assert {s.chunk_id for s in result.sources} == {"c1", "c2"}
    assert "[9]" not in result.answer  # hallucinated citation stripped
    assert result.model == "fake-llm"


@pytest.mark.unit
async def test_offline_echo_provider_grounds_answer() -> None:
    chunks = [
        _chunk("c1", "Payments gateway timed out due to a deployment regression."),
        _chunk("c2", "Authentication failed due to a capacity shortfall."),
    ]
    result = await generate_grounded_answer(
        "why did it fail", FakeRetrieval(chunks), EchoLLMProvider(), top_k=8
    )
    # The echo provider cites real record indices, so we get non-empty sources.
    assert result.sources
    assert result.used_indices
    assert all(i in {1, 2} for i in result.used_indices)
