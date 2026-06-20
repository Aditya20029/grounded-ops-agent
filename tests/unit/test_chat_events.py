"""Unit tests for the streamed agent event protocol and SSE serializer."""

from __future__ import annotations

import json

import pytest

from app.agent.generation import Source
from app.agent.orchestrator import Orchestrator
from app.api.chat import _default
from app.core.settings import get_settings
from app.llm.types import LLMResponse, LLMUsage, ToolSpec
from app.retrieval.types import RetrievedChunk
from tests.fakes.providers import FakeLLMProvider, FakeRetrieval, FakeToolExecutor


@pytest.mark.unit
async def test_events_emit_full_protocol() -> None:
    settings = get_settings().model_copy(update={"max_agent_steps": 2, "retrieval_top_k": 1})
    chunk = RetrievedChunk(
        "c1", "D1", "postmortem", "T", "payments deployment regression", 0, 30, 1.0, "rrf"
    )
    llm = FakeLLMProvider(
        responses=[LLMResponse("The cause was deployment [1].", usage=LLMUsage(5, 5), model="fake")]
    )
    orch = Orchestrator(
        llm=llm,
        retrieval=FakeRetrieval([chunk]),
        executor=FakeToolExecutor([ToolSpec("aggregate", "a", {"type": "object"})]),
        settings=settings,
    )

    types: list[str] = []
    sources_payload = None
    async for event in orch.events("q"):
        types.append(event.type)
        if event.type == "sources":
            sources_payload = event.data

    assert types[0] == "status"
    assert types[-1] == "done"
    assert {"status", "token", "sources", "trace", "done"} <= set(types)
    # The sources payload (dataclasses) serializes via the SSE default encoder.
    assert "c1" in json.dumps(sources_payload, default=_default)


@pytest.mark.unit
def test_default_serializer_handles_dataclasses() -> None:
    source = Source(1, "c1", "d1", "ticket", "title", "snippet", 0, 1)
    assert _default(source)["chunk_id"] == "c1"
    with pytest.raises(TypeError):
        _default(object())
