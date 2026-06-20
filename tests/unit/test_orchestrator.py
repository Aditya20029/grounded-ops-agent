"""Unit tests for the agent orchestrator and its guardrails (fakes only)."""

from __future__ import annotations

import pytest

from app.agent.orchestrator import Orchestrator
from app.agent.tool_executor import ToolOutcome
from app.core.settings import Settings, get_settings
from app.llm.types import LLMResponse, LLMUsage, ToolCall, ToolSpec
from app.retrieval.types import RetrievedChunk
from tests.fakes.providers import FakeLLMProvider, FakeRetrieval, FakeToolExecutor

_TOOLS = [
    ToolSpec("aggregate", "grouped aggregation", {"type": "object", "properties": {}}),
    ToolSpec("search_records", "semantic search", {"type": "object", "properties": {}}),
]


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


def _seed_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        "c1",
        "DOC-1",
        "postmortem",
        "Payments outage",
        "payments deployment regression caused a P1 outage",
        0,
        50,
        1.0,
        "rrf",
    )


@pytest.mark.unit
async def test_headline_flow_calls_tool_then_answers() -> None:
    settings = _settings(max_agent_steps=4, retrieval_top_k=2)
    llm = FakeLLMProvider(
        responses=[
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall(
                        "t1",
                        "aggregate",
                        {
                            "table": "incidents",
                            "group_by": ["root_cause_category"],
                            "agg_fn": "count",
                        },
                    ),
                ),
                usage=LLMUsage(20, 10),
                model="fake",
            ),
            LLMResponse(
                text="The top recurring cause was deployment regression [1].",
                usage=LLMUsage(30, 15),
                model="fake",
            ),
        ]
    )
    executor = FakeToolExecutor(
        _TOOLS,
        {
            "aggregate": ToolOutcome(
                '[{"root_cause_category": "deployment", "agg_value": 12}]', None, False
            )
        },
    )
    orch = Orchestrator(
        llm=llm, retrieval=FakeRetrieval([_seed_chunk()]), executor=executor, settings=settings
    )

    result = await orch.run("avg P1 resolution and top root causes last quarter")

    assert result.stop_reason == "answered"
    assert (
        "aggregate",
        {"table": "incidents", "group_by": ["root_cause_category"], "agg_fn": "count"},
    ) in executor.calls
    assert any(t.tool == "aggregate" and "deployment" in t.result_summary for t in result.trace)
    assert result.used_indices == [1]
    assert [s.chunk_id for s in result.sources] == ["c1"]
    assert result.usage.input_tokens == 50  # 20 + 30


@pytest.mark.unit
async def test_step_cap_forces_final_answer() -> None:
    settings = _settings(max_agent_steps=2)
    tool_resp = LLMResponse(
        text="",
        tool_calls=(ToolCall("t", "search_records", {"query": "x"}),),
        usage=LLMUsage(10, 5),
        model="fake",
    )
    llm = FakeLLMProvider(
        responses=[
            tool_resp,
            tool_resp,
            LLMResponse("Final grounded answer.", usage=LLMUsage(5, 5), model="fake"),
        ]
    )
    orch = Orchestrator(
        llm=llm, retrieval=FakeRetrieval([]), executor=FakeToolExecutor(_TOOLS), settings=settings
    )

    result = await orch.run("a question")
    assert result.stop_reason == "step_cap"
    assert result.steps == 2
    assert result.answer == "Final grounded answer."


@pytest.mark.unit
async def test_token_budget_stops_early() -> None:
    settings = _settings(max_agent_steps=6, per_request_token_budget=300, max_output_tokens=256)
    llm = FakeLLMProvider(
        responses=[LLMResponse("Budgeted final answer.", usage=LLMUsage(1, 1), model="fake")]
    )
    orch = Orchestrator(
        llm=llm, retrieval=FakeRetrieval([]), executor=FakeToolExecutor(_TOOLS), settings=settings
    )

    result = await orch.run(
        "a deliberately long question repeated to inflate the estimated prompt "
        "token count well past the tiny budget so the step is unaffordable"
    )
    assert result.stop_reason == "token_budget"
    assert result.answer == "Budgeted final answer."


@pytest.mark.unit
async def test_duplicate_tool_call_is_blocked() -> None:
    settings = _settings(max_agent_steps=3)
    dup = ToolCall("t", "search_records", {"query": "same"})
    llm = FakeLLMProvider(
        responses=[
            LLMResponse("", tool_calls=(dup,), usage=LLMUsage(5, 5), model="fake"),
            LLMResponse("", tool_calls=(dup,), usage=LLMUsage(5, 5), model="fake"),
            LLMResponse("Answer.", usage=LLMUsage(5, 5), model="fake"),
        ]
    )
    executor = FakeToolExecutor(_TOOLS)
    orch = Orchestrator(llm=llm, retrieval=FakeRetrieval([]), executor=executor, settings=settings)

    result = await orch.run("q")
    # The tool was only actually executed once; the repeat was blocked.
    assert executor.calls.count(("search_records", {"query": "same"})) == 1
    assert any("blocked" in t.result_summary for t in result.trace)
