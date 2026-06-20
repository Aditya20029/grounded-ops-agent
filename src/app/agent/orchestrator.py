"""Agent orchestrator: seed retrieval, then a guarded tool loop, then a grounded
answer.

The loop is an async generator of ``AgentEvent``s (``events()``) so the API can
stream it over SSE; ``run()`` consumes those events into an ``AgentResult`` for
non-streaming callers and tests.

Guardrails: a hard step cap, cycle detection on hashed (tool, args), a
per-request token budget (estimate before each step, reconcile actual usage
after), tool timeouts with a retry cap, cost accounting, and a structured
tool-call trace. Search results returned by tools are folded into the citation
registry with their stable indices so the model can cite them.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.agent.citation_registry import CitationRegistry, render_records
from app.agent.citations import validate_and_strip
from app.agent.generation import Source, sources_for
from app.agent.guardrails import CycleDetector, TokenBudget
from app.agent.prompts import AGENT_SYSTEM, build_agent_prompt
from app.agent.tool_executor import MCPToolExecutor, ToolExecutor, ToolOutcome
from app.core.errors import ToolError
from app.core.pricing import cost_usd
from app.core.settings import Settings
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.types import LLMMessage, LLMResponse, LLMUsage, ToolCall
from app.mcp_client.client import MCPClient
from app.retrieval.service import RetrievalService, build_retrieval_service
from app.retrieval.types import RetrievedChunk, SearchFilters

_TOKEN_CHUNK = 24


@dataclass(frozen=True)
class TraceStep:
    step: int
    tool: str
    arguments: dict[str, Any]
    latency_ms: float
    result_summary: str
    is_error: bool


@dataclass(frozen=True)
class AgentEvent:
    """One streamed event: status | tool_call | tool_result | token | sources |
    trace | done | error."""

    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    sources: list[Source]
    used_indices: list[int]
    trace: list[TraceStep]
    usage: LLMUsage
    cost_usd: float
    model: str
    steps: int
    stop_reason: str  # "answered" | "step_cap" | "token_budget"


def _summary(text: str) -> str:
    return " ".join(text.split())[:200]


def _extract_chunk_records(outcome: ToolOutcome) -> list[RetrievedChunk]:
    """Pull chunk records out of a tool result (e.g. search_records)."""
    data: Any = outcome.structured
    if data is None:
        try:
            data = json.loads(outcome.text)
        except (ValueError, TypeError):
            return []
    if isinstance(data, dict) and "result" in data:
        data = data["result"]
    if not isinstance(data, list):
        return []
    chunks: list[RetrievedChunk] = []
    for item in data:
        if isinstance(item, dict) and "chunk_id" in item:
            content = str(item.get("snippet") or item.get("content") or "")
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(item["chunk_id"]),
                    doc_id=str(item.get("doc_id", "")),
                    source_type=str(item.get("source_type", "")),
                    title=str(item.get("title", "")),
                    content=content,
                    char_start=int(item.get("char_start", 0) or 0),
                    char_end=int(item.get("char_end", 0) or 0),
                    score=float(item.get("score", 0.0) or 0.0),
                    retriever="tool",
                )
            )
    return chunks


class Orchestrator:
    """Runs the guarded agent loop for a single request."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        retrieval: RetrievalService,
        executor: ToolExecutor,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._retrieval = retrieval
        self._executor = executor
        self._settings = settings

    async def events(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the loop, yielding streamed events."""
        s = self._settings
        budget = TokenBudget(s.per_request_token_budget, s.max_output_tokens)
        cycles = CycleDetector()
        registry = CitationRegistry()
        trace: list[TraceStep] = []
        usage = LLMUsage()
        model = self._llm.model_name

        yield AgentEvent("status", {"phase": "seed_retrieval"})
        seed = await self._retrieval.retrieve(
            question, top_k=top_k or s.retrieval_top_k, filters=filters
        )
        registry.add(seed)

        tools = await self._executor.list_tools()
        messages: list[LLMMessage] = [LLMMessage.user(build_agent_prompt(question, registry))]

        step = 0
        final_text = ""
        stop_reason = "answered"
        while step < s.max_agent_steps:
            projected = await self._llm.count_tokens(
                messages=messages, system=AGENT_SYSTEM, tools=tools
            )
            if not budget.can_afford(projected):
                final = await self._final_completion(messages)
                usage = usage + final.usage
                model = final.model or model
                final_text, stop_reason = final.text, "token_budget"
                break

            yield AgentEvent("status", {"phase": "planning", "step": step})
            response = await self._llm.complete(
                system=AGENT_SYSTEM, messages=messages, tools=tools, max_tokens=s.max_output_tokens
            )
            usage = usage + response.usage
            budget.reconcile(response.usage)
            model = response.model or model

            if not response.tool_calls:
                final_text, stop_reason = response.text, "answered"
                break

            messages.append(
                LLMMessage(
                    role="assistant", content=response.text or None, tool_calls=response.tool_calls
                )
            )
            for tc in response.tool_calls:
                async for ev in self._run_tool(tc, registry, trace, cycles, step, messages):
                    yield ev
            step += 1
        else:
            final = await self._final_completion(messages)
            usage = usage + final.usage
            model = final.model or model
            final_text, stop_reason = final.text, "step_cap"

        cleaned, used = validate_and_strip(final_text, registry)
        for start in range(0, len(cleaned), _TOKEN_CHUNK):
            yield AgentEvent("token", {"text": cleaned[start : start + _TOKEN_CHUNK]})

        sources = sources_for(registry, used)
        yield AgentEvent("sources", {"sources": sources})
        yield AgentEvent("trace", {"trace": trace})
        yield AgentEvent(
            "done",
            {
                "used_indices": used,
                "usage": usage,
                "cost_usd": cost_usd(model, usage.input_tokens, usage.output_tokens),
                "model": model,
                "steps": step,
                "stop_reason": stop_reason,
            },
        )

    async def _run_tool(
        self,
        tc: ToolCall,
        registry: CitationRegistry,
        trace: list[TraceStep],
        cycles: CycleDetector,
        step: int,
        messages: list[LLMMessage],
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent("tool_call", {"tool": tc.name, "arguments": tc.arguments})
        if cycles.is_repeat(tc.name, tc.arguments):
            trace.append(
                TraceStep(step, tc.name, tc.arguments, 0.0, "blocked: duplicate call", True)
            )
            messages.append(
                LLMMessage.tool_result(
                    tc.id,
                    "Duplicate call blocked. Try a different approach or answer.",
                    is_error=True,
                )
            )
            yield AgentEvent(
                "tool_result",
                {"tool": tc.name, "summary": "blocked: duplicate call", "is_error": True},
            )
            return

        outcome, latency = await self._call_with_retries(tc)
        content = self._register_and_format(registry, outcome)
        trace.append(
            TraceStep(
                step, tc.name, tc.arguments, latency, _summary(outcome.text), outcome.is_error
            )
        )
        messages.append(LLMMessage.tool_result(tc.id, content, is_error=outcome.is_error))
        yield AgentEvent(
            "tool_result",
            {
                "tool": tc.name,
                "summary": _summary(outcome.text),
                "is_error": outcome.is_error,
                "latency_ms": round(latency, 1),
            },
        )

    async def _call_with_retries(self, tc: ToolCall) -> tuple[ToolOutcome, float]:
        s = self._settings
        start = perf_counter()
        attempt = 0
        while True:
            try:
                outcome = await self._executor.call(
                    tc.name, tc.arguments, timeout=float(s.tool_timeout_seconds)
                )
                return outcome, (perf_counter() - start) * 1000.0
            except ToolError as exc:
                attempt += 1
                if attempt > s.max_tool_retries:
                    return (
                        ToolOutcome(
                            text=f"Tool error: {exc.message}", structured=None, is_error=True
                        ),
                        (perf_counter() - start) * 1000.0,
                    )

    def _register_and_format(self, registry: CitationRegistry, outcome: ToolOutcome) -> str:
        chunks = _extract_chunk_records(outcome)
        if not chunks:
            return outcome.text
        indices = registry.add(chunks)
        return "Search results (cite these by index):\n" + render_records(
            registry.entries_for(indices)
        )

    async def _final_completion(self, messages: list[LLMMessage]) -> LLMResponse:
        messages.append(
            LLMMessage.user(
                "Stop using tools. Give your final grounded answer now with [n] citations."
            )
        )
        return await self._llm.complete(
            system=AGENT_SYSTEM,
            messages=messages,
            tools=None,
            max_tokens=self._settings.max_output_tokens,
        )

    async def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> AgentResult:
        """Consume the event stream into a single result."""
        answer_parts: list[str] = []
        sources: list[Source] = []
        trace: list[TraceStep] = []
        done: dict[str, Any] = {}
        async for ev in self.events(question, top_k=top_k, filters=filters):
            if ev.type == "token":
                answer_parts.append(ev.data["text"])
            elif ev.type == "sources":
                sources = ev.data["sources"]
            elif ev.type == "trace":
                trace = ev.data["trace"]
            elif ev.type == "done":
                done = ev.data
            elif ev.type == "error":
                raise RuntimeError(ev.data.get("detail", "agent error"))
        return AgentResult(
            answer="".join(answer_parts),
            sources=sources,
            used_indices=done.get("used_indices", []),
            trace=trace,
            usage=done.get("usage", LLMUsage()),
            cost_usd=done.get("cost_usd", 0.0),
            model=done.get("model", self._llm.model_name),
            steps=done.get("steps", 0),
            stop_reason=done.get("stop_reason", "answered"),
        )


def build_orchestrator(settings: Settings, executor: ToolExecutor) -> Orchestrator:
    """Wire an orchestrator from settings with a given (connected) tool executor."""
    return Orchestrator(
        llm=get_llm_provider(settings),
        retrieval=build_retrieval_service(settings),
        executor=executor,
        settings=settings,
    )


def mcp_executor(client: MCPClient) -> MCPToolExecutor:
    """Convenience wrapper to build a ToolExecutor from a connected MCP client."""
    return MCPToolExecutor(client)
