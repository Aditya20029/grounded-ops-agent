"""Agent orchestrator: seed retrieval, then a guarded tool loop, then a grounded
answer.

Guardrails: a hard step cap, cycle detection on hashed (tool, args), a
per-request token budget (estimate before each step, reconcile actual usage
after), tool timeouts with a retry cap, cost accounting, and a structured
tool-call trace. Search results returned by tools are folded into the citation
registry with their stable indices so the model can cite them.
"""

from __future__ import annotations

import json
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
from app.core.logging import get_logger
from app.core.pricing import cost_usd
from app.core.settings import Settings
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.types import LLMMessage, LLMUsage, ToolCall
from app.mcp_client.client import MCPClient
from app.retrieval.service import RetrievalService, build_retrieval_service
from app.retrieval.types import RetrievedChunk, SearchFilters

logger = get_logger(__name__)


@dataclass(frozen=True)
class TraceStep:
    step: int
    tool: str
    arguments: dict[str, Any]
    latency_ms: float
    result_summary: str
    is_error: bool


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

    async def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> AgentResult:
        s = self._settings
        budget = TokenBudget(s.per_request_token_budget, s.max_output_tokens)
        cycles = CycleDetector()
        registry = CitationRegistry()
        trace: list[TraceStep] = []
        usage = LLMUsage()
        model = self._llm.model_name

        seed = await self._retrieval.retrieve(
            question, top_k=top_k or s.retrieval_top_k, filters=filters
        )
        registry.add(seed)

        tools = await self._executor.list_tools()
        messages: list[LLMMessage] = [LLMMessage.user(build_agent_prompt(question, registry))]

        step = 0
        while step < s.max_agent_steps:
            projected = await self._llm.count_tokens(
                messages=messages, system=AGENT_SYSTEM, tools=tools
            )
            if not budget.can_afford(projected):
                return await self._finalize(
                    messages, registry, trace, usage, model, step, "token_budget"
                )

            response = await self._llm.complete(
                system=AGENT_SYSTEM, messages=messages, tools=tools, max_tokens=s.max_output_tokens
            )
            usage = usage + response.usage
            budget.reconcile(response.usage)
            model = response.model or model

            if not response.tool_calls:
                cleaned, used = validate_and_strip(response.text, registry)
                return self._build(cleaned, registry, used, trace, usage, model, step, "answered")

            messages.append(
                LLMMessage(
                    role="assistant", content=response.text or None, tool_calls=response.tool_calls
                )
            )
            for tc in response.tool_calls:
                await self._run_tool(tc, registry, trace, cycles, step, messages)
            step += 1

        return await self._finalize(messages, registry, trace, usage, model, step, "step_cap")

    async def _run_tool(
        self,
        tc: ToolCall,
        registry: CitationRegistry,
        trace: list[TraceStep],
        cycles: CycleDetector,
        step: int,
        messages: list[LLMMessage],
    ) -> None:
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
            return
        outcome, latency = await self._call_with_retries(tc)
        content = self._register_and_format(registry, outcome)
        trace.append(
            TraceStep(
                step, tc.name, tc.arguments, latency, _summary(outcome.text), outcome.is_error
            )
        )
        messages.append(LLMMessage.tool_result(tc.id, content, is_error=outcome.is_error))

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

    async def _finalize(
        self,
        messages: list[LLMMessage],
        registry: CitationRegistry,
        trace: list[TraceStep],
        usage: LLMUsage,
        model: str,
        step: int,
        reason: str,
    ) -> AgentResult:
        messages.append(
            LLMMessage.user(
                "Stop using tools. Give your final grounded answer now with [n] citations."
            )
        )
        final = await self._llm.complete(
            system=AGENT_SYSTEM,
            messages=messages,
            tools=None,
            max_tokens=self._settings.max_output_tokens,
        )
        usage = usage + final.usage
        model = final.model or model
        cleaned, used = validate_and_strip(final.text, registry)
        return self._build(cleaned, registry, used, trace, usage, model, step, reason)

    def _build(
        self,
        answer: str,
        registry: CitationRegistry,
        used: list[int],
        trace: list[TraceStep],
        usage: LLMUsage,
        model: str,
        step: int,
        reason: str,
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            sources=sources_for(registry, used),
            used_indices=used,
            trace=trace,
            usage=usage,
            cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
            model=model,
            steps=step,
            stop_reason=reason,
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
