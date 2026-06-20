"""Single-shot grounded RAG generation.

Retrieve -> register citations -> ground -> answer -> post-validate citations ->
assemble the sources array and cost. The agent orchestrator (Phase 7) extends
this with the MCP tool loop, but reuses the same citation registry, validation,
and source-assembly pieces.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.citation_registry import CitationRegistry
from app.agent.citations import validate_and_strip
from app.agent.prompts import ANSWER_SYSTEM, build_answer_prompt
from app.core.pricing import cost_usd
from app.llm.base import LLMProvider
from app.llm.types import LLMMessage, LLMUsage
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchFilters


@dataclass(frozen=True)
class Source:
    """One citation registry entry surfaced in the response."""

    index: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    snippet: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    sources: list[Source]
    used_indices: list[int]
    usage: LLMUsage
    cost_usd: float
    model: str


def sources_for(registry: CitationRegistry, used: list[int]) -> list[Source]:
    """Build the sources array from the entries actually cited."""
    used_set = set(used)
    return [
        Source(
            index=e.index,
            chunk_id=e.chunk_id,
            doc_id=e.doc_id,
            source_type=e.source_type,
            title=e.title,
            snippet=e.snippet,
            char_start=e.char_start,
            char_end=e.char_end,
        )
        for e in registry.entries()
        if e.index in used_set
    ]


async def generate_grounded_answer(
    question: str,
    retrieval: RetrievalService,
    llm: LLMProvider,
    *,
    top_k: int = 8,
    max_tokens: int = 1500,
    filters: SearchFilters | None = None,
) -> GroundedAnswer:
    """Produce a grounded answer with validated inline citations and sources."""
    chunks = await retrieval.retrieve(question, top_k=top_k, filters=filters)
    registry = CitationRegistry()
    registry.add(chunks)

    prompt = build_answer_prompt(question, registry)
    response = await llm.complete(
        system=ANSWER_SYSTEM,
        messages=[LLMMessage.user(prompt)],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    cleaned, used = validate_and_strip(response.text, registry)
    model = response.model or llm.model_name
    return GroundedAnswer(
        answer=cleaned,
        sources=sources_for(registry, used),
        used_indices=used,
        usage=response.usage,
        cost_usd=cost_usd(model, response.usage.input_tokens, response.usage.output_tokens),
        model=model,
    )
