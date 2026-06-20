"""Deterministic, offline test doubles for providers and the chunk store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.agent.tool_executor import ToolOutcome
from app.ingestion.store import ChunkRow
from app.llm.base import LLMProvider
from app.llm.embeddings import HashingEmbeddingProvider
from app.llm.types import LLMMessage, LLMResponse, LLMUsage, ToolSpec
from app.retrieval.types import RetrievedChunk, SearchFilters


class FakeEmbeddingProvider(HashingEmbeddingProvider):
    """Deterministic embeddings for tests (no model download, no network)."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__(dim)
        self.model_name = f"fake-embed-{dim}"


class InMemoryChunkWriter:
    """In-memory ChunkWriter for unit-testing idempotent ingestion."""

    def __init__(self) -> None:
        self._rows: dict[str, ChunkRow] = {}

    async def upsert_chunks(self, rows: list[ChunkRow]) -> None:
        for row in rows:
            self._rows[row.chunk_id] = row

    async def delete_orphans(self, doc_id: str, keep: int) -> int:
        victims = [
            cid
            for cid, row in self._rows.items()
            if row.doc_id == doc_id and row.chunk_index >= keep
        ]
        for cid in victims:
            del self._rows[cid]
        return len(victims)

    async def count(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[ChunkRow]:
        return list(self._rows.values())


class FakeLLMProvider(LLMProvider):
    """Scripted LLM. Returns queued responses, else a default answer."""

    def __init__(
        self,
        *,
        answer_text: str = "Fake grounded answer [1].",
        responses: list[LLMResponse] | None = None,
        model_name: str = "fake-llm",
    ) -> None:
        self.model_name = model_name
        self._answer_text = answer_text
        self._responses = list(responses or [])
        self.calls: list[tuple[list[LLMMessage], str | None, tuple[ToolSpec, ...]]] = []

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append((messages, system, tuple(tools or ())))
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(text=self._answer_text, usage=LLMUsage(10, 5), model=self.model_name)

    async def stream(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        for word in self._answer_text.split(" "):
            yield word + " "

    async def count_tokens(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        return sum(len(m.content or "") for m in messages) // 4 + 1


class FakeRetrieval:
    """Duck-typed RetrievalService returning a fixed chunk list."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        use_faiss: bool = False,
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


class FakeToolExecutor:
    """Returns scripted tool outcomes; records calls."""

    def __init__(
        self, tools: list[ToolSpec], outcomes: dict[str, ToolOutcome] | None = None
    ) -> None:
        self._tools = tools
        self._outcomes = outcomes or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, name: str, arguments: dict[str, Any], *, timeout: float) -> ToolOutcome:
        self.calls.append((name, arguments))
        return self._outcomes.get(
            name,
            ToolOutcome(text=f"(no scripted result for {name})", structured=None, is_error=False),
        )
