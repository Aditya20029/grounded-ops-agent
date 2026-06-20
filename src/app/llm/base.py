"""The single LLM provider interface used across the app."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.llm.types import LLMMessage, LLMResponse, ToolSpec


class LLMProvider(ABC):
    """Non-streaming + streaming completion with native tool-calling support.

    Token counting is provider-specific (Anthropic's count-tokens endpoint vs a
    local tokenizer for OpenAI), so it lives behind this interface and is async.
    """

    model_name: str

    @abstractmethod
    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return a single completion (may include tool calls)."""

    @abstractmethod
    def stream(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Yield text deltas of a final answer (no tool calls on this path)."""

    @abstractmethod
    async def count_tokens(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        """Estimate input tokens for the given request."""
