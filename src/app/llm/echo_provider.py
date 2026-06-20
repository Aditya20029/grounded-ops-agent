"""Deterministic, offline LLM provider for keyless demos and smoke tests.

It parses the labelled, fenced records out of the answering prompt and emits a
grounded answer that cites the first few record indices. This lets the entire
pipeline (retrieve -> ground -> cite -> validate -> sources) run from a clean
clone with no API key. It does not plan or call tools.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.agent.citation_registry import RECORD_CLOSE, RECORD_OPEN
from app.llm.base import LLMProvider
from app.llm.types import LLMMessage, LLMResponse, LLMUsage, ToolSpec

_RECORD_RE = re.compile(
    rf"\[(\d+)\][^\n]*\n{re.escape(RECORD_OPEN)}\n(.*?)\n{re.escape(RECORD_CLOSE)}",
    re.DOTALL,
)


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class EchoLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.model_name = "echo-offline"

    @staticmethod
    def _last_user(messages: list[LLMMessage]) -> str:
        for m in reversed(messages):
            if m.role == "user" and m.content:
                return m.content
        return ""

    def _answer(self, messages: list[LLMMessage]) -> str:
        records = _RECORD_RE.findall(self._last_user(messages))
        if not records:
            return "I do not have enough grounded information to answer that."
        parts = []
        for index, content in records[:3]:
            first_sentence = " ".join(content.split()).split(". ")[0].rstrip(".")
            parts.append(f"{first_sentence} [{index}].")
        return "Based on the retrieved operational records, " + " ".join(parts)

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        answer = self._answer(messages)
        usage = LLMUsage(
            _tokens(self._last_user(messages)) + _tokens(system or ""),
            _tokens(answer),
        )
        return LLMResponse(text=answer, usage=usage, stop_reason="end_turn", model=self.model_name)

    async def stream(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        for word in self._answer(messages).split(" "):
            yield word + " "

    async def count_tokens(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        return sum(_tokens(m.content or "") for m in messages) + _tokens(system or "")
