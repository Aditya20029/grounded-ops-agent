"""Anthropic (Claude) LLM provider.

Notes specific to current Claude models:
- Opus 4.7/4.8 and Fable reject sampling parameters, so ``temperature`` is only
  sent for models that accept it. Determinism for graded eval otherwise relies on
  the deterministic prompt (documented in docs/architecture.md).
- Adaptive thinking is opt-in (``LLM_THINKING``); off by default for predictable
  token budgeting and a simpler agent loop.
- Token counting uses the count-tokens endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.llm.base import LLMProvider
from app.llm.types import LLMMessage, LLMResponse, LLMUsage, ToolCall, ToolSpec

# Models that reject temperature/top_p/top_k.
_NO_SAMPLING_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable", "claude-mythos")


def _supports_temperature(model: str) -> bool:
    return not any(model.startswith(p) for p in _NO_SAMPLING_PREFIXES)


def _tools_param(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _to_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Translate normalized messages into Anthropic's format.

    Consecutive tool results are merged into one user turn (the API expects all
    tool_result blocks for the preceding assistant turn in a single message).
    """
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for m in messages:
        if m.role == "tool":
            pending.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "",
                    "is_error": m.is_error,
                }
            )
            continue
        flush()
        if m.role == "user":
            out.append({"role": "user", "content": m.content or ""})
        else:  # assistant
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks or (m.content or "")})
    flush()
    return out


class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str, *, thinking: bool = False) -> None:
        self.model_name = model_name
        self._thinking = thinking
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=4)

    def _base_params(
        self, messages: list[LLMMessage], system: str | None, tools: list[ToolSpec] | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model_name, "messages": _to_messages(messages)}
        if system:
            params["system"] = system
        tool_param = _tools_param(tools)
        if tool_param:
            params["tools"] = tool_param
        return params

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        params = self._base_params(messages, system, tools)
        params["max_tokens"] = max_tokens
        if self._thinking:
            params["thinking"] = {"type": "adaptive"}
        if temperature is not None and _supports_temperature(self.model_name):
            params["temperature"] = temperature

        resp = await self._client.messages.create(**params)
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = tuple(
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        )
        usage = LLMUsage(resp.usage.input_tokens, resp.usage.output_tokens)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=resp.stop_reason or "end_turn",
            model=resp.model,
        )

    async def stream(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        params = self._base_params(messages, system, None)
        params["max_tokens"] = max_tokens
        if self._thinking:
            params["thinking"] = {"type": "adaptive"}
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def count_tokens(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        params = self._base_params(messages, system, tools)
        resp = await self._client.messages.count_tokens(**params)
        return resp.input_tokens
