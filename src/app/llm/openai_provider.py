"""OpenAI LLM provider (chat completions with native tool-calling).

Token counting uses tiktoken locally (no API round-trip), which is an estimate;
the agent reconciles against actual usage returned by the API after each call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

import tiktoken

from app.llm.base import LLMProvider
from app.llm.types import LLMMessage, LLMResponse, LLMUsage, ToolCall, ToolSpec


@lru_cache(maxsize=4)
def _encoder(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _tools_param(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _to_messages(messages: list[LLMMessage], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content or ""})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
        else:  # assistant
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
    return out


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, max_retries=4)

    async def complete(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": _to_messages(messages, system),
            "max_tokens": max_tokens,
        }
        tool_param = _tools_param(tools)
        if tool_param:
            kwargs["tools"] = tool_param
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message
        tool_calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        )
        usage = (
            LLMUsage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            if resp.usage
            else LLMUsage()
        )
        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.finish_reason or "stop",
            model=resp.model,
        )

    async def stream(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": _to_messages(messages, system),
            "max_tokens": max_tokens,
            "stream": True,
        }
        result: Any = await self._client.chat.completions.create(**kwargs)
        async for chunk in result:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def count_tokens(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> int:
        enc = _encoder(self.model_name)
        total = len(enc.encode(system)) if system else 0
        for m in messages:
            total += len(enc.encode(m.content or "")) + 4
            for tc in m.tool_calls:
                total += len(enc.encode(tc.name + json.dumps(tc.arguments)))
        if tools:
            total += len(enc.encode(json.dumps([t.input_schema for t in tools])))
        return total
