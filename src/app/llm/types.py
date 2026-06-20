"""Provider-agnostic conversation and tool-calling types.

These normalize OpenAI and Anthropic differences so the orchestrator works with
one shape. The tool adapter (Phase 6) converts MCP tool schemas into ``ToolSpec``
and each provider translates ``ToolSpec`` and ``LLMMessage`` into its own wire
format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A model request to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A provider-agnostic tool definition (JSON-schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class LLMMessage:
    """One normalized conversation turn."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # set on role="tool"
    is_error: bool = False

    @staticmethod
    def user(text: str) -> LLMMessage:
        return LLMMessage(role="user", content=text)

    @staticmethod
    def tool_result(tool_call_id: str, content: str, *, is_error: bool = False) -> LLMMessage:
        return LLMMessage(
            role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error
        )


@dataclass(frozen=True)
class LLMUsage:
    """Token usage reported by a provider."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """A normalized model response."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = field(default_factory=LLMUsage)
    stop_reason: str = "end_turn"
    model: str = ""
