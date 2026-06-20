"""Agent guardrails: token budget and cycle detection.

The step cap, tool timeouts, and retries live in the orchestrator loop; these are
the stateful pieces it consults. The token budget estimates the next step's input
with the provider's counter, reserves headroom for output, and is reconciled
against actual usage after each call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.llm.types import LLMUsage


def cycle_key(name: str, arguments: dict[str, Any]) -> str:
    """Stable hash key for a tool call: name + normalized (sorted) arguments."""
    return f"{name}::{json.dumps(arguments, sort_keys=True, default=str)}"


@dataclass
class CycleDetector:
    """Blocks a tool call that exactly repeats one already made this request."""

    _seen: set[str] = field(default_factory=set)

    def is_repeat(self, name: str, arguments: dict[str, Any]) -> bool:
        key = cycle_key(name, arguments)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False


@dataclass
class TokenBudget:
    """Per-request token budget enforced before each step and reconciled after."""

    limit: int
    reserve_output: int
    consumed: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.consumed)

    def can_afford(self, projected_input_tokens: int) -> bool:
        """Whether the next step (estimated input + reserved output) fits."""
        return self.consumed + projected_input_tokens + self.reserve_output <= self.limit

    def reconcile(self, usage: LLMUsage) -> None:
        """Add the actual usage returned by the provider."""
        self.consumed += usage.input_tokens + usage.output_tokens
