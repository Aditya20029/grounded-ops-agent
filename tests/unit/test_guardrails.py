"""Unit tests for agent guardrails: cycle detection and token budget."""

from __future__ import annotations

import pytest

from app.agent.guardrails import CycleDetector, TokenBudget, cycle_key
from app.llm.types import LLMUsage


@pytest.mark.unit
def test_cycle_detector_blocks_exact_repeat() -> None:
    cd = CycleDetector()
    assert cd.is_repeat("aggregate", {"table": "incidents"}) is False
    assert cd.is_repeat("aggregate", {"table": "incidents"}) is True  # repeat blocked
    assert cd.is_repeat("aggregate", {"table": "tickets"}) is False  # different args ok


@pytest.mark.unit
def test_cycle_key_is_argument_order_independent() -> None:
    assert cycle_key("t", {"a": 1, "b": 2}) == cycle_key("t", {"b": 2, "a": 1})


@pytest.mark.unit
def test_token_budget_affordability_and_reconcile() -> None:
    budget = TokenBudget(limit=1000, reserve_output=200)
    assert budget.can_afford(700) is True  # 0 + 700 + 200 = 900 <= 1000
    assert budget.can_afford(801) is False  # 1001 > 1000

    budget.reconcile(LLMUsage(input_tokens=500, output_tokens=100))
    assert budget.consumed == 600
    assert budget.remaining == 400
    assert budget.can_afford(300) is False  # 600 + 300 + 200 = 1100 > 1000
