"""Unit tests for token cost accounting."""

from __future__ import annotations

import math

import pytest

from app.core.pricing import cost_usd, model_pricing


@pytest.mark.unit
def test_known_model_present() -> None:
    price = model_pricing("claude-opus-4-7")
    assert price is not None
    assert price["input"] == 5.00
    assert price["output"] == 25.00


@pytest.mark.unit
def test_cost_is_actual_usage_times_rate() -> None:
    # 1M input tokens at $5/1M; 1M output tokens at $25/1M.
    assert math.isclose(cost_usd("claude-opus-4-7", 1_000_000, 0), 5.00)
    assert math.isclose(cost_usd("claude-opus-4-7", 0, 1_000_000), 25.00)
    assert math.isclose(cost_usd("claude-opus-4-7", 500_000, 200_000), 2.5 + 5.0)


@pytest.mark.unit
def test_unknown_model_costs_zero() -> None:
    # Accounting must never crash on a model missing from the pricing config.
    assert cost_usd("totally-unknown-model", 1_000_000, 1_000_000) == 0.0
