"""Token pricing, loaded from ``config/pricing.json`` (never hardcoded in logic).

Cost is always computed from the *actual* token usage reported by a provider,
multiplied by the per-million-token rates in the config file. Prices are
configurable and may be out of date; that caveat is documented in the README.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict, cast

from app.core.errors import ConfigError

# config/pricing.json lives at the repo root, three parents up from this file
# (src/app/core/pricing.py -> src/app/core -> src/app -> src -> <root>).
_PRICING_PATH = Path(__file__).resolve().parents[3] / "config" / "pricing.json"


class ModelPrice(TypedDict):
    input: float
    output: float
    kind: str


@lru_cache(maxsize=1)
def _models() -> dict[str, ModelPrice]:
    if not _PRICING_PATH.exists():
        raise ConfigError(f"Pricing config not found at {_PRICING_PATH}")
    raw = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    return cast("dict[str, ModelPrice]", raw["models"])


def model_pricing(model: str) -> ModelPrice | None:
    """Return the price entry for a model, or ``None`` if it is not configured."""
    return _models().get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for actual token usage.

    Unknown models cost 0.0 (the caller is responsible for logging that the
    model is missing from the pricing config); this never raises so accounting
    cannot crash a request.
    """
    price = _models().get(model)
    if price is None:
        return 0.0
    return (input_tokens / 1_000_000) * price["input"] + (output_tokens / 1_000_000) * price[
        "output"
    ]
