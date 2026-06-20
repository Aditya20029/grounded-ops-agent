"""Integration tests: migrations + idempotent seeding against Postgres."""

from __future__ import annotations

import pytest

from app.core.settings import get_settings
from app.seed.generator import generate_dataset
from app.seed.loader import load_database

pytestmark = pytest.mark.integration


async def test_seed_is_idempotent(db_engine: object) -> None:
    dataset = generate_dataset(get_settings().data_seed)

    counts1 = await load_database(dataset, truncate=True)
    counts2 = await load_database(dataset, truncate=True)

    # Re-running the seed produces identical row counts (no duplicates).
    assert counts1 == counts2
    assert counts1["slas"] == len(dataset.slas)
    assert counts1["customers"] == len(dataset.customers)
    assert counts1["incidents"] == len(dataset.incidents)
    assert counts1["tickets"] == len(dataset.tickets)
    assert counts1["metrics_daily"] == len(dataset.metrics)
