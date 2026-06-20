"""Integration-test fixtures.

Integration tests need a real Postgres+pgvector with migrations applied. When no
database is reachable they are skipped rather than failed, so the offline unit
suite (and a keyless `make test`) stays green.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.core.db import dispose_engine, get_engine


async def _database_reachable() -> bool:
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
async def db_engine() -> object:
    """Yield the engine if the DB is reachable, else skip the test."""
    if not await _database_reachable():
        pytest.skip("no Postgres reachable on DATABASE_URL")
    yield get_engine()
    await dispose_engine()
