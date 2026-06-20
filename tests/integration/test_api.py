"""Integration tests for the API DB endpoints (in-process via ASGI transport).

Uses an async client in the same event loop as the engine fixture, avoiding the
sync-TestClient / async-engine loop mismatch. Forces offline embeddings.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.api.deps import shared_embedding_provider, shared_retrieval_service
from app.core.db import session_scope
from app.core.settings import get_settings
from app.main import app

pytestmark = pytest.mark.integration


async def _truncate_chunks() -> None:
    async with session_scope() as session:
        await session.execute(text("TRUNCATE chunks"))


async def test_ingest_then_search(db_engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    get_settings.cache_clear()
    shared_embedding_provider.cache_clear()
    shared_retrieval_service.cache_clear()

    await _truncate_chunks()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.headers.get("X-Request-ID")

            ingest = await client.post(
                "/ingest",
                json={
                    "documents": [
                        {
                            "doc_id": "DOC-A",
                            "source_type": "postmortem",
                            "title": "Payments outage",
                            "text": "payments gateway timed out due to a deployment regression",
                            "markdown": False,
                        }
                    ]
                },
            )
            assert ingest.status_code == 200
            assert ingest.json()["chunks"] >= 1

            search = await client.post(
                "/search", json={"query": "payments deployment regression", "top_k": 3}
            )
            assert search.status_code == 200
            assert any(r["doc_id"] == "DOC-A" for r in search.json()["results"])
    finally:
        get_settings.cache_clear()
        shared_embedding_provider.cache_clear()
        shared_retrieval_service.cache_clear()
