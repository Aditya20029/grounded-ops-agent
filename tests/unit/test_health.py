"""Smoke tests for the FastAPI app: /health and request-id middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.main import app


@pytest.mark.unit
def test_health_reports_version_and_checks() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    # Without a DB the status is "degraded"; with one it is "ok".
    assert body["status"] in {"ok", "degraded"}
    assert body["version"] == __version__
    assert "database" in body["checks"]
    assert "faiss_index" in body["checks"]


@pytest.mark.unit
def test_request_id_header_is_attached() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")
