"""Smoke test for the FastAPI app and the /health endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.main import app


@pytest.mark.unit
def test_health_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
