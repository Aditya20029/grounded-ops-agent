"""Common response models used across endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Result of ``GET /health``."""

    status: str = Field(description="'ok' if all checks pass, else 'degraded'.")
    version: str
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Per-dependency status, e.g. {'database': 'ok', 'faiss': 'missing'}.",
    )


class ErrorResponse(BaseModel):
    """Structured error envelope returned for any handled failure."""

    error: str = Field(description="Stable machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
    request_id: str | None = None
