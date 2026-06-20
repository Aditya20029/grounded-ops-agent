"""FastAPI application entrypoint.

Phase 0 ships a minimal but runnable app exposing ``GET /health``. Later phases
add ``/chat`` (SSE), ``/ingest``, and ``/search`` routers, the asyncpg pool, and
request-id middleware. Run with: ``uvicorn app.main:app``.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from app.schemas.common import HealthResponse

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="grounded-ops-agent",
        version=__version__,
        summary="Grounded RAG + agentic MCP tool-use assistant with citation grounding.",
    )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness probe. Dependency checks (DB, FAISS) are wired in Phase 8."""
        return HealthResponse(status="ok", version=__version__)

    logger.info(
        "app.configured",
        extra={
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
        },
    )
    return app


app = create_app()
