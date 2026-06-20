"""FastAPI application: routers, request-id middleware, structured errors.

Endpoints: ``POST /chat`` (SSE), ``POST /ingest``, ``POST /search``,
``GET /health``. Run with ``uvicorn app.main:app``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from app import __version__
from app.api import chat, health, ingest, search
from app.core.db import dispose_engine
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger, get_request_id, set_request_id
from app.core.settings import get_settings
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)

# Map error codes to HTTP status; everything else is a 500.
_STATUS_BY_CODE = {
    "validation_error": 400,
    "not_found": 404,
    "embedding_mismatch": 409,
    "tool_error": 502,
    "provider_error": 502,
    "retrieval_error": 502,
    "guardrail_error": 500,
    "config_error": 500,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="grounded-ops-agent",
        version=__version__,
        summary="Grounded RAG + agentic MCP tool-use assistant with citation grounding.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status = _STATUS_BY_CODE.get(exc.code, 500)
        logger.warning("api.error", extra={"code": exc.code, "detail": exc.message})
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error=exc.code, detail=exc.message, request_id=get_request_id()
            ).model_dump(),
        )

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(chat.router)
    return app


app = create_app()
