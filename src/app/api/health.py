"""Health endpoint: checks DB connectivity and FAISS index availability."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select, text

from app import __version__
from app.core.db import session_scope
from app.db.models import Chunk
from app.retrieval.faiss_store import DEFAULT_INDEX_DIR
from app.schemas.common import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    checks: dict[str, str] = {}
    status = "ok"

    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
            count = (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()
        checks["database"] = "ok"
        checks["chunks"] = str(count)
    except Exception:
        checks["database"] = "unavailable"
        status = "degraded"

    checks["faiss_index"] = "present" if (DEFAULT_INDEX_DIR / "chunks.faiss").exists() else "absent"
    return HealthResponse(status=status, version=__version__, checks=checks)
