"""Raw retrieval endpoint (debugging and the benchmark)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import shared_retrieval_service
from app.core.errors import ValidationAppError
from app.core.settings import get_settings
from app.retrieval.types import SearchFilters
from app.schemas.chat import ChunkResult, SearchRequest, SearchResponse

router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    settings = get_settings()
    if len(req.query) > settings.max_query_chars:
        raise ValidationAppError(f"query exceeds {settings.max_query_chars} characters")

    filters = SearchFilters(source_types=tuple(req.source_types)) if req.source_types else None
    chunks = await shared_retrieval_service().retrieve(req.query, top_k=req.top_k, filters=filters)
    return SearchResponse(
        query=req.query,
        results=[
            ChunkResult(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_type=c.source_type,
                title=c.title,
                snippet=" ".join(c.content.split())[:300],
                score=round(c.score, 4),
                retriever=c.retriever,
            )
            for c in chunks
        ],
    )
