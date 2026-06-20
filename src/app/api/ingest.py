"""Ingest endpoint: add documents to the corpus."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import shared_embedding_provider
from app.core.errors import ValidationAppError
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import PgChunkWriter
from app.schemas.chat import IngestRequest, IngestResponse

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    for doc in req.documents:
        if len(doc.text) > settings.max_document_chars:
            raise ValidationAppError(
                f"document '{doc.doc_id}' exceeds {settings.max_document_chars} characters"
            )

    docs = [
        SourceDoc(
            doc_id=d.doc_id,
            source_type=d.source_type,
            title=d.title,
            text=d.text,
            markdown=d.markdown,
        )
        for d in req.documents
    ]
    stats = await ingest_documents(docs, shared_embedding_provider(), PgChunkWriter())
    return IngestResponse(
        documents=stats.documents, chunks=stats.chunks, embedding_model=stats.embedding_model
    )
