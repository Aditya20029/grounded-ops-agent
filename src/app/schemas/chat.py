"""Request/response models for the API surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source_types: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    source_types: list[str] | None = None


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    snippet: str
    score: float
    retriever: str


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]


class IngestDocument(BaseModel):
    doc_id: str
    source_type: str
    title: str
    text: str = Field(min_length=1)
    markdown: bool = False


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    embedding_model: str
