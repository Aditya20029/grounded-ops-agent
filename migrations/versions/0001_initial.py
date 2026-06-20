"""initial schema: vector extension, structured tables, chunks + indexes

Revision ID: 0001
Revises:
Create Date: 2026-06-19

The ``chunks.embedding`` column dimension is read from the active embedding
model in settings (the "one embedding model at a time" invariant). Switching to
an embedding model with a different dimension requires recreating this table and
re-embedding the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from app.core.settings import get_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "slas",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("tier", sa.String(), nullable=False),
        sa.Column("target_resolution_minutes", sa.Integer(), nullable=False),
        sa.Column("uptime_target_pct", sa.Float(), nullable=False),
        sa.CheckConstraint("tier in ('enterprise','business','standard')", name="ck_slas_tier"),
    )

    op.create_table(
        "customers",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("plan", sa.String(), nullable=False),
        sa.Column("region", sa.String(), nullable=False),
        sa.Column("sla_id", sa.String(), sa.ForeignKey("slas.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "incidents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("service", sa.String(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("root_cause_category", sa.String(), nullable=True),
        sa.Column("customer_id", sa.String(), sa.ForeignKey("customers.id"), nullable=True),
        sa.CheckConstraint("severity in ('P1','P2','P3','P4')", name="ck_incidents_severity"),
    )
    op.create_index(
        "ix_incidents_sev_service_opened",
        "incidents",
        ["severity", "service", "opened_at"],
    )

    op.create_table(
        "tickets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("customer_id", sa.String(), sa.ForeignKey("customers.id"), nullable=True),
        sa.Column("priority", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("priority in ('P1','P2','P3','P4')", name="ck_tickets_priority"),
        sa.CheckConstraint(
            "status in ('open','pending','resolved','closed')", name="ck_tickets_status"
        ),
    )
    op.create_index(
        "ix_tickets_priority_status_created",
        "tickets",
        ["priority", "status", "created_at"],
    )

    op.create_table(
        "metrics_daily",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("service", sa.String(), nullable=False),
        sa.Column("metric_name", sa.String(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.UniqueConstraint("date", "service", "metric_name", name="uq_metrics_daily_key"),
    )
    op.create_index(
        "ix_metrics_service_name_date",
        "metrics_daily",
        ["service", "metric_name", "date"],
    )

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(), primary_key=True),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_DIM), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_source_type", "chunks", ["source_type"])

    # HNSW index for cosine similarity search over the embedding column.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    # GIN index for Postgres full-text (keyword) search over chunk content.
    op.execute(
        "CREATE INDEX ix_chunks_content_fts ON chunks USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("metrics_daily")
    op.drop_table("tickets")
    op.drop_table("incidents")
    op.drop_table("customers")
    op.drop_table("slas")
    # The vector extension is left installed; it is harmless and may be shared.
