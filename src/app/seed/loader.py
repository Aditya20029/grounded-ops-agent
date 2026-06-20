"""Idempotent load of the generated dataset into Postgres.

Structured tables are truncate-and-reloaded (guarded by ``truncate``). Because
the dataset is deterministic for a given seed, re-running produces identical rows
and never duplicates. The ``chunks`` table is intentionally left alone: it is
owned by ingestion (Phase 3), not seeding.
"""

from __future__ import annotations

from sqlalchemy import func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import session_scope
from app.db.models import Customer, Incident, MetricDaily, Sla, Ticket
from app.seed.generator import Dataset

# Truncated together (CASCADE handles FK order); excludes chunks by design.
_STRUCTURED_TABLES = "metrics_daily, tickets, incidents, customers, slas"


async def _count(session: AsyncSession, model: type) -> int:
    result = await session.execute(select(func.count()).select_from(model))
    return int(result.scalar_one())


async def load_database(dataset: Dataset, *, truncate: bool = True) -> dict[str, int]:
    """Load structured rows; return per-table row counts read back from the DB."""
    async with session_scope() as session:
        if truncate:
            await session.execute(text(f"TRUNCATE {_STRUCTURED_TABLES} RESTART IDENTITY CASCADE"))

        await session.execute(
            insert(Sla),
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "tier": s.tier,
                    "target_resolution_minutes": s.target_resolution_minutes,
                    "uptime_target_pct": s.uptime_target_pct,
                }
                for s in dataset.slas
            ],
        )
        await session.execute(
            insert(Customer),
            [
                {
                    "id": c.id,
                    "name": c.name,
                    "plan": c.plan,
                    "region": c.region,
                    "sla_id": c.sla_id,
                    "created_at": c.created_at,
                }
                for c in dataset.customers
            ],
        )
        await session.execute(
            insert(Incident),
            [
                {
                    "id": i.id,
                    "severity": i.severity,
                    "service": i.service,
                    "opened_at": i.opened_at,
                    "resolved_at": i.resolved_at,
                    "root_cause_category": i.root_cause_category,
                    "customer_id": i.customer_id,
                }
                for i in dataset.incidents
            ],
        )
        await session.execute(
            insert(Ticket),
            [
                {
                    "id": t.id,
                    "title": t.title,
                    "body": t.body,
                    "customer_id": t.customer_id,
                    "priority": t.priority,
                    "status": t.status,
                    "created_at": t.created_at,
                    "resolved_at": t.resolved_at,
                }
                for t in dataset.tickets
            ],
        )
        await session.execute(
            insert(MetricDaily),
            [
                {
                    "date": m.date,
                    "service": m.service,
                    "metric_name": m.metric_name,
                    "value": m.value,
                }
                for m in dataset.metrics
            ],
        )

        return {
            "slas": await _count(session, Sla),
            "customers": await _count(session, Customer),
            "incidents": await _count(session, Incident),
            "tickets": await _count(session, Ticket),
            "metrics_daily": await _count(session, MetricDaily),
        }
