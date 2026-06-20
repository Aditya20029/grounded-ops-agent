"""Declarative base and shared column helpers."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""


# Canonical domain vocabularies. These double as the security whitelist surface
# for the analytics tools (Phase 6): only these literal values are ever valid.
SERVICES: tuple[str, ...] = (
    "payments",
    "auth",
    "search",
    "notifications",
    "billing",
    "api-gateway",
)

SEVERITIES: tuple[str, ...] = ("P1", "P2", "P3", "P4")

ROOT_CAUSE_CATEGORIES: tuple[str, ...] = (
    "deployment",
    "configuration",
    "capacity",
    "dependency-failure",
    "database",
    "network",
    "human-error",
    "third-party",
)

TICKET_STATUSES: tuple[str, ...] = ("open", "pending", "resolved", "closed")

SLA_TIERS: tuple[str, ...] = ("enterprise", "business", "standard")

METRIC_NAMES: tuple[str, ...] = (
    "uptime_pct",
    "latency_p95_ms",
    "error_rate",
    "incident_count",
)
