"""Deterministic, seeded synthetic dataset for a fictional SaaS company's ops.

Everything here is reproducible: given the same seed (and Faker version), the
generated corpus and structured rows are identical across runs and machines. A
fixed reference timestamp (not ``now``) anchors all dates so "last quarter" is
stable for the demo. Swap the domain by replacing this generator; no other code
changes.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from faker import Faker

from app.db.base import (
    METRIC_NAMES,
    SERVICES,
    SLA_TIERS,
)

# Anchor for all generated dates. Fixed so the dataset is reproducible and the
# "last quarter" demo window is stable regardless of when seeding runs.
REFERENCE = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

DEFAULT_SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "seed"

N_CUSTOMERS = 40
N_INCIDENTS = 320
N_TICKETS = 500
N_POSTMORTEMS = 60
N_RUNBOOKS = 30

REGIONS = ("NA", "EU", "APAC", "LATAM")
PLANS = ("Enterprise", "Business", "Starter")
_PLAN_TO_TIER = {"Enterprise": "enterprise", "Business": "business", "Starter": "standard"}

# Severity mix and per-severity expected resolution time (minutes).
_SEVERITY_WEIGHTS = {"P1": 0.12, "P2": 0.23, "P3": 0.35, "P4": 0.30}
_SEVERITY_MTTR_MIN = {"P1": 150.0, "P2": 360.0, "P3": 900.0, "P4": 2000.0}
_ROOT_CAUSE_WEIGHTS = {
    "deployment": 0.22,
    "configuration": 0.18,
    "capacity": 0.14,
    "dependency-failure": 0.12,
    "database": 0.11,
    "network": 0.09,
    "human-error": 0.08,
    "third-party": 0.06,
}

_SYMPTOMS = (
    "elevated error rates",
    "increased latency",
    "failed logins",
    "request timeouts",
    "degraded throughput",
    "intermittent 5xx responses",
    "slow database queries",
    "webhook delivery failures",
    "payment declines",
    "missing notifications",
)


@dataclass(frozen=True)
class Sla:
    id: str
    name: str
    tier: str
    target_resolution_minutes: int
    uptime_target_pct: float


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    plan: str
    region: str
    sla_id: str
    created_at: datetime


@dataclass(frozen=True)
class Incident:
    id: str
    severity: str
    service: str
    opened_at: datetime
    resolved_at: datetime | None
    root_cause_category: str
    customer_id: str | None


@dataclass(frozen=True)
class Ticket:
    id: str
    title: str
    body: str
    customer_id: str | None
    priority: str
    status: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass(frozen=True)
class MetricDaily:
    date: datetime
    service: str
    metric_name: str
    value: float


@dataclass(frozen=True)
class Doc:
    """An unstructured corpus document (postmortem or runbook)."""

    doc_id: str
    source_type: str
    title: str
    rel_path: str
    markdown: str


@dataclass(frozen=True)
class Dataset:
    slas: list[Sla]
    customers: list[Customer]
    incidents: list[Incident]
    tickets: list[Ticket]
    metrics: list[MetricDaily]
    postmortems: list[Doc]
    runbooks: list[Doc]


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "unresolved"


def _slas() -> list[Sla]:
    specs = {
        "enterprise": (240, 99.95),
        "business": (480, 99.9),
        "standard": (1440, 99.5),
    }
    return [
        Sla(
            id=f"SLA-{tier[:3].upper()}",
            name=f"{tier.capitalize()} SLA",
            tier=tier,
            target_resolution_minutes=specs[tier][0],
            uptime_target_pct=specs[tier][1],
        )
        for tier in SLA_TIERS
    ]


def _customers(rng: random.Random, fake: Faker, slas: list[Sla]) -> list[Customer]:
    by_tier = {s.tier: s for s in slas}
    out: list[Customer] = []
    for i in range(1, N_CUSTOMERS + 1):
        plan = rng.choices(PLANS, weights=[0.25, 0.4, 0.35], k=1)[0]
        created = REFERENCE - timedelta(days=rng.randint(120, 1000))
        out.append(
            Customer(
                id=f"CUST-{i:03d}",
                name=fake.company(),
                plan=plan,
                region=rng.choice(REGIONS),
                sla_id=by_tier[_PLAN_TO_TIER[plan]].id,
                created_at=created,
            )
        )
    return out


def _incidents(rng: random.Random, customers: list[Customer]) -> list[Incident]:
    out: list[Incident] = []
    for i in range(1, N_INCIDENTS + 1):
        severity = _weighted(rng, _SEVERITY_WEIGHTS)
        opened = REFERENCE - timedelta(days=rng.random() * 365, seconds=rng.randint(0, 86_400))
        # Recent incidents may still be open; everything else is resolved.
        age_days = (REFERENCE - opened).total_seconds() / 86_400
        if age_days < 4 and rng.random() < 0.5:
            resolved: datetime | None = None
        else:
            mean = _SEVERITY_MTTR_MIN[severity]
            minutes = max(15.0, rng.lognormvariate(math.log(mean), 0.5))
            resolved = opened + timedelta(minutes=minutes)
        out.append(
            Incident(
                id=f"INC-{i:04d}",
                severity=severity,
                service=rng.choice(SERVICES),
                opened_at=opened,
                resolved_at=resolved,
                root_cause_category=_weighted(rng, _ROOT_CAUSE_WEIGHTS),
                customer_id=(rng.choice(customers).id if rng.random() < 0.7 else None),
            )
        )
    return out


def _tickets(rng: random.Random, fake: Faker, customers: list[Customer]) -> list[Ticket]:
    out: list[Ticket] = []
    for i in range(1, N_TICKETS + 1):
        service = rng.choice(SERVICES)
        symptom = rng.choice(_SYMPTOMS)
        priority = _weighted(rng, _SEVERITY_WEIGHTS)
        status = rng.choices(
            ["open", "pending", "resolved", "closed"], weights=[0.15, 0.1, 0.3, 0.45]
        )[0]
        created = REFERENCE - timedelta(days=rng.random() * 365, seconds=rng.randint(0, 86_400))
        resolved = (
            created + timedelta(hours=rng.randint(1, 96))
            if status in ("resolved", "closed")
            else None
        )
        customer = rng.choice(customers) if rng.random() < 0.85 else None
        who = customer.name if customer else "A customer"
        body = (
            f"{who} reports {symptom} affecting the {service} service. "
            f"{fake.paragraph(nb_sentences=4)} "
            f"The issue was first observed around {_fmt(created)} and assessed as "
            f"priority {priority}. "
            + (
                "On-call applied a temporary workaround while investigating."
                if rng.random() < 0.5
                else "Escalated to the service owning team for root-cause analysis."
            )
        )
        out.append(
            Ticket(
                id=f"TCK-{i:04d}",
                title=f"{symptom.capitalize()} on {service}",
                body=body,
                customer_id=customer.id if customer else None,
                priority=priority,
                status=status,
                created_at=created,
                resolved_at=resolved,
            )
        )
    return out


def _metrics(rng: random.Random) -> list[MetricDaily]:
    out: list[MetricDaily] = []
    start = (REFERENCE - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    for day in range(365):
        date = start + timedelta(days=day)
        for service in SERVICES:
            for metric in METRIC_NAMES:
                out.append(
                    MetricDaily(
                        date=date,
                        service=service,
                        metric_name=metric,
                        value=_metric_value(rng, metric),
                    )
                )
    return out


def _metric_value(rng: random.Random, metric: str) -> float:
    if metric == "uptime_pct":
        return round(min(100.0, 99.0 + rng.random()), 3)
    if metric == "latency_p95_ms":
        return round(rng.uniform(80, 600), 1)
    if metric == "error_rate":
        return round(rng.uniform(0.0, 0.05), 4)
    # incident_count
    return float(rng.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0])


def _postmortems(rng: random.Random, fake: Faker, incidents: list[Incident]) -> list[Doc]:
    candidates = [i for i in incidents if i.severity in ("P1", "P2") and i.resolved_at]
    rng.shuffle(candidates)
    out: list[Doc] = []
    for inc in candidates[:N_POSTMORTEMS]:
        out.append(_postmortem_doc(rng, fake, inc))
    return out


def _postmortem_doc(rng: random.Random, fake: Faker, inc: Incident) -> Doc:
    symptom = rng.choice(_SYMPTOMS)
    detect = inc.opened_at + timedelta(minutes=rng.randint(2, 30))
    mitigate = inc.opened_at + timedelta(minutes=rng.randint(30, 120))
    resolved = inc.resolved_at or inc.opened_at + timedelta(hours=2)
    title = f"Postmortem {inc.id}: {inc.service} {inc.severity}"
    md = f"""# {title}

**Service:** {inc.service}
**Severity:** {inc.severity}
**Opened:** {_fmt(inc.opened_at)}
**Resolved:** {_fmt(inc.resolved_at)}
**Root cause category:** {inc.root_cause_category}

## Summary

A {inc.severity} incident on the {inc.service} service caused {symptom} for
affected customers. {fake.sentence(nb_words=18)}

## Timeline

- {_fmt(inc.opened_at)}: Alerting fired for {symptom} on {inc.service}.
- {_fmt(detect)}: On-call acknowledged and began investigation.
- {_fmt(mitigate)}: Mitigation applied; customer impact began to subside.
- {_fmt(resolved)}: Service fully restored and incident resolved.

## Root cause

The root cause was categorized as **{inc.root_cause_category}**.
{fake.paragraph(nb_sentences=4)} The {inc.root_cause_category} issue propagated
to the {inc.service} service before automated checks caught it.

## Remediation

- {fake.sentence(nb_words=10)}
- Added monitoring for {symptom} on {inc.service}.
- Documented the {inc.root_cause_category} failure mode in the runbook.

## Impact

Customer-facing impact lasted from {_fmt(inc.opened_at)} until {_fmt(resolved)}.
{fake.sentence(nb_words=14)}
"""
    return Doc(
        doc_id=inc.id,
        source_type="postmortem",
        title=title,
        rel_path=f"postmortems/{inc.id}.md",
        markdown=md,
    )


def _runbooks(rng: random.Random, fake: Faker) -> list[Doc]:
    topics = [
        "Responding to elevated latency",
        "Database failover procedure",
        "Rotating service credentials",
        "Scaling the API gateway",
        "Investigating elevated error rates",
        "Recovering stuck webhook deliveries",
        "Handling a payments outage",
        "Draining a degraded node",
        "Rolling back a bad deployment",
        "Diagnosing slow database queries",
        "Restoring from a backup",
        "Mitigating a third-party dependency outage",
        "Clearing a notification backlog",
        "Responding to an authentication outage",
        "Capacity planning checklist",
    ]
    out: list[Doc] = []
    for i in range(N_RUNBOOKS):
        topic = topics[i % len(topics)]
        service = SERVICES[i % len(SERVICES)]
        slug = topic.lower().replace(" ", "-")
        doc_id = f"RB-{i + 1:03d}"
        title = f"Runbook: {topic} ({service})"
        md = f"""# {title}

**Applies to:** {service}
**Category:** operations

## When to use

Use this runbook when you observe symptoms related to "{topic.lower()}" on the
{service} service. {fake.sentence(nb_words=16)}

## Procedure

1. {fake.sentence(nb_words=10)}
2. Check the {service} dashboards for error rate and latency anomalies.
3. {fake.sentence(nb_words=10)}
4. If impact is customer-facing, declare an incident and page the on-call.
5. {fake.sentence(nb_words=10)}

## Verification

- Confirm error rate and p95 latency for {service} have returned to baseline.
- {fake.sentence(nb_words=12)}

## Escalation

If the issue is not resolved within the SLA window, escalate to the service
owner and the incident commander.
"""
        out.append(
            Doc(
                doc_id=doc_id,
                source_type="runbook",
                title=title,
                rel_path=f"runbooks/{doc_id}-{slug}.md",
                markdown=md,
            )
        )
    return out


def generate_dataset(seed: int) -> Dataset:
    """Generate the full reproducible dataset for the given seed."""
    rng = random.Random(seed)
    fake = Faker("en_US")
    fake.seed_instance(seed)

    slas = _slas()
    customers = _customers(rng, fake, slas)
    incidents = _incidents(rng, customers)
    tickets = _tickets(rng, fake, customers)
    metrics = _metrics(rng)
    postmortems = _postmortems(rng, fake, incidents)
    runbooks = _runbooks(rng, fake)
    return Dataset(
        slas=slas,
        customers=customers,
        incidents=incidents,
        tickets=tickets,
        metrics=metrics,
        postmortems=postmortems,
        runbooks=runbooks,
    )


def write_corpus(dataset: Dataset, seed_dir: Path = DEFAULT_SEED_DIR) -> int:
    """Write the unstructured corpus to ``seed_dir`` (idempotent overwrite).

    Returns the number of files written. ``tickets.jsonl`` plus the postmortem
    and runbook Markdown files form the retrieval corpus consumed by ingestion.
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("postmortems", "runbooks"):
        target = seed_dir / sub
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    written = 0
    tickets_path = seed_dir / "tickets.jsonl"
    with tickets_path.open("w", encoding="utf-8") as fh:
        for t in dataset.tickets:
            fh.write(
                json.dumps(
                    {
                        "id": t.id,
                        "title": t.title,
                        "body": t.body,
                        "customer_id": t.customer_id,
                        "priority": t.priority,
                        "status": t.status,
                        "created_at": t.created_at.isoformat(),
                    }
                )
                + "\n"
            )
    written += 1

    for doc in (*dataset.postmortems, *dataset.runbooks):
        path = seed_dir / doc.rel_path
        path.write_text(doc.markdown, encoding="utf-8")
        written += 1

    return written
