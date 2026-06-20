"""Unit tests for the seeded synthetic data generator (offline, no DB)."""

from __future__ import annotations

import json

import pytest

from app.seed.generator import (
    N_CUSTOMERS,
    N_INCIDENTS,
    N_POSTMORTEMS,
    N_RUNBOOKS,
    N_TICKETS,
    generate_dataset,
    write_corpus,
)


@pytest.mark.unit
def test_dataset_is_deterministic() -> None:
    a = generate_dataset(1337)
    b = generate_dataset(1337)
    assert a == b


@pytest.mark.unit
def test_dataset_changes_with_seed() -> None:
    assert generate_dataset(1) != generate_dataset(2)


@pytest.mark.unit
def test_dataset_shapes() -> None:
    ds = generate_dataset(1337)
    assert len(ds.customers) == N_CUSTOMERS
    assert len(ds.incidents) == N_INCIDENTS
    assert len(ds.tickets) == N_TICKETS
    assert len(ds.postmortems) == N_POSTMORTEMS
    assert len(ds.runbooks) == N_RUNBOOKS
    assert len(ds.metrics) == 365 * 6 * 4  # days * services * metric_names
    assert {s.tier for s in ds.slas} == {"enterprise", "business", "standard"}


@pytest.mark.unit
def test_incident_invariants() -> None:
    ds = generate_dataset(1337)
    for inc in ds.incidents:
        assert inc.severity in {"P1", "P2", "P3", "P4"}
        if inc.resolved_at is not None:
            assert inc.resolved_at > inc.opened_at
    # Postmortems only cover resolved P1/P2 incidents.
    pm_ids = {d.doc_id for d in ds.postmortems}
    by_id = {i.id: i for i in ds.incidents}
    for pm_id in pm_ids:
        assert by_id[pm_id].severity in {"P1", "P2"}
        assert by_id[pm_id].resolved_at is not None


@pytest.mark.unit
def test_write_corpus(tmp_path: object) -> None:
    from pathlib import Path

    out = Path(str(tmp_path)) / "seed"
    ds = generate_dataset(1337)
    files = write_corpus(ds, out)
    assert files == 1 + N_POSTMORTEMS + N_RUNBOOKS

    lines = (out / "tickets.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == N_TICKETS
    assert json.loads(lines[0])["id"].startswith("TCK-")
    assert len(list((out / "postmortems").glob("*.md"))) == N_POSTMORTEMS
    assert len(list((out / "runbooks").glob("*.md"))) == N_RUNBOOKS

    # Idempotent: rewriting yields identical content.
    write_corpus(ds, out)
    assert len((out / "tickets.jsonl").read_text(encoding="utf-8").splitlines()) == N_TICKETS
