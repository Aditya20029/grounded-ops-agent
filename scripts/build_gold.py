"""Regenerate the gold eval set from the seeded dataset (deterministic).

Usage: python scripts/build_gold.py [--seed N] [--n 30]

Templated questions are written over known seeded documents (postmortems,
runbooks, tickets) so relevance is unambiguous and reproducible.
"""

from __future__ import annotations

import argparse
import json
import random

from app.core.settings import get_settings
from app.eval.gold import DEFAULT_GOLD_PATH
from app.seed.generator import generate_dataset


def build_gold(seed: int, n: int) -> list[dict[str, object]]:
    ds = generate_dataset(seed)
    rng = random.Random(seed + 1)
    items: list[dict[str, object]] = []

    # ~half from postmortems (root cause / remediation questions)
    postmortems = list(ds.postmortems)
    rng.shuffle(postmortems)
    incidents_by_id = {i.id: i for i in ds.incidents}
    for doc in postmortems[: n // 2]:
        inc = incidents_by_id[doc.doc_id]
        items.append(
            {
                "question": (
                    f"Why did incident {inc.id} on the {inc.service} service happen, "
                    "and how was it remediated?"
                ),
                "relevant_doc_ids": [inc.id],
                "reference_answer": (
                    f"The {inc.severity} incident on {inc.service} was caused by a "
                    f"{inc.root_cause_category} issue and remediated per the postmortem."
                ),
            }
        )

    # ~a third from runbooks (procedure questions)
    runbooks = list(ds.runbooks)
    rng.shuffle(runbooks)
    for doc in runbooks[: n // 3]:
        topic = doc.title.replace("Runbook: ", "")
        items.append(
            {
                "question": f"What is the operational procedure described in: {topic}?",
                "relevant_doc_ids": [doc.doc_id],
                "reference_answer": f"Follow the runbook steps for: {topic}.",
            }
        )

    # remainder from tickets (issue-description questions)
    tickets = list(ds.tickets)
    rng.shuffle(tickets)
    for ticket in tickets[: n - len(items)]:
        items.append(
            {
                "question": f"What issue did support ticket {ticket.id} report?",
                "relevant_doc_ids": [ticket.id],
                "reference_answer": ticket.title,
            }
        )
    return items[:n]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Regenerate the gold eval set.")
    parser.add_argument("--seed", type=int, default=settings.data_seed)
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    items = build_gold(args.seed, args.n)
    with DEFAULT_GOLD_PATH.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    print(f"Wrote {len(items)} gold items to {DEFAULT_GOLD_PATH}")


if __name__ == "__main__":
    main()
