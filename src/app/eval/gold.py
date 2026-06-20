"""Gold evaluation set: load and (re)generate.

Each item is ``{question, relevant_doc_ids, reference_answer}``. The set is
bootstrapped deterministically from the seeded dataset (templated questions over
known documents) so the eval is reproducible without an LLM or a human pass. In
a production setting you would LLM-generate candidate questions and human-review
them; here templated questions over seeded data keep the harness reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GOLD_PATH = Path(__file__).parent / "gold.jsonl"


@dataclass(frozen=True)
class GoldItem:
    question: str
    relevant_doc_ids: list[str]
    reference_answer: str


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        items.append(
            GoldItem(
                question=record["question"],
                relevant_doc_ids=record["relevant_doc_ids"],
                reference_answer=record["reference_answer"],
            )
        )
    return items
