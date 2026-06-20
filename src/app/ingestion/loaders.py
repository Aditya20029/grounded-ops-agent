"""Corpus loaders: turn seed files into a uniform list of source documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.seed.generator import DEFAULT_SEED_DIR


@dataclass(frozen=True)
class SourceDoc:
    """A unit of unstructured content to be chunked and indexed."""

    doc_id: str
    source_type: str  # "ticket" | "postmortem" | "runbook"
    title: str
    text: str
    markdown: bool


def _markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_tickets(path: Path) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    if not path.exists():
        return docs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        docs.append(
            SourceDoc(
                doc_id=rec["id"],
                source_type="ticket",
                title=rec["title"],
                text=f"{rec['title']}\n\n{rec['body']}",
                markdown=False,
            )
        )
    return docs


def load_markdown_dir(directory: Path, source_type: str) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    if not directory.exists():
        return docs
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            SourceDoc(
                doc_id=path.stem,
                source_type=source_type,
                title=_markdown_title(text, path.stem),
                text=text,
                markdown=True,
            )
        )
    return docs


def load_corpus(seed_dir: Path = DEFAULT_SEED_DIR) -> list[SourceDoc]:
    """Load tickets, postmortems, and runbooks from the seed directory."""
    return [
        *load_tickets(seed_dir / "tickets.jsonl"),
        *load_markdown_dir(seed_dir / "postmortems", "postmortem"),
        *load_markdown_dir(seed_dir / "runbooks", "runbook"),
    ]
