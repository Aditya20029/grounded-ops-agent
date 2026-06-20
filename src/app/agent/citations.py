"""Citation post-validation: strip hallucinated indices, report used ones."""

from __future__ import annotations

import re

from app.agent.citation_registry import CitationRegistry

_CITE = re.compile(r"\[(\d+)\]")


def validate_and_strip(answer: str, registry: CitationRegistry) -> tuple[str, list[int]]:
    """Remove any cited index not present in the registry; return cleaned text
    and the sorted list of valid indices actually used."""
    used: set[int] = set()

    def _replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if registry.has_index(index):
            used.add(index)
            return match.group(0)
        return ""  # hallucinated citation -> stripped

    cleaned = _CITE.sub(_replace, answer)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    return cleaned, sorted(used)
