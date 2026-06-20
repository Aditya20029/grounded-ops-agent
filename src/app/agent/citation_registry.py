"""Per-request citation registry with stable indices.

Every chunk surfaced across all retrieval steps is added here and assigned a
stable 1-based index exactly once. Indices never shift or collide between steps:
re-adding a chunk already seen returns its existing index. The registry both
renders the labelled, delimited records for the prompt and backs the ``sources``
array and citation post-validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.types import RetrievedChunk

# Fence delimiting untrusted record content in the prompt (content never holds it).
RECORD_OPEN = "<<<RECORD"
RECORD_CLOSE = "RECORD>>>"
_SNIPPET_CHARS = 240


def _snippet(content: str) -> str:
    flat = " ".join(content.split())
    return flat[:_SNIPPET_CHARS]


@dataclass(frozen=True)
class CitationEntry:
    index: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    snippet: str
    char_start: int
    char_end: int
    content: str


class CitationRegistry:
    """Assigns and remembers stable citation indices for chunks."""

    def __init__(self) -> None:
        self._by_chunk: dict[str, CitationEntry] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def add(self, chunks: list[RetrievedChunk]) -> list[int]:
        """Add chunks; return their (stable) indices in input order."""
        indices: list[int] = []
        for chunk in chunks:
            entry = self._by_chunk.get(chunk.chunk_id)
            if entry is None:
                entry = CitationEntry(
                    index=len(self._order) + 1,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    snippet=_snippet(chunk.content),
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    content=chunk.content,
                )
                self._by_chunk[chunk.chunk_id] = entry
                self._order.append(chunk.chunk_id)
            indices.append(entry.index)
        return indices

    def entries(self) -> list[CitationEntry]:
        return [self._by_chunk[cid] for cid in self._order]

    def get(self, index: int) -> CitationEntry | None:
        if 1 <= index <= len(self._order):
            return self._by_chunk[self._order[index - 1]]
        return None

    def has_index(self, index: int) -> bool:
        return 1 <= index <= len(self._order)

    def render_for_prompt(self) -> str:
        """Render entries as labelled, fenced records for the answering prompt."""
        blocks = []
        for e in self.entries():
            blocks.append(
                f"[{e.index}] source_type={e.source_type} doc_id={e.doc_id} "
                f'title="{e.title}"\n'
                f"{RECORD_OPEN}\n{e.content}\n{RECORD_CLOSE}"
            )
        return "\n\n".join(blocks)
