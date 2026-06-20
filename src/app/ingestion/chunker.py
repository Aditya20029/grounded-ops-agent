"""Token-aware chunking with overlap and header-aware Markdown splitting.

A generic tokenizer (tiktoken ``cl100k_base``) is used purely for *sizing*
chunks; the stored text and char spans are sliced from the original document so
that ``content == document_text[char_start:char_end]`` always holds. Markdown is
split on headers first so postmortem sections (Summary, Root cause, ...) stay
coherent within a chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken

TARGET_TOKENS = 500
OVERLAP_TOKENS = 80


@dataclass(frozen=True)
class ChunkPiece:
    """One chunk's position and text within a single source document."""

    chunk_index: int
    char_start: int
    char_end: int
    content: str


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _prefix_chars(tokens: list[int], k: int) -> int:
    """Character length of the text formed by the first ``k`` tokens."""
    if k <= 0:
        return 0
    return len(_encoder().decode(tokens[:k]))


def _token_chunks(
    text: str,
    *,
    offset: int,
    start_index: int,
    target: int,
    overlap: int,
) -> tuple[list[ChunkPiece], int]:
    """Chunk ``text`` by tokens; char spans are offset into the parent document."""
    enc = _encoder()
    tokens = enc.encode(text)
    if not tokens:
        return [], start_index

    step = max(1, target - overlap)
    pieces: list[ChunkPiece] = []
    index = start_index
    i = 0
    while i < len(tokens):
        end = min(i + target, len(tokens))
        local_start = _prefix_chars(tokens, i)
        local_end = _prefix_chars(tokens, end)
        content = text[local_start:local_end]
        if content.strip():
            pieces.append(
                ChunkPiece(
                    chunk_index=index,
                    char_start=offset + local_start,
                    char_end=offset + local_end,
                    content=content,
                )
            )
            index += 1
        if end == len(tokens):
            break
        i += step
    return pieces, index


def split_markdown_sections(text: str) -> list[tuple[str, int]]:
    """Split Markdown into (section_text, char_offset) pairs at header lines.

    A new section starts at each line beginning with ``#``. Each section keeps
    its header plus the body up to the next header.
    """
    sections: list[tuple[str, int]] = []
    cur: list[str] = []
    section_offset = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#") and cur:
            sections.append(("".join(cur), section_offset))
            cur = []
            section_offset = offset
        cur.append(line)
        offset += len(line)
    if cur:
        sections.append(("".join(cur), section_offset))
    return sections


def chunk_document(
    text: str,
    *,
    markdown: bool,
    target: int = TARGET_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[ChunkPiece]:
    """Chunk a document; ``markdown`` enables header-aware splitting first."""
    pieces: list[ChunkPiece] = []
    if not markdown:
        pieces, _ = _token_chunks(text, offset=0, start_index=0, target=target, overlap=overlap)
        return pieces

    index = 0
    for section_text, section_offset in split_markdown_sections(text):
        sub, index = _token_chunks(
            section_text,
            offset=section_offset,
            start_index=index,
            target=target,
            overlap=overlap,
        )
        pieces.extend(sub)
    return pieces
