"""Hierarchy assembly for small-to-large retrieval.

Chunkers emit leaf chunks tagged with a ``section_ord``; this module
wraps them with the two parent levels and assigns the document-local
link keys that persist time resolves to real ``chunk_id`` values:

- ``"document"`` — one root row, text is the full parsed markdown;
- ``"section:{ord}"`` — one intermediate row per non-empty header
  section (header-less structural blocks included);
- ``"chunk:{i}"`` — leaf rows in source order.

Output order is root, sections in ordinal order, leaves in source
order, with dense ``chunk_index`` values across the flattened list.
Only leaves later enter the derived dense/sparse ANN index; parents
store content only.
"""
from __future__ import annotations

from collections.abc import Callable

from vector_service.chunking.base import (
    LEVEL_CHUNK,
    LEVEL_DOCUMENT,
    LEVEL_SECTION,
    Chunk,
)
from vector_service.chunking.structure import header_prefix, join_header

DOCUMENT_KEY = "document"
SECTION_KEY_PREFIX = "section:"
CHUNK_KEY_PREFIX = "chunk:"


def section_key(ord: int) -> str:
    return f"{SECTION_KEY_PREFIX}{ord}"


def chunk_key(index: int) -> str:
    return f"{CHUNK_KEY_PREFIX}{index}"


def build_hierarchy(
    markdown: str,
    sections: list,
    leaves: list[Chunk],
    *,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Return root + section parents + tagged leaves in one flat list."""
    text = markdown.strip()

    document = Chunk(
        text=text,
        chunk_index=0,
        token_count=token_counter(text),
        section_header="",
        page_number=None,
        key=DOCUMENT_KEY,
        parent_key=None,
        level=LEVEL_DOCUMENT,
        char_start=0,
        char_end=len(markdown),
    )

    section_rows: list[Chunk] = []
    for section in sections:
        body = section.body
        if not body:
            continue
        prefix = header_prefix(section.header_path)
        rendered = (prefix + body).strip()
        section_rows.append(
            Chunk(
                text=rendered,
                chunk_index=0,  # assigned after flattening
                token_count=token_counter(rendered),
                section_header=join_header(section.header_path),
                page_number=section.page_number,
                key=section_key(section.ord),
                parent_key=DOCUMENT_KEY,
                level=LEVEL_SECTION,
                char_start=section.char_start,
                char_end=section.char_end,
            )
        )

    tagged_leaves: list[Chunk] = []
    for i, leaf in enumerate(leaves):
        parent = (
            section_key(leaf.section_ord)
            if leaf.section_ord is not None
            else DOCUMENT_KEY
        )
        tagged_leaves.append(
            Chunk(
                text=leaf.text,
                chunk_index=0,
                token_count=leaf.token_count,
                section_header=leaf.section_header,
                page_number=leaf.page_number,
                context=leaf.context,
                summary=leaf.summary,
                key=chunk_key(i),
                parent_key=parent,
                level=LEVEL_CHUNK,
                section_ord=leaf.section_ord,
                char_start=leaf.char_start,
                char_end=leaf.char_end,
            )
        )

    rows = [document, *section_rows, *tagged_leaves]
    for i, row in enumerate(rows):
        row.chunk_index = i
    return rows
