"""Markdown-aware recursive chunker.

Splitting rules (in order):

1. Markdown headers (``^#+\\s+``) split the doc into hierarchical
   sections; the breadcrumb header path is tracked.
2. Each section is split into paragraphs, then sentences, then
   whitespace words as a last resort.
3. Code blocks (````` ... ```````) stay atomic — the chunker never
   splits inside one. A code block that exceeds ``chunk_size`` is
   emitted as its own chunk and a warning is logged.
4. After every accumulation the running token count is checked.
   When it crosses ``chunk_size`` a chunk is emitted; the next
   chunk starts with up to ``chunk_overlap`` tokens from the
   previous chunk's tail.

The structural primitives now live in
:mod:`vector_service.chunking.structure` and token helpers in
:mod:`vector_service.chunking.tokens`; this module keeps the
recursive packing core and re-exports the historical symbols
(``Chunk``, ``count_tokens``, ``_split_into_sections``,
``_split_around_code_blocks`` …) for backwards compatibility.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from vector_service.chunking.base import BaseChunker, Chunk, register_strategy
from vector_service.chunking.structure import (
    CODE_FENCE_RX as _CODE_FENCE_RX,
    HEADER_RX as _HEADER_RX,
    Section as _Section,
    header_prefix,
    join_header,
    merge_small_chunks,
    pack_units,
    prepare_sections,
    split_around_code_blocks as _split_around_code_blocks,
    split_into_sections as _split_into_sections,
    units_for_section,
)
from vector_service.chunking.tokens import (
    _get_encoding,
    count_tokens,
)


@register_strategy("recursive")
class RecursiveChunker(BaseChunker):
    """Recursive markdown-aware chunker.

    Args:
        chunk_size: Target upper bound on tokens per emitted chunk.
            Emission triggers once the threshold is crossed at a
            paragraph boundary; the chunker never splits inside a
            code block.
        chunk_overlap: Trailing tokens from the previous chunk
            prefixed onto the next. Must be ``< chunk_size``.
        token_counter: Optional override for the token-counting
            callable (tests supply a deterministic char counter).
    """

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Split ``markdown" into leaf chunks. Empty input → ``[]``.

        Chunks never cross section boundaries: every leaf belongs to
        exactly one section (tagged with its ``section_ord``) so the
        hierarchy can parent it deterministically.
        """
        if not markdown or not markdown.strip():
            return []
        sections = prepare_sections(markdown, page_numbers)
        chunks = _chunk_sections(
            markdown,
            sections,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            token_counter=self.token_counter,
        )
        return merge_small_chunks(
            chunks,
            chunk_size=self.chunk_size,
            token_counter=self.token_counter,
            min_size=self.min_chunk_size,
        )


# ---- internals ---------------------------------------------------------


def _chunk_sections(
    markdown: str,
    sections: list[_Section],
    *,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Build leaf chunks independently per section.

    For each non-empty section: build code-atomic / paragraph units
    (each unit does NOT become its own chunk — that used to turn
    line-per-paragraph Docling output into hundreds of one-line
    fragments), translate their offsets to document positions,
    reconstruct the breadcrumb header path onto the first unit, then
    greedily pack.
    """
    chunks: list[Chunk] = []
    find_cursor = 0

    for section in sections:
        body = section.body
        if not body:
            continue

        # Locate the stripped body in the document (sections are
        # visited in source order) so unit spans become document offsets.
        base = markdown.find(body, find_cursor)
        find_cursor = base + 1

        units = units_for_section(
            body, chunk_size=chunk_size, token_counter=token_counter,
        )
        for unit in units:
            if unit.char_start is not None:
                unit.char_start = base + unit.char_start
                unit.char_end = base + unit.char_end

        # Reconstruct the full breadcrumb header path on the first
        # unit so a leaf reads in isolation; those ancestor headers
        # are not at this source position and stay out of the span.
        if section.header_path:
            units[0].text = header_prefix(section.header_path) + units[0].text

        section_chunks = pack_units(
            units,
            chunk_size=chunk_size,
            token_counter=token_counter,
            section_header=join_header(section.header_path),
            page_number=section.page_number,
            separator="\n\n",
            overlap=chunk_overlap,
            section_ord=section.ord,
        )

        # The section's OWN header is physically here and belongs in
        # the first leaf's span (ancestor headers do not).
        if section.header_path and section_chunks:
            first = section_chunks[0]
            starts = [
                s for s in (first.char_start, section.char_start)
                if s is not None
            ]
            first.char_start = min(starts)
        chunks.extend(section_chunks)

    return chunks


# Re-export the page annotation entry point so legacy imports of
# private helpers keep working.
def _annotate_pages(
    sections: list[_Section],
    pages: list[int],
    full_text: str,
) -> None:
    from vector_service.chunking.structure import annotate_pages
    annotate_pages(sections, pages, full_text)
