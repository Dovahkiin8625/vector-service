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
    tail_tokens as _tail_tokens,
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
        """Split ``markdown`` into chunks. Empty input → ``[]``."""
        if not markdown or not markdown.strip():
            return []
        sections = prepare_sections(markdown, page_numbers)
        chunks = _chunk_sections(
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
    sections: list[_Section],
    *,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Walk the section list and emit chunks.

    State carried across section boundaries:

    - ``current_text`` / ``current_tokens`` — running chunk buffer;
    - ``overlap_text`` — trailing tokens from the latest emission.

    Sections whose body alone exceeds ``chunk_size`` are recursively
    split into paragraphs, sentences, words.
    """
    chunks: list[Chunk] = []
    overlap_text: str = ""
    current_text: str = ""
    current_tokens: int = 0
    current_section: list[str] = []
    current_page: int | None = None

    def _flush() -> None:
        """Emit the current buffer as a chunk."""
        nonlocal current_text, current_tokens, overlap_text
        body = current_text.strip()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    section_header=join_header(current_section),
                    page_number=current_page,
                )
            )
            overlap_text = _tail_tokens(body, chunk_overlap)
        else:
            overlap_text = ""
        current_text = ""
        current_tokens = 0

    for section in sections:
        header_path = section.header_path
        page = section.page_number
        body = section.body
        if not body:
            continue

        # Section header is prepended only when starting a new chunk
        # so it isn't double-counted across overlapping chunks.
        header_prefix = ""
        if header_path:
            header_prefix = "\n".join(
                f"{'#' * (i + 1)} {h}" for i, h in enumerate(header_path)
            )
            header_prefix += "\n\n"

        # A section bigger than chunk_size cannot share a chunk.
        # Build code-atomic / paragraph units and greedily pack them
        # (each unit does NOT become its own chunk — that used to
        # turn line-per-paragraph Docling output into hundreds of
        # one-line fragments).
        if token_counter(body) > chunk_size:
            _flush()
            units = units_for_section(
                body, chunk_size=chunk_size, token_counter=token_counter,
            )
            if units and header_path:
                units[0].text = header_prefix(header_path) + units[0].text
            chunks.extend(
                pack_units(
                    units,
                    chunk_size=chunk_size,
                    token_counter=token_counter,
                    section_header=join_header(header_path),
                    page_number=page,
                    separator="\n\n",
                    overlap=chunk_overlap,
                )
            )
            if chunks:
                overlap_text = _tail_tokens(chunks[-1].text, chunk_overlap)
            current_section = list(header_path)
            current_page = page
            continue

        # Section fits one chunk — provided it fits with whatever is
        # already buffered.
        candidate_text = (
            (current_text + "\n\n" + header_prefix + body) if current_text
            else (header_prefix + body)
        )
        candidate_tokens = token_counter(candidate_text)

        if candidate_tokens > chunk_size and current_text:
            _flush()
            current_text = overlap_text
            current_tokens = token_counter(current_text) if overlap_text else 0
            current_section = list(header_path)
            current_page = page
            candidate_text = (
                (current_text + "\n\n" + header_prefix + body) if current_text
                else (header_prefix + body)
            )
            candidate_tokens = token_counter(candidate_text)
            # overlap + section still too big → drop the overlap so
            # the section still gets a chunk of its own.
            if candidate_tokens > chunk_size and current_text:
                current_text = ""
                current_tokens = 0
                candidate_text = header_prefix + body
                candidate_tokens = token_counter(candidate_text)

        current_text = candidate_text
        current_tokens = candidate_tokens
        current_section = list(header_path)
        # The latest page seen wins for the in-progress chunk.
        if page is not None:
            current_page = page

    if current_text.strip():
        _flush()

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
