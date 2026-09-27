"""Paragraph-first chunker.

Structure-based strategy (the family that the 2025 chunking
taxonomy studies find most cost-effective for in-corpus
retrieval):

- markdown headers still split the document into sections, so
  breadcrumb / page metadata is preserved;
- within each section, paragraphs are the atomic units and are
  greedily packed up to ``chunk_size`` — short paragraphs merge,
  unlike the recursive chunker which also splits small structures;
- a paragraph that alone exceeds ``chunk_size`` degrades to
  sentences, then words;
- fenced code blocks always stay atomic.

The section's literal header line is prepended to the first unit
so the title participates in the embedding, not just the stored
breadcrumb.
"""
from __future__ import annotations

from collections.abc import Iterable

from vector_service.chunking.base import BaseChunker, Chunk, register_strategy
from vector_service.chunking.structure import (
    header_prefix,
    join_header,
    merge_small_chunks,
    prepare_sections,
    pack_units,
    units_for_section,
)


@register_strategy("paragraph")
class ParagraphChunker(BaseChunker):
    """Paragraph-based greedy packer."""

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Chunk ``markdown`` by paragraph. Empty input → ``[]``."""
        if not markdown or not markdown.strip():
            return []

        sections = prepare_sections(markdown, page_numbers)
        out: list[Chunk] = []
        for section in sections:
            if not section.body:
                continue
            units = units_for_section(
                section.body,
                chunk_size=self.chunk_size,
                token_counter=self.token_counter,
            )
            # Seed the first unit with the literal markdown headers.
            if units and section.header_path:
                units[0].text = header_prefix(section.header_path) + units[0].text

            out.extend(
                pack_units(
                    units,
                    chunk_size=self.chunk_size,
                    token_counter=self.token_counter,
                    section_header=join_header(section.header_path),
                    page_number=section.page_number,
                    separator="\n\n",
                    overlap=self.chunk_overlap,
                )
            )
        return merge_small_chunks(
            out,
            chunk_size=self.chunk_size,
            token_counter=self.token_counter,
            min_size=self.min_chunk_size,
        )
