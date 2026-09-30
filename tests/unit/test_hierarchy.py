"""Hierarchy assembly tests for small-to-large retrieval.

Pins ``build_hierarchy``: the flat row order (document, sections,
leaves), document-local link keys and their parent resolution, dense
``chunk_index`` assignment and the character spans carried from the
chunker onto the stored rows.
"""
from __future__ import annotations

from vector_service.chunking.base import (
    LEVEL_CHUNK,
    LEVEL_DOCUMENT,
    LEVEL_SECTION,
    Chunk,
)
from vector_service.chunking.hierarchy import (
    CHUNK_KEY_PREFIX,
    DOCUMENT_KEY,
    SECTION_KEY_PREFIX,
    build_hierarchy,
)
from vector_service.chunking.recursive_chunker import RecursiveChunker
from vector_service.chunking.structure import prepare_sections

MD = "# Title\n\nIntro.\n\n## A\n\nAlpha paragraph.\n\n## B\n\nBeta paragraph."


def _build(md: str = MD, *, chunk_size: int = 10_000):
    sections = prepare_sections(md)
    leaves = RecursiveChunker(
        chunk_size=chunk_size, chunk_overlap=0, token_counter=len
    ).chunk(md)
    return sections, leaves, build_hierarchy(
        md, sections, leaves, token_counter=len
    )


def test_row_order_levels_and_dense_chunk_index():
    _, _, rows = _build()
    assert [r.level for r in rows] == [
        LEVEL_DOCUMENT,
        LEVEL_SECTION,
        LEVEL_SECTION,
        LEVEL_SECTION,
        LEVEL_CHUNK,
        LEVEL_CHUNK,
        LEVEL_CHUNK,
    ]
    assert [r.chunk_index for r in rows] == list(range(len(rows)))


def test_document_root_fields():
    _, _, rows = _build()
    root = rows[0]
    assert root.key == DOCUMENT_KEY
    assert root.parent_key is None
    assert root.text == MD.strip()
    assert root.char_start == 0
    assert root.char_end == len(MD)


def test_section_rows_link_to_document_and_keep_spans():
    sections, _, rows = _build()
    section_rows = [r for r in rows if r.level == LEVEL_SECTION]
    assert [r.key for r in section_rows] == [
        f"{SECTION_KEY_PREFIX}{s.ord}" for s in sections
    ]
    assert {r.parent_key for r in section_rows} == {DOCUMENT_KEY}
    for row, section in zip(section_rows, sections):
        assert row.char_start == section.char_start
        assert row.char_end == section.char_end
        # Span starts exactly at the section header.
        assert MD[row.char_start : row.char_end].lstrip().startswith("#")


def test_leaves_link_to_their_section_and_keep_spans():
    sections, leaves, rows = _build()
    leaf_rows = [r for r in rows if r.level == LEVEL_CHUNK]
    assert [r.key for r in leaf_rows] == [
        f"{CHUNK_KEY_PREFIX}{i}" for i in range(len(leaves))
    ]
    ord_to_key = {s.ord: f"{SECTION_KEY_PREFIX}{s.ord}" for s in sections}
    for row, leaf, section in zip(leaf_rows, leaves, sections):
        assert row.parent_key == ord_to_key[leaf.section_ord]
        assert row.char_start == leaf.char_start
        assert row.char_end == leaf.char_end
        # The raw span starts at the literal section header and covers
        # its body; the leaf text additionally reconstructs ancestor
        # breadcrumb headers that are not physically at this position.
        literal_header = "#" * len(section.header_path)
        expected_span = (
            f"{literal_header} {section.header_path[-1]}\n\n{section.body}"
        )
        assert MD[row.char_start : row.char_end].strip() == expected_span


def test_leaf_without_section_parents_document():
    leaves = [
        Chunk(
            text="orphan text",
            chunk_index=0,
            token_count=11,
            section_header="",
            section_ord=None,
            char_start=0,
            char_end=11,
        )
    ]
    rows = build_hierarchy("orphan text", [], leaves, token_counter=len)
    assert len(rows) == 2
    assert rows[0].level == LEVEL_DOCUMENT
    leaf = rows[1]
    assert leaf.level == LEVEL_CHUNK
    assert leaf.parent_key == DOCUMENT_KEY


def test_empty_body_sections_get_no_parent_rows():
    class _Empty:
        body = ""
        ord = 0

    leaves = [
        Chunk(
            text="x",
            chunk_index=0,
            token_count=1,
            section_header="",
            section_ord=None,
        )
    ]
    rows = build_hierarchy("x", [_Empty()], leaves, token_counter=len)
    assert [r.level for r in rows] == [LEVEL_DOCUMENT, LEVEL_CHUNK]
