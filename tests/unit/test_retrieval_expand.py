"""Tests for small-to-large context expansion.

Pins the §2 contract the §4 retrieval rewrite will consume:

- leaves expand to their ancestor at the requested level;
- sibling leaves sharing an ancestor collapse to one output;
- distinct parents stay distinct;
- expansion happens on the already-truncated list (caller controls
  the point in the pipeline);
- an unresolved leaf / ``context_level="chunk"`` passes through.

Also an end-to-end case against a real :class:`CorpusRepository`:
rows written with hierarchy fields, expanded through
``get_chunk_ancestor_rows``.
"""
from __future__ import annotations

from vector_service.corpus import ChunkRecord, CorpusRepository
from vector_service.retrieval.base import RetrievedChunk
from vector_service.retrieval.expand import (
    CONTEXT_LEVELS,
    expand_chunks,
    expand_to_level,
)


def _leaf(chunk_id: str, *, text: str = "leaf text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        fields={"text": text},
        fusion_score=0.5,
        matched_channels=["dense"],
    )


def _ancestor(ancestor_id: str, level: str, text: str) -> dict:
    return {
        "chunk_id": ancestor_id,
        "text": text,
        "level": level,
        "char_start": 0,
        "char_end": 1,
    }


class _FakeRepo:
    def __init__(self, rows):
        self._rows = rows
        self.requested_level = None

    def get_chunk_ancestor_rows(self, chunk_ids, level):
        self.requested_level = level
        return {cid: self._rows[cid] for cid in chunk_ids if cid in self._rows}


def test_context_levels_constant():
    assert CONTEXT_LEVELS == ("chunk", "section", "document")


def test_expand_to_section_collapses_sibling_leaves():
    repo = _FakeRepo({
        "c1": _ancestor("s1", "section", "section text"),
        "c2": _ancestor("s1", "section", "section text"),
    })
    out = expand_to_level(
        [_leaf("c1"), _leaf("c2")], repo=repo, level="section"
    )
    assert repo.requested_level == "section"
    assert len(out) == 1
    anchor = out[0]
    assert anchor.chunk_id == "s1"
    assert anchor.fields["text"] == "section text"
    assert anchor.fields["level"] == "section"
    # Scores/provenance are carried from the first contributing leaf.
    assert anchor.fusion_score == 0.5
    assert anchor.matched_channels == ["dense"]


def test_distinct_section_parents_stay_distinct():
    repo = _FakeRepo({
        "c1": _ancestor("s1", "section", "section one"),
        "c2": _ancestor("s2", "section", "section two"),
    })
    out = expand_to_level(
        [_leaf("c1"), _leaf("c2")], repo=repo, level="section"
    )
    assert [c.chunk_id for c in out] == ["s1", "s2"]
    assert [c.fields["text"] for c in out] == [
        "section one", "section two"
    ]


def test_expand_to_document_collapses_all_leaves_of_one_doc():
    repo = _FakeRepo({
        "c1": _ancestor("d1", "document", "full document"),
        "c2": _ancestor("d1", "document", "full document"),
        "c3": _ancestor("d1", "document", "full document"),
    })
    out = expand_to_level(
        [_leaf("c1"), _leaf("c2"), _leaf("c3")],
        repo=repo, level="document",
    )
    assert len(out) == 1
    assert out[0].chunk_id == "d1"
    assert out[0].fields["text"] == "full document"


def test_expand_runs_after_top_k_truncation():
    # The caller truncates first; even though c3 has a distinct
    # parent, it never reaches expansion because top_k=2.
    repo = _FakeRepo({
        "c1": _ancestor("s1", "section", "one"),
        "c2": _ancestor("s1", "section", "one"),
        "c3": _ancestor("s2", "section", "two"),
    })
    truncated = [_leaf("c1"), _leaf("c2"), _leaf("c3")][:2]
    out = expand_to_level(truncated, repo=repo, level="section")
    assert [c.chunk_id for c in out] == ["s1"]


def test_unresolved_leaf_passes_through():
    out = expand_chunks([_leaf("c1")], ancestor_rows={})
    assert len(out) == 1
    assert out[0].chunk_id == "c1"


def test_missing_leaf_beside_resolved_sibling_passes_through():
    out = expand_chunks(
        [_leaf("c1"), _leaf("c2")],
        ancestor_rows={"c2": _ancestor("s2", "section", "two")},
    )
    assert [c.chunk_id for c in out] == ["c1", "s2"]


def test_chunk_level_is_noop():
    repo = _FakeRepo({})
    chunks = [_leaf("c1"), _leaf("c2")]
    assert expand_to_level(chunks, repo=repo, level="chunk") is chunks
    assert expand_to_level([], repo=repo, level="section") == []


def test_unknown_level_raises():
    import pytest

    with pytest.raises(ValueError):
        expand_to_level([_leaf("c1")], repo=_FakeRepo({}), level="chapter")


# ---- end-to-end against the real corpus repository --------------------


def test_expand_against_real_repository(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    now = 1.0

    def _record(chunk_id, *, chunk_index, text, level, parent_id=None):
        return ChunkRecord(
            chunk_id=chunk_id,
            doc_id="doc1",
            database="default",
            collection="ingest",
            chunk_index=chunk_index,
            text=text,
            level=level,
            parent_id=parent_id,
            char_start=0,
            char_end=len(text),
            created_ts=now,
        )

    document = _record(
        "doc1_0", chunk_index=0, text="# Title\n\nall of it",
        level="document",
    )
    section = _record(
        "doc1_1", chunk_index=1, text="# Title\n\nbody",
        level="section", parent_id="doc1_0",
    )
    leaf = _record(
        "doc1_2", chunk_index=2, text="body",
        level="chunk", parent_id="doc1_1",
    )

    # store_document requires a matching documents row; insert the
    # document record directly alongside a document row.
    from vector_service.corpus import DocumentRecord

    repo.store_document(
        DocumentRecord(
            doc_id="doc1",
            database="default",
            collection="ingest",
            filename=None,
            mime="text/plain",
            content_hash="hash",
            created_ts=now,
        ),
        [document, section, leaf],
    )

    out = expand_to_level(
        [_leaf("doc1_2")], repo=repo, level="section"
    )
    assert len(out) == 1
    assert out[0].chunk_id == "doc1_1"
    assert out[0].fields["level"] == "section"

    out = expand_to_level(
        [_leaf("doc1_2")], repo=repo, level="document"
    )
    assert [c.chunk_id for c in out] == ["doc1_0"]
