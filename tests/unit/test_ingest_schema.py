"""Schema validation tests for the ingest API Pydantic models.

These tests cover the request/response shapes that the three ingest
endpoints (``/v1/parse``, ``/v1/chunk``, ``/v1/ingest``) expose.
Routes should not need unit testing of the Pydantic models — the
HTTP layer validates via FastAPI's dependency injection — but the
shapes themselves are part of the public API contract and worth
exercising directly.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.schemas.ingest import (
    ChunkItem,
    ChunkResponse,
    IngestResponse,
    ParseMetadata,
    ParseResponse,
    metadata_to_field_dict,
)


# ---- /v1/parse schemas -------------------------------------------------


def test_parse_metadata_accepts_full_payload():
    m = ParseMetadata(
        page_count=12,
        title="Annual Report",
        author="Acme Inc.",
        mime_type="application/pdf",
    )
    assert m.page_count == 12
    assert m.title == "Annual Report"
    assert m.author == "Acme Inc."
    assert m.mime_type == "application/pdf"


def test_parse_metadata_accepts_partial_payload():
    """Title / author / page_count are all optional — partial
    metadata must still validate."""
    m = ParseMetadata(mime_type="text/plain")
    assert m.page_count is None
    assert m.title is None
    assert m.author is None
    assert m.mime_type == "text/plain"


def test_parse_response_includes_metadata_block():
    md = "# Title\n\nBody text."
    meta = ParseMetadata(page_count=1, mime_type="text/markdown")
    resp = ParseResponse(markdown=md, metadata=meta)
    assert resp.markdown == md
    assert resp.metadata.page_count == 1


def test_parse_response_requires_markdown():
    with pytest.raises(ValidationError):
        ParseResponse(metadata=ParseMetadata())  # type: ignore[call-arg]


# ---- /v1/chunk schemas -------------------------------------------------


def test_chunk_item_carries_all_fields():
    item = ChunkItem(
        text="hello",
        chunk_index=3,
        token_count=487,
        section_header="A > B",
        page_number=2,
    )
    assert item.text == "hello"
    assert item.chunk_index == 3
    assert item.token_count == 487
    assert item.section_header == "A > B"
    assert item.page_number == 2


def test_chunk_item_defaults():
    """``section_header`` and ``page_number`` default to empty /
    ``None`` so callers can omit them."""
    item = ChunkItem(text="hi", chunk_index=0, token_count=1)
    assert item.section_header == ""
    assert item.page_number is None


def test_chunk_response_envelope():
    items = [
        ChunkItem(text="a", chunk_index=0, token_count=1),
        ChunkItem(text="b", chunk_index=1, token_count=1),
    ]
    resp = ChunkResponse(chunks=items)
    assert len(resp.chunks) == 2
    assert [c.chunk_index for c in resp.chunks] == [0, 1]


# ---- /v1/ingest schemas ------------------------------------------------


def test_ingest_response_defaults():
    resp = IngestResponse(
        doc_id="0a1b2c3d-4e5f-6789-abcd-ef0123456789",
        chunk_count=0,
        tokens_used=0,
    )
    assert resp.doc_id == "0a1b2c3d-4e5f-6789-abcd-ef0123456789"
    assert resp.chunk_count == 0
    assert resp.page_count is None  # default
    assert resp.tokens_used == 0


def test_ingest_response_with_page_count():
    resp = IngestResponse(
        doc_id="abc",
        chunk_count=10,
        page_count=5,
        tokens_used=5000,
    )
    assert resp.page_count == 5


def test_ingest_response_requires_doc_id():
    with pytest.raises(ValidationError):
        IngestResponse(chunk_count=1, tokens_used=1)  # type: ignore[call-arg]


def test_ingest_response_chunk_count_non_negative():
    """``chunk_count`` and ``tokens_used`` are non-negative ints."""
    with pytest.raises(ValidationError):
        IngestResponse(doc_id="x", chunk_count=-1, tokens_used=0)
    with pytest.raises(ValidationError):
        IngestResponse(doc_id="x", chunk_count=0, tokens_used=-1)


# ---- helpers ----------------------------------------------------------


def test_metadata_to_field_dict_full():
    md = ParseMetadata(
        page_count=12, title="Annual Report", author="Acme",
        mime_type="application/pdf",
    )
    out = metadata_to_field_dict(md)
    assert out == {
        "title": "Annual Report",
        "author": "Acme",
        "page_count": 12,
    }


def test_metadata_to_field_dict_partial():
    md = ParseMetadata(mime_type="text/plain")
    out = metadata_to_field_dict(md)
    # Only the keys that were populated are present — no None values.
    assert out == {}


def test_metadata_to_field_dict_none_input():
    assert metadata_to_field_dict(None) == {}


def test_metadata_to_field_dict_coerces_to_strings():
    md = ParseMetadata(title="123", author="456")
    out = metadata_to_field_dict(md)
    assert out["title"] == "123"
    assert out["author"] == "456"
    assert "page_count" not in out
