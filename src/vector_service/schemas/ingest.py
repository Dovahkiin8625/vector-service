"""Pydantic schemas for the document-ingest API.

Three endpoint families share this module:

- ``POST /v1/parse`` — file → markdown + metadata. Returns
  :class:`ParseResponse`.
- ``POST /v1/chunk`` — markdown → list of chunks. Returns
  :class:`ChunkResponse`.
- ``POST /v1/ingest`` — file → parse → chunk → embed → upsert (with
  atomic rollback on any post-upsert failure). Returns
  :class:`IngestResponse`.

Schema names follow the conventions used elsewhere in
:mod:`vector_service.schemas` — request bodies live in
``*Request`` and responses in ``*Response`` / ``*Item``.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---- /v1/parse --------------------------------------------------------


class ParseMetadata(BaseModel):
    """Metadata block returned by ``POST /v1/parse``.

    ``page_count`` is ``None`` for formats that don't have pages
    (markdown / text) — the JSON schema marks it nullable so clients
    don't have to special-case 0 vs missing.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page_count": 12,
                "title": "Annual Report",
                "author": "Acme Inc.",
                "mime_type": "application/pdf",
            }
        }
    )

    page_count: int | None = Field(
        default=None,
        description=(
            "Number of pages in the source document. ``None`` for "
            "formats without pages (markdown, plain text)."
        ),
        examples=[12],
    )
    title: str | None = Field(
        default=None,
        description="Document title if the parser could extract one.",
        examples=["Annual Report"],
    )
    author: str | None = Field(
        default=None,
        description="Document author if the parser could extract one.",
        examples=["Acme Inc."],
    )
    mime_type: str | None = Field(
        default=None,
        description="Source MIME type the parser accepted.",
        examples=["application/pdf"],
    )


class ParseResponse(BaseModel):
    """Response of ``POST /v1/parse``.

    Carries the full markdown representation produced by the parser
    plus a small metadata block. The orchestrator route (``/v1/ingest``)
    forwards ``markdown`` directly into the chunker; standalone
    callers (typically debugging / preview UIs) get the full text
    back so they can inspect what the parser produced before
    embedding.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "markdown": "# Introduction\n\nThis is the first paragraph...",
                "metadata": {
                    "page_count": 12,
                    "title": "Annual Report",
                    "author": "Acme Inc.",
                    "mime_type": "application/pdf",
                },
            }
        }
    )

    markdown: str = Field(
        description="Full markdown representation of the parsed document.",
    )
    metadata: ParseMetadata = Field(
        description="Parser-supplied metadata about the source document.",
    )


# ---- /v1/chunk --------------------------------------------------------


class ChunkItem(BaseModel):
    """One chunk returned by ``POST /v1/chunk``.

    ``section_header`` is the breadcrumb header path (``"1.
    Introduction > 1.1 Background"``) under which the chunk was
    emitted; ``""`` for chunks emitted from header-less input.
    ``page_number`` is ``None`` for sources without pages.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The quick brown fox jumps over the lazy dog.",
                "chunk_index": 0,
                "token_count": 487,
                "section_header": "1. Introduction > 1.1 Background",
                "page_number": 1,
            }
        }
    )

    text: str = Field(description="Chunk text.")
    chunk_index: int = Field(
        description="0-based position of this chunk within the input.",
        examples=[0],
    )
    token_count: int = Field(
        description="Number of tokens in ``text`` under ``cl100k_base``.",
        examples=[487],
    )
    section_header: str = Field(
        default="",
        description=(
            "Breadcrumb header path for the chunk. Empty when the "
            "input had no headers."
        ),
        examples=["1. Introduction > 1.1 Background"],
    )
    page_number: int | None = Field(
        default=None,
        description=(
            "Source page number when the parser supplied one; "
            "``None`` for sources without pages (markdown, plain text)."
        ),
        examples=[1],
    )


class ChunkResponse(BaseModel):
    """Response of ``POST /v1/chunk``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunks": [
                    {
                        "text": "First chunk text.",
                        "chunk_index": 0,
                        "token_count": 487,
                        "section_header": "1. Introduction > 1.1 Background",
                        "page_number": 1,
                    },
                ]
            }
        }
    )

    chunks: list[ChunkItem] = Field(
        description="Chunks in source order.",
    )


# ---- /v1/ingest -------------------------------------------------------


class IngestResponse(BaseModel):
    """Response of ``POST /v1/ingest``.

    On success the orchestrator returns the new ``doc_id`` plus the
    totals that an operator might want to log or surface in the UI.
    On failure (any post-upsert error) the route first deletes the
    partial upsert by ``doc_id`` filter and then re-raises — so a
    caller that retries the same file gets a fresh ``doc_id`` and a
    clean collection state.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "0a1b2c3d-4e5f-6789-abcd-ef0123456789",
                "chunk_count": 23,
                "page_count": 12,
                "tokens_used": 12453,
            }
        }
    )

    doc_id: str = Field(
        description=(
            "UUID4 identifier assigned to this document. All "
            "vectors written by this request carry this id in their "
            "``doc_id`` scalar field so callers can later filter or "
            "delete by document."
        ),
    )
    chunk_count: int = Field(
        ge=0,
        description="Number of chunks produced and stored.",
        examples=[23],
    )
    page_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of pages reported by the parser. ``None`` for "
            "formats without pages."
        ),
        examples=[12],
    )
    tokens_used: int = Field(
        ge=0,
        description=(
            "Sum of token counts across all emitted chunks — a "
            "loose accounting figure useful for cost estimation."
        ),
        examples=[12453],
    )


# Re-export metadata helper for the route layer to use when
# forwarding the per-document metadata into Milvus ``fields``.
def metadata_to_field_dict(metadata: ParseMetadata | None) -> dict[str, Any]:
    """Flatten a :class:`ParseMetadata` into a Milvus ``fields`` row.

    The Milvus schema only declares ``varchar``/``int``/``float``
    scalar fields, so the helper coerces each known key to its
    declared type. Unknown keys are dropped — callers that need a
    richer payload should add fields to the collection schema
    first.
    """
    if metadata is None:
        return {}
    out: dict[str, Any] = {}
    if metadata.title is not None:
        out["title"] = str(metadata.title)
    if metadata.author is not None:
        out["author"] = str(metadata.author)
    if metadata.page_count is not None:
        out["page_count"] = int(metadata.page_count)
    return out
