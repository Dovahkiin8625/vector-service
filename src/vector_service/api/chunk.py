"""``POST /v1/chunk`` — markdown → list of chunks.

Standalone route so callers can iterate on chunking parameters
without paying the parse cost on every tweak. The orchestrator
(``POST /v1/ingest``) calls the chunker in-process via
``RecursiveChunker.chunk`` rather than via HTTP.

The request body carries the markdown + chunk config; the response
is the chunk list in source order. ``page_numbers`` is optional —
callers that ran ``/v1/parse`` on a paged format can pass the page
numbers extracted by the parser through to the chunker.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter

from vector_service.chunking.recursive_chunker import (
    Chunk,
    RecursiveChunker,
)
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.ingest import ChunkItem, ChunkResponse

router = APIRouter(prefix="/v1", tags=["ingest"])


class ChunkRequest(BaseModel):
    """Request body for ``POST /v1/chunk``.

    ``metadata`` is a free-form dict callers can use to tag the
    chunk job (filename, doc id, …). It's returned verbatim on
    each emitted :class:`vector_service.schemas.ingest.ChunkItem`
    via the ``metadata`` echo field — handy for debugging chunk
    provenance when you're tuning chunk_size / chunk_overlap.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "markdown": "# Introduction\n\nFirst paragraph...",
                "chunk_size": 500,
                "chunk_overlap": 75,
                "metadata": {"title": "Annual Report", "filename": "report.pdf"},
            }
        }
    )

    markdown: str = Field(
        min_length=1,
        description="Markdown text to chunk. Must be non-empty.",
    )
    chunk_size: int = Field(
        default=500,
        ge=1,
        le=8192,
        description="Target upper bound on tokens per emitted chunk.",
    )
    chunk_overlap: int = Field(
        default=75,
        ge=0,
        le=4096,
        description=(
            "Tokens from the previous chunk's tail to prefix onto "
            "the next chunk. Must be strictly less than ``chunk_size``."
        ),
    )
    page_numbers: list[int] | None = Field(
        default=None,
        description=(
            "Optional list of page numbers in source order. When "
            "supplied the chunker propagates a ``page_number`` onto "
            "each emitted chunk."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form key/value tags echoed on every emitted chunk "
            "for debugging / provenance."
        ),
    )


def _validate_overlap_lt_size(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
            f"({chunk_size})"
        )


def _chunks_to_items(chunks: list[Chunk], echo_metadata: dict[str, Any]) -> list[ChunkItem]:
    """Convert internal :class:`Chunk` objects to API response items.

    The chunker emits ``Chunk`` directly — we wrap it in
    :class:`ChunkItem` so OpenAPI picks up the schema and clients
    see a stable shape (no ``metadata`` echo on items; we keep
    that at the request level).
    """
    return [
        ChunkItem(
            text=c.text,
            chunk_index=c.chunk_index,
            token_count=c.token_count,
            section_header=c.section_header,
            page_number=c.page_number,
        )
        for c in chunks
    ]


@router.post(
    "/chunk",
    response_model=ChunkResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid chunk config."},
        422: {"model": ErrorEnvelope, "description": "Validation failed."},
    },
    summary="Chunk markdown text",
    description=(
        "Split markdown into chunks sized for embedding. Recursive "
        "splitting is markdown-aware: code blocks stay atomic, "
        "headers are tracked as a breadcrumb path, and a configurable "
        "token overlap is preserved between consecutive chunks."
    ),
)
def chunk_markdown(body: ChunkRequest):
    try:
        _validate_overlap_lt_size(body.chunk_size, body.chunk_overlap)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "code": "invalid_chunk_config",
                "message": str(e),
                "chunk_size": body.chunk_size,
                "chunk_overlap": body.chunk_overlap,
            }},
        )

    chunker = RecursiveChunker(
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    chunks = chunker.chunk(body.markdown, page_numbers=body.page_numbers)
    return ChunkResponse(chunks=_chunks_to_items(chunks, body.metadata))
