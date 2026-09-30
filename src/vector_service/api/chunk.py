"""``POST /v1/chunk`` — markdown → list of chunks.

Standalone route so callers can iterate on chunking parameters
without paying the parse cost on every tweak. The orchestrator
(``POST /v1/ingest``) builds the same chunkers in-process via
:func:`vector_service.chunking.build_chunker`.

The request carries the markdown, the strategy name and its
options. Available strategies:

- ``fixed`` — hard token windows (option ``protect_code``);
- ``paragraph`` — paragraph-first greedy packing;
- ``recursive`` (default) — markdown-aware recursive splitting;
- ``semantic`` — sentence embedding breakpoints (needs the text
  embedder loaded);
- ``llm`` — LLM-selected topic boundaries (needs VS_LLM__*).

``add_context=true`` additionally runs Anthropic-style contextual
enrichment; each returned chunk may then carry a ``context``
prefix used at embed time.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from vector_service.chunking import build_chunker, list_strategies
from vector_service.chunking.base import Chunk
from vector_service.chunking.llm_chunker import (
    contextualize_chunks,
    get_chat_client,
    is_llm_configured,
    summarize_chunks,
)
from vector_service.core.config import get_settings
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.ingest import ChunkItem, ChunkResponse

router = APIRouter(prefix="/v1", tags=["ingest"])

STRATEGY_NAMES = ("fixed", "paragraph", "recursive", "semantic", "llm")
StrategyName = Literal["fixed", "paragraph", "recursive", "semantic", "llm"]


class ChunkRequest(BaseModel):
    """Request body for ``POST /v1/chunk``.

    ``options`` carries strategy-specific parameters; unknown keys
    are ignored (with a server log), never fatal. ``metadata`` is a
    free-form dict echoed for debugging provenance.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "markdown": "# Introduction\n\nFirst paragraph...",
                "strategy": "recursive",
                "options": {},
                "add_context": False,
                "add_summary": False,
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
    strategy: StrategyName = Field(
        default="recursive",
        description="Chunking strategy name.",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific options (breakpoint_percentile, protect_code, ...).",
    )
    add_context: bool = Field(
        default=False,
        description="Generate an LLM situating context per chunk (needs VS_LLM__*).",
    )
    add_summary: bool = Field(
        default=False,
        description="Generate an LLM factual summary per chunk (needs VS_LLM__*).",
    )
    chunk_size: int = Field(default=500, ge=1, le=8192)
    chunk_overlap: int = Field(default=75, ge=0, le=4096)
    page_numbers: list[int] | None = Field(
        default=None,
        description="Optional page numbers in source order, propagated onto chunks.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form key/value tags (kept at the request level).",
    )


def _validate_overlap_lt_size(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
            f"({chunk_size})"
        )


def _chunks_to_items(chunks: list[Chunk]) -> list[ChunkItem]:
    """Convert internal chunks to API response items."""
    return [
        ChunkItem(
            text=c.text,
            chunk_index=c.chunk_index,
            token_count=c.token_count,
            section_header=c.section_header,
            page_number=c.page_number,
            context=c.context,
            summary=c.summary,
        )
        for c in chunks
    ]


def _bad_request(code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=400, detail={"error": {"code": code, "message": message, **extra}})


def _unavailable(code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=503, detail={"error": {"code": code, "message": message, **extra}})


@router.post(
    "/chunk",
    response_model=ChunkResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Invalid chunk config."},
        422: {"model": ErrorEnvelope, "description": "Validation failed."},
        503: {"model": ErrorEnvelope, "description": "Required model/LLM unavailable."},
    },
    summary="Chunk markdown text",
    description=(
        "Split markdown into chunks sized for embedding using the "
        "selected strategy. Recursive splitting (default) is "
        "markdown-aware: code blocks stay atomic and headers are "
        "tracked as breadcrumbs. Semantic/llm strategies and "
        "add_context require a loaded embedder / configured LLM."
    ),
)
def chunk_markdown(request: Request, body: ChunkRequest):
    try:
        _validate_overlap_lt_size(body.chunk_size, body.chunk_overlap)
    except ValueError as e:
        raise _bad_request(
            "invalid_chunk_config", str(e),
            chunk_size=body.chunk_size, chunk_overlap=body.chunk_overlap,
        )

    if body.strategy not in list_strategies():
        # Defensive — Literal already guards the JSON body.
        raise _bad_request(
            "invalid_strategy",
            f"unknown chunking strategy {body.strategy!r}",
            allowed=list_strategies(),
        )

    # Strategy dependency resolution (pre-flight style).
    embed_fn = None
    chat_fn = None

    if body.strategy == "semantic":
        embedder = getattr(request.app.state, "embedder", None)
        if embedder is None:
            raise _unavailable(
                "embedder_unavailable",
                "semantic chunking needs a loaded text embedder; "
                "call POST /v1/models/{model}/load first",
            )
        embed_fn = embedder.embed_documents

    settings = get_settings()
    if body.strategy == "llm" or body.add_context or body.add_summary:
        if not is_llm_configured(settings):
            raise _unavailable(
                "llm_unavailable",
                "LLM is not configured; set VS_LLM__BASE_URL and "
                "VS_LLM__MODEL first",
            )
        chat_fn = get_chat_client(settings).as_chat_fn()

    chunker = build_chunker(
        body.strategy,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        options=body.options,
        embed_fn=embed_fn,
        chat_fn=chat_fn,
    )
    chunks = chunker.chunk(body.markdown, page_numbers=body.page_numbers)

    if body.add_context and chunks:
        contextualize_chunks(
            chunks,
            document=body.markdown,
            chat_fn=chat_fn,
            max_concurrency=settings.llm.max_concurrency,
        )

    if body.add_summary and chunks:
        summarize_chunks(
            chunks,
            chat_fn=chat_fn,
            max_concurrency=settings.llm.max_concurrency,
        )

    return ChunkResponse(chunks=_chunks_to_items(chunks))
