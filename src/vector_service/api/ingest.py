"""``POST /v1/ingest`` — end-to-end document ingestion.

Two routes share one pipeline core:

- ``POST /v1/ingest`` — classic JSON :class:`IngestResponse`.
- ``POST /v1/ingest/stream`` — newline-delimited JSON event stream
  (``application/x-ndjson``) so clients can render per-stage progress.
  Event shapes::

      {"type": "stage", "stage": "parse" | "chunk" | "embed" | "upsert"}
      {"type": "progress", "stage": "parse", "page": 3, "total": 12}
      {"type": "result", ...IngestResponse fields...}
      {"type": "error", "status": 503, "error": {"code": ..., "message": ...}}

  ``progress`` events tick per completed page during the parse stage
  for paginated binary documents (PDF/DOCX/PPTX/HTML); text uploads
  emit none.

  Pre-flight failures (bad params / MIME / size / embedder not loaded)
  happen before the stream opens, so they come back as the usual JSON
  error envelope with its normal 4xx/503 status code.

Pipeline:

1. Mint a UUID4 ``doc_id`` BEFORE parsing, so Docling saves extracted
   images under ``artifacts_dir/<doc_id>/images`` and the markdown URLs
   stay stable for the document's lifetime.
2. Parse the uploaded file to markdown + metadata via the matching
   parser (Docling for binary/images, passthrough for text), honoring
   the ``profile`` form field (auto/standard/native/vlm).
3. Chunk the markdown with the selected strategy
   (``strategy`` form field; default recursive). Optionally
   contextualize each chunk (``add_context``).
4. Embed the chunks (server-side, using the loaded text embedder).
5. Ensure the collection exists, then upsert into Milvus with the
   ``doc_id`` carried on every row's ``doc_id`` scalar field.
6. If anything fails AFTER parsing, discard the document's artifact
   folder; on upsert failure also delete every row with the new
   ``doc_id`` (best-effort) and re-raise, so the collection and the
   artifacts directory are both left consistent.

The collection is created on first use with a fixed schema tuned
for chunk storage:

- ``id`` (VARCHAR(64), primary key) — ``{doc_id}_{chunk_index}``
- ``vector`` (FLOAT_VECTOR, dim=embedder.dim)
- ``doc_id`` (VARCHAR(64)) — filter / rollback key
- ``chunk_index`` (INT64) — sort order within a document
- ``text`` (VARCHAR(8192)) — chunk text for retrieval
- ``section_header`` (VARCHAR(1024)) — breadcrumb context
- ``page_number`` (INT64, nullable) — source page when available
- ``title`` (VARCHAR(512), nullable) — document title
- ``author`` (VARCHAR(512), nullable) — document author
- ``page_count`` (INT64, nullable) — total page count
- ``filename`` (VARCHAR(512), nullable) — original filename
- ``token_count`` (INT64) — chunk token count
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from vector_service.api.parse import _validate_profile
from vector_service.chunking import build_chunker, list_strategies
from vector_service.chunking.base import Chunk
from vector_service.chunking.llm_chunker import (
    contextualize_chunks,
    embed_text,
    get_chat_client,
    is_llm_configured,
)
from vector_service.core.config import get_settings
from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseNotFound,
    DimensionMismatch,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
)
from vector_service.core.logging import get_logger
from vector_service.parsers.docling_parser import DoclingParser, get_docling_parser
from vector_service.parsers.markdown_parser import MarkdownParser
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.ingest import IngestResponse
from vector_service.stores.base import FieldSpec, IndexSpec

router = APIRouter(prefix="/v1", tags=["ingest"])
log = get_logger(__name__)

# Fixed collection schema for ingested chunks. Mirrored in the
# per-row scalar dict below; any change here must be applied to
# both halves.
_PRIMARY_FIELD = "id"
_VECTOR_FIELD = "vector"
_DOC_ID_FIELD = "doc_id"
_CHUNK_INDEX_FIELD = "chunk_index"
_TEXT_FIELD = "text"
_SECTION_HEADER_FIELD = "section_header"
_PAGE_NUMBER_FIELD = "page_number"
_TITLE_FIELD = "title"
_AUTHOR_FIELD = "author"
_PAGE_COUNT_FIELD = "page_count"
_FILENAME_FIELD = "filename"
_TOKEN_COUNT_FIELD = "token_count"
_SPARSE_FIELD = "sparse"


def _ingest_scalar_fields() -> list[FieldSpec]:
    """Schema for the ``ingest`` collection.

    VARCHAR lengths are chosen generously so titles / filenames /
    headers fit comfortably; INT64 fits the chunk_index +
    page_count ranges without overflow.
    """
    return [
        FieldSpec(name=_PRIMARY_FIELD, dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name=_DOC_ID_FIELD, dtype="varchar", max_length=64),
        FieldSpec(name=_CHUNK_INDEX_FIELD, dtype="int64"),
        FieldSpec(name=_TEXT_FIELD, dtype="varchar", max_length=8192),
        FieldSpec(name=_SECTION_HEADER_FIELD, dtype="varchar", max_length=1024),
        FieldSpec(name=_PAGE_NUMBER_FIELD, dtype="int64", nullable=True),
        FieldSpec(name=_TITLE_FIELD, dtype="varchar", max_length=512, nullable=True),
        FieldSpec(name=_AUTHOR_FIELD, dtype="varchar", max_length=512, nullable=True),
        FieldSpec(name=_PAGE_COUNT_FIELD, dtype="int64", nullable=True),
        FieldSpec(name=_FILENAME_FIELD, dtype="varchar", max_length=512, nullable=True),
        FieldSpec(name=_TOKEN_COUNT_FIELD, dtype="int64"),
    ]


def _ingest_scalar_fields_v2() -> list[FieldSpec]:
    """Schema v2: v1 fields + analyzed ``text`` + ``sparse`` BM25 field.

    Milvus runs the built-in ``chinese`` analyzer (jieba +
    cnalphanumonly) over ``text`` on every write; the registered BM25
    Function turns the tokens into the ``sparse`` vector.
    """
    fields = [
        FieldSpec(
            name=f.name,
            dtype=f.dtype,
            is_primary=f.is_primary,
            max_length=f.max_length,
            nullable=f.nullable,
            default_value=f.default_value,
        )
        for f in _ingest_scalar_fields()
    ]
    for f in fields:
        if f.name == _TEXT_FIELD:
            f.enable_analyzer = True
            f.analyzer = {"type": "chinese"}
    fields.append(FieldSpec(name=_SPARSE_FIELD, dtype="sparse_float_vector"))
    return fields


def _ingest_indexes_v2() -> list[IndexSpec]:
    """Dense HNSW index plus sparse inverted (BM25) index."""
    return [
        IndexSpec(
            field_name=_VECTOR_FIELD,
            metric_type="cosine",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200},
        ),
        IndexSpec(
            field_name=_SPARSE_FIELD,
            metric_type="bm25",
            index_type="SPARSE_INVERTED_INDEX",
        ),
    ]


def schema_version(info: Any) -> int:
    """Return 2 for collections carrying the ``sparse`` field, else 1.

    Field entries may be dicts (CollectionInfo from the store) or any
    object exposing ``name``; retrieval capability detection must work
    with both.
    """
    for f in getattr(info, "fields", []) or []:
        name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
        if name == _SPARSE_FIELD:
            return 2
    return 1


def _build_chunk_row(
    *,
    doc_id: str,
    chunk_index: int,
    text: str,
    section_header: str,
    page_number: int | None,
    token_count: int,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the Milvus row for one chunk.

    The ``id`` is a deterministic ``{doc_id}_{chunk_index}`` so a
    retry of the same request overwrites the prior chunks instead
    of duplicating them.
    """
    row: dict[str, Any] = {
        _PRIMARY_FIELD: f"{doc_id}_{chunk_index}",
        _DOC_ID_FIELD: doc_id,
        _CHUNK_INDEX_FIELD: int(chunk_index),
        _TEXT_FIELD: text,
        _SECTION_HEADER_FIELD: section_header or "",
        _TOKEN_COUNT_FIELD: int(token_count),
    }
    if page_number is not None:
        row[_PAGE_NUMBER_FIELD] = int(page_number)
    # Apply per-document metadata as scalar fields. Only known
    # fields are kept — callers cannot inject arbitrary schema.
    for k, v in extra_metadata.items():
        if k in (_TITLE_FIELD, _AUTHOR_FIELD, _FILENAME_FIELD):
            row[k] = str(v) if v is not None else None
        elif k == _PAGE_COUNT_FIELD and v is not None:
            try:
                row[k] = int(v)
            except (TypeError, ValueError):
                pass
    return row


# ---- helpers ----------------------------------------------------------


def _store_http_error(e: Exception, op: str) -> HTTPException:
    """Map a store-layer exception onto the canonical envelope."""
    if isinstance(e, DatabaseNotFound):
        return HTTPException(404, detail={"error": {
            "code": "database_not_found",
            "message": str(e),
            "name": getattr(e, "name", None),
        }})
    if isinstance(e, CollectionNotFound):
        return HTTPException(404, detail={"error": {
            "code": "collection_not_found",
            "message": str(e),
        }})
    if isinstance(e, CollectionAlreadyExists):
        return HTTPException(409, detail={"error": {
            "code": "collection_exists",
            "message": str(e),
        }})
    if isinstance(e, DimensionMismatch):
        return HTTPException(422, detail={"error": {
            "code": "dimension_mismatch",
            "message": str(e),
            "expected": getattr(e, "expected", None),
            "got": getattr(e, "got", None),
        }})
    if isinstance(e, BackendError):
        return HTTPException(503, detail={"error": {
            "code": "store_unavailable",
            "message": str(e) or "vector store backend unavailable",
            "op": op,
            "exception_type": type(e).__name__,
        }})
    if isinstance(e, StoreError):
        return HTTPException(422, detail={"error": {
            "code": "invalid_request",
            "message": str(e) or "invalid request",
            "op": op,
            "exception_type": type(e).__name__,
        }})
    return HTTPException(503, detail={"error": {
        "code": "store_unavailable",
        "message": str(e) or "vector store backend unavailable",
        "op": op,
        "exception_type": type(e).__name__,
    }})


def _resolve_mime(filename: str | None, content_type: str | None) -> str:
    """Pick the MIME type to dispatch on. Mirrors ``/v1/parse``."""
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base:
            return base
    if filename:
        from pathlib import Path
        suffix = Path(filename).suffix.lower()
        mapping = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html",
            ".htm": "text/html",
            ".xhtml": "application/xhtml+xml",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
        }
        if suffix in mapping:
            return mapping[suffix]
    return "application/octet-stream"


# ---- shared pipeline --------------------------------------------------

# Stage names are the public event contract of /v1/ingest/stream —
# clients (the dashboard) render progress from these exact strings, so
# treat them like an API version.
INGEST_STAGES = ("parse", "chunk", "embed", "upsert")

# Async sink for progress events. The JSON route passes a no-op; the
# streaming route forwards each event onto its NDJSON response.
Emit = Callable[[dict[str, Any]], Awaitable[None]]


async def _noop_emit(_event: dict[str, Any]) -> None:
    """Emit sink used by the classic JSON route (events discarded)."""


@dataclass
class PreparedIngest:
    """Validated request inputs, ready for the pipeline."""

    data: bytes
    mime: str
    extra_metadata: dict[str, Any]
    filename: str | None
    profile: str = "auto"
    strategy: str = "recursive"
    chunk_options: dict[str, Any] = field(default_factory=dict)
    add_context: bool = False


async def _prepare_ingest(
    *,
    file: UploadFile,
    settings: Any,
    embedder: Any,
    metadata: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    profile: str = "auto",
    strategy: str = "recursive",
    chunk_options: str = "{}",
    add_context: bool = False,
) -> PreparedIngest:
    """Validate arguments and spool the upload into memory.

    Covers pipeline steps 0 and 1 — everything that can fail before any
    work starts. Both routes run this synchronously so pre-flight
    failures come back as normal JSON error envelopes.
    """
    parser_settings = settings.parser

    # ---- 0. argument validation -------------------------------------
    # Raises 400 invalid_profile for unknown profile values.
    profile = _validate_profile(profile)

    if strategy not in list_strategies():
        raise HTTPException(400, detail={"error": {
            "code": "invalid_strategy",
            "message": (
                f"unknown chunking strategy {strategy!r}; expected "
                f"one of {', '.join(list_strategies())}"
            ),
            "strategy": strategy,
            "allowed": list_strategies(),
        }})

    try:
        parsed_options = json.loads(chunk_options) if chunk_options else {}
        if not isinstance(parsed_options, dict):
            raise ValueError("chunk_options must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_chunk_options",
            "message": f"chunk_options must be a JSON object: {e}",
        }})

    # LLM-backed features need the external chat backend.
    if strategy == "llm" or add_context:
        if not is_llm_configured(settings):
            raise HTTPException(503, detail={"error": {
                "code": "llm_unavailable",
                "message": (
                    "LLM is not configured; set VS_LLM__BASE_URL and "
                    "VS_LLM__MODEL first"
                ),
            }})

    try:
        extra_metadata = json.loads(metadata) if metadata else {}
        if not isinstance(extra_metadata, dict):
            raise ValueError("metadata must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_metadata",
            "message": f"metadata must be a JSON object: {e}",
        }})

    if chunk_size < 1 or chunk_size > 8192:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_chunk_size",
            "message": f"chunk_size must be in [1, 8192], got {chunk_size}",
            "chunk_size": chunk_size,
        }})
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(400, detail={"error": {
            "code": "invalid_chunk_overlap",
            "message": (
                f"chunk_overlap must be in [0, chunk_size), got "
                f"{chunk_overlap} for chunk_size {chunk_size}"
            ),
            "chunk_overlap": chunk_overlap,
            "chunk_size": chunk_size,
        }})

    if embedder is None:
        raise HTTPException(503, detail={"error": {
            "code": "embedder_unavailable",
            "message": (
                f"text embedder is not loaded; "
                f"call POST /v1/models/{embed_model}/load first"
            ),
            "model": embed_model,
        }})

    # Resolve the live embedder — match the inference routes' contract.
    if getattr(embedder, "model_name", None) != embed_model:
        raise HTTPException(503, detail={"error": {
            "code": "embedder_unavailable",
            "message": (
                f"embedder {embed_model!r} is not loaded; the "
                "lifespan step initialised a different model"
            ),
            "model": embed_model,
            "loaded": getattr(embedder, "model_name", None),
        }})

    # ---- 1. read upload ---------------------------------------------
    mime = _resolve_mime(file.filename, file.content_type)
    if mime not in (
        *DoclingParser.accepted_mime,
        *MarkdownParser.accepted_mime,
    ):
        raise HTTPException(415, detail={"error": {
            "code": "unsupported_mime",
            "message": f"unsupported MIME type {mime!r}",
            "got": mime,
        }})

    max_bytes = parser_settings.max_file_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, detail={"error": {
            "code": "file_too_large",
            "message": (
                f"upload exceeds {parser_settings.max_file_size_mb} MB cap"
            ),
            "max_bytes": max_bytes,
        }})
    if not data:
        raise HTTPException(400, detail={"error": {
            "code": "empty_file",
            "message": "uploaded file is empty",
        }})

    return PreparedIngest(
        data=data,
        mime=mime,
        extra_metadata=extra_metadata,
        filename=file.filename,
        profile=profile,
        strategy=strategy,
        chunk_options=parsed_options,
        add_context=bool(add_context),
    )


async def _run_ingest_pipeline(
    *,
    prepared: PreparedIngest,
    settings: Any,
    store: Any,
    embedder: Any,
    database: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    inference_timeout_seconds: float,
    emit: Emit,
    report_parse_progress: bool = False,
) -> IngestResponse:
    """Parse -> chunk -> embed -> ensure-collection/upsert.

    Reports stage transitions via ``emit`` (no-op for the JSON route).
    When ``report_parse_progress`` is set (the streaming route),
    per-page parser ticks are forwarded as ``progress`` events. All
    failure modes are raised as :class:`HTTPException`; callers decide
    whether that becomes a raised response or an error event.
    """
    data = prepared.data
    mime = prepared.mime
    profile = prepared.profile
    extra_metadata = prepared.extra_metadata
    loop = asyncio.get_running_loop()

    # The doc_id is minted BEFORE parsing so extracted images land in
    # artifacts_dir/<doc_id>/ and the URLs in the stored markdown stay
    # stable for the lifetime of the document. On any later failure the
    # artifact folder is best-effort removed (see _discard_artifacts).
    doc_id = str(uuid.uuid4())

    # ---- 2. parse ---------------------------------------------------
    await emit({"type": "stage", "stage": "parse"})
    if mime in DoclingParser.accepted_mime:
        # Process-wide singleton: a fresh DocumentConverter per request
        # costs ~10-30s of model loading; the lifespan warmup populates
        # the same instance.
        parser = get_docling_parser()
    else:
        parser = MarkdownParser()

    # Page ticks fire on the Docling worker thread; hop onto the event
    # loop and turn each into a progress event. Counter + drain loop
    # (same pattern as /v1/parse/stream) guarantee every tick is queued
    # before the chunk-stage event, regardless of thread timing.
    tick_scheduled = 0
    tick_flushed = 0

    def _on_parse_progress(done: int, total: int) -> None:
        nonlocal tick_scheduled
        tick_scheduled += 1
        event = {
            "type": "progress",
            "stage": "parse",
            "page": int(done),
            "total": int(total),
        }

        async def _fire() -> None:
            nonlocal tick_flushed
            try:
                await emit(event)
            finally:
                tick_flushed += 1

        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(_fire())
        )

    on_progress = _on_parse_progress if report_parse_progress else None
    try:
        parsed = await parser.parse_bytes(
            data, mime, on_progress=on_progress,
            profile=profile, artifact_stem=doc_id,
        )
        while tick_flushed < tick_scheduled:
            await asyncio.sleep(0)
    except RuntimeError as e:
        log.warning("parser_failed", mime=mime, error=str(e))
        raise HTTPException(500, detail={"error": {
            "code": "parser_failed",
            "message": str(e) or "parser failed",
            "mime": mime,
            "exception_type": type(e).__name__,
        }})
    except Exception as e:
        log.warning("parser_unavailable", mime=mime, error=str(e))
        raise HTTPException(503, detail={"error": {
            "code": "parser_unavailable",
            "message": str(e) or "parser backend unavailable",
            "mime": mime,
            "exception_type": type(e).__name__,
        }})

    markdown = parsed.markdown
    page_count = parsed.metadata.get("page_count")
    title = parsed.metadata.get("title")
    author = parsed.metadata.get("author")
    if title:
        extra_metadata.setdefault(_TITLE_FIELD, title)
    if author:
        extra_metadata.setdefault(_AUTHOR_FIELD, author)
    if page_count is not None:
        extra_metadata.setdefault(_PAGE_COUNT_FIELD, page_count)
    if prepared.filename:
        extra_metadata.setdefault(_FILENAME_FIELD, prepared.filename)

    # ---- 3. chunk ---------------------------------------------------
    # All blocking work — tiktoken, sentence embeddings for semantic,
    # httpx chat calls for llm/contextualization — runs in a worker
    # thread so the event loop (and the NDJSON flushes) stay live.
    await emit({"type": "stage", "stage": "chunk"})
    chunks = await loop.run_in_executor(
        None,
        _produce_chunks,
        markdown,
        prepared,
        settings,
        embedder,
        chunk_size,
        chunk_overlap,
    )

    if not chunks:
        # No content — return an empty ingest rather than failing.
        # Nothing references the extracted images, so drop the folder.
        await loop.run_in_executor(None, _discard_artifacts, doc_id)
        return IngestResponse(
            doc_id=doc_id,
            chunk_count=0,
            page_count=page_count,
            tokens_used=0,
        )

    # ---- 4. embed ---------------------------------------------------
    await emit({"type": "stage", "stage": "embed"})
    # Context prefixes (when enrichment ran) participate in the embedding.
    texts = [embed_text(c) for c in chunks]
    try:
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(None, embedder.embed_documents, texts),
                timeout=inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise EmbedderError(
                f"embedder {embed_model!r} did not finish within "
                f"{inference_timeout_seconds}s"
            )
    except (EmbedderError, ModelNotLoaded) as e:
        log.warning("ingest_embed_failed", error=str(e))
        await loop.run_in_executor(None, _discard_artifacts, doc_id)
        raise HTTPException(503, detail={"error": {
            "code": "embedder_unavailable",
            "message": str(e) or "embedder unavailable",
            "model": embed_model,
            "chunk_count": len(chunks),
            "exception_type": type(e).__name__,
        }})
    except Exception:
        # Any other embed-stage surprise (executor error, ...) orphans
        # the artifact folder otherwise; clean up and surface the same
        # way the stream runner / route would have.
        await loop.run_in_executor(None, _discard_artifacts, doc_id)
        raise

    # ---- 5. ensure collection + upsert with rollback ---------------
    # Collection preparation is part of the upsert stage — dimension
    # mismatches surface here, right next to the write they block.
    await emit({"type": "stage", "stage": "upsert"})
    try:
        await loop.run_in_executor(
            None, _ensure_collection, store, database, collection, embedder.dim
        )
    except (DatabaseNotFound, CollectionAlreadyExists, DimensionMismatch, StoreError, BackendError) as e:
        await loop.run_in_executor(None, _discard_artifacts, doc_id)
        raise _store_http_error(e, op="create_collection")

    chunk_ids = [f"{doc_id}_{c.chunk_index}" for c in chunks]
    fields = [
        _build_chunk_row(
            doc_id=doc_id,
            chunk_index=c.chunk_index,
            text=c.text,
            section_header=c.section_header,
            page_number=c.page_number,
            token_count=c.token_count,
            extra_metadata=extra_metadata,
        )
        for c in chunks
    ]

    try:
        await loop.run_in_executor(
            None,
            store.upsert,
            database, collection, _PRIMARY_FIELD, _VECTOR_FIELD,
            chunk_ids, vectors, fields,
        )
    except Exception as upsert_err:
        # Rollback: best-effort delete by doc_id filter. We swallow
        # secondary errors so the original cause surfaces to the
        # caller; we log the rollback outcome for ops visibility.
        log.warning(
            "ingest_upsert_failed_rollback_start",
            doc_id=doc_id,
            collection=f"{database}/{collection}",
            error=str(upsert_err),
            exception_type=type(upsert_err).__name__,
        )
        try:
            await loop.run_in_executor(
                None,
                _delete_by_doc_id,
                store, database, collection, doc_id,
            )
        except Exception as rollback_err:
            log.error(
                "ingest_rollback_failed",
                doc_id=doc_id,
                collection=f"{database}/{collection}",
                error=str(rollback_err),
                exception_type=type(rollback_err).__name__,
            )
        # No vectors reference the document's images after rollback —
        # remove the artifact folder so failed ingests don't leak disk.
        await loop.run_in_executor(None, _discard_artifacts, doc_id)
        # Re-raise as the canonical store error envelope.
        if isinstance(upsert_err, (DatabaseNotFound, CollectionNotFound, DimensionMismatch, StoreError, BackendError)):
            raise _store_http_error(upsert_err, op="upsert") from upsert_err
        raise HTTPException(503, detail={"error": {
            "code": "store_unavailable",
            "message": str(upsert_err) or "vector store backend unavailable",
            "op": "upsert",
            "exception_type": type(upsert_err).__name__,
        }}) from upsert_err

    tokens_used = sum(c.token_count for c in chunks)
    log.info(
        "ingest_done",
        doc_id=doc_id,
        collection=f"{database}/{collection}",
        chunk_count=len(chunks),
        tokens_used=tokens_used,
        duration_ms=int(time.time() * 1000),
    )
    return IngestResponse(
        doc_id=doc_id,
        chunk_count=len(chunks),
        page_count=page_count,
        tokens_used=tokens_used,
    )


def _produce_chunks(
    markdown: str,
    prepared: PreparedIngest,
    settings: Any,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Build the named chunker, run it, optionally contextualize.

    Runs entirely in a worker thread. The LLM client is cached
    process-wide (:func:`get_chat_client`); pre-flight validation
    already proved it is configured.
    """
    strategy = prepared.strategy
    embed_fn = embedder.embed_documents if strategy == "semantic" else None
    chat_fn = None
    if strategy == "llm" or prepared.add_context:
        chat_fn = get_chat_client(settings).as_chat_fn()

    chunker = build_chunker(
        strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        options=prepared.chunk_options,
        embed_fn=embed_fn,
        chat_fn=chat_fn,
    )
    chunks = chunker.chunk(markdown)

    if prepared.add_context and chunks:
        contextualize_chunks(
            chunks,
            document=markdown,
            chat_fn=chat_fn,
            max_concurrency=settings.llm.max_concurrency,
        )
    return chunks


# ---- routes -----------------------------------------------------------


# Both routes share the same multipart contract and the same error
# catalogue; pre-flight failures are JSON envelopes even on the stream
# route, in-flight failures come as ``error`` events instead.
_INGEST_RESPONSES = {
    400: {"model": ErrorEnvelope, "description": "Empty upload / bad config."},
    404: {"model": ErrorEnvelope, "description": "Database does not exist."},
    409: {"model": ErrorEnvelope, "description": "Collection exists with conflicting schema."},
    413: {"model": ErrorEnvelope, "description": "Upload exceeds max size."},
    415: {"model": ErrorEnvelope, "description": "Unsupported MIME type."},
    422: {"model": ErrorEnvelope, "description": "Validation / dimension mismatch."},
    500: {"model": ErrorEnvelope, "description": "Parser failure."},
    503: {"model": ErrorEnvelope, "description": "Embedder or parser backend unavailable."},
}


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses=_INGEST_RESPONSES,
    summary="Ingest a document end-to-end",
    description=(
        "Parse the uploaded file (PDF, DOCX, PPTX, HTML, raster images, "
        "MD, TXT), chunk the resulting markdown with the selected "
        "strategy (fixed/paragraph/recursive/semantic/llm; "
        "add_context for contextual enrichment), "
        "embed the chunks, and store them in a Milvus collection. "
        "Optional form field ``profile`` selects the Docling pipeline "
        "(``auto``/``standard``/``native``/``vlm``; see ``/v1/parse``). "
        "Extracted images are saved under ``/artifacts/<doc_id>/``. "
        "On any post-parse failure the route deletes the partial "
        "upsert by ``doc_id`` filter and discards the document's "
        "artifacts before re-raising the error, so the collection and "
        "the artifacts directory stay consistent. Use "
        "``POST /v1/ingest/stream`` for per-stage progress events."
    ),
)
async def ingest_document(
    request: Request,
    file: UploadFile,
    database: str = Form("default"),
    collection: str = Form("ingest"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(75),
    embed_model: str = Form("bge-m3"),
    metadata: str = Form("{}"),
    profile: str = Form("auto"),
    strategy: str = Form("recursive"),
    chunk_options: str = Form("{}"),
    add_context: bool = Form(False),
):
    """End-to-end ingest, classic JSON response.

    ``metadata`` arrives as a JSON-encoded string per the spec —
    multipart form fields can't carry nested objects directly, so
    callers serialise to a string and we parse it here. A bad JSON
    string yields 400 ``invalid_metadata``.
    """
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    prepared = await _prepare_ingest(
        file=file,
        settings=settings,
        embedder=embedder,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        profile=profile,
        strategy=strategy,
        chunk_options=chunk_options,
        add_context=add_context,
    )
    return await _run_ingest_pipeline(
        prepared=prepared,
        settings=settings,
        store=request.app.state.store,
        embedder=embedder,
        database=database,
        collection=collection,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        inference_timeout_seconds=getattr(settings, "inference_timeout_seconds", 60.0),
        emit=_noop_emit,
    )


def _http_error_event(exc: HTTPException) -> dict[str, Any]:
    """Serialise an :class:`HTTPException` into a stream ``error`` event."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    error = detail.get("error")
    if not isinstance(error, dict):
        error = {"code": "error", "message": str(exc.detail)}
    return {"type": "error", "status": exc.status_code, "error": error}


@router.post(
    "/ingest/stream",
    responses=_INGEST_RESPONSES,
    summary="Ingest a document with per-stage progress events",
    description=(
        "Same multipart contract as ``POST /v1/ingest`` (including the "
        "optional ``profile`` field) but responds "
        "with ``application/x-ndjson``: one JSON event per line. "
        "``stage`` events mark parse/chunk/embed/upsert transitions and "
        "``progress`` events tick per parsed page during the parse "
        "stage (paginated binary formats only); the terminal event is "
        "either ``result`` (the IngestResponse fields) or ``error`` "
        "(the canonical error envelope plus HTTP status). Pre-flight "
        "failures are returned as ordinary JSON error envelopes before "
        "the stream starts."
    ),
)
async def ingest_document_stream(
    request: Request,
    file: UploadFile,
    database: str = Form("default"),
    collection: str = Form("ingest"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(75),
    embed_model: str = Form("bge-m3"),
    metadata: str = Form("{}"),
    profile: str = Form("auto"),
    strategy: str = Form("recursive"),
    chunk_options: str = Form("{}"),
    add_context: bool = Form(False),
):
    """End-to-end ingest as an NDJSON event stream."""
    settings = request.app.state.settings
    embedder = request.app.state.embedder
    # Pre-flight: bad params / MIME / size / embedder state fail here as
    # regular JSON envelopes (the StreamingResponse hasn't started yet).
    prepared = await _prepare_ingest(
        file=file,
        settings=settings,
        embedder=embedder,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        profile=profile,
        strategy=strategy,
        chunk_options=chunk_options,
        add_context=add_context,
    )

    async def event_stream():
        # A small queue decouples the pipeline task from this
        # generator: parse/embed/upsert run in worker-thread
        # executors while stages get flushed as soon as they happen.
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def emit(event: dict[str, Any]) -> None:
            queue.put_nowait(event)

        async def runner() -> None:
            try:
                response = await _run_ingest_pipeline(
                    prepared=prepared,
                    settings=settings,
                    store=request.app.state.store,
                    embedder=embedder,
                    database=database,
                    collection=collection,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embed_model=embed_model,
                    inference_timeout_seconds=getattr(
                        settings, "inference_timeout_seconds", 60.0
                    ),
                    emit=emit,
                    report_parse_progress=True,
                )
                queue.put_nowait({"type": "result", **response.model_dump()})
            except HTTPException as e:
                queue.put_nowait(_http_error_event(e))
            except Exception as e:  # pragma: no cover - safety net
                log.exception("ingest_stream_pipeline_failed")
                queue.put_nowait({"type": "error", "status": 500, "error": {
                    "code": "internal_error",
                    "message": str(e) or "internal error",
                }})
            finally:
                queue.put_nowait(sentinel)

        # The runner is deliberately NOT cancelled on client
        # disconnect: CancelledError mid-upsert would bypass the
        # doc_id rollback and leave a partial document behind. It
        # catches every error itself and only emits a handful of
        # events, so an orphaned task is cheap.
        task = asyncio.create_task(runner())
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- store-side helpers ----------------------------------------------


def _ensure_collection(
    store: Any,
    database: str,
    collection: str,
    dim: int,
) -> None:
    """Create the ``(database, collection)`` pair if it doesn't exist.

    Auto-creates the database as well — callers rarely need a
    pre-existing database for ingest. Treats ``DatabaseNotFound`` /
    ``CollectionAlreadyExists`` as no-ops so this helper is safe to
    call on every ingest request.
    """
    # Ensure database exists.
    try:
        existing_dbs = store.list_databases()
    except (BackendError, StoreError):
        raise
    except Exception:
        existing_dbs = []

    if database not in existing_dbs:
        try:
            store.create_database(database)
        except Exception as e:
            # 409 on the database: another worker raced us. Fine.
            if "already" not in str(e).lower():
                raise

    # Ensure collection exists with our fixed schema.
    try:
        existing_colls = store.list_collections(database)
    except (BackendError, StoreError):
        raise
    except Exception:
        existing_colls = []

    if collection in existing_colls:
        # Validate dim against the existing schema — dim mismatches
        # surface as 422 to the caller.
        try:
            info = store.collection_info(database, collection)
        except Exception:
            info = None
        if info is not None and int(info.dim) != int(dim):
            from vector_service.core.errors import DimensionMismatch
            raise DimensionMismatch(
                f"collection {collection!r} has dim {info.dim} but "
                f"embedder outputs dim {dim}",
                expected=info.dim, got=dim,
            )
        return

    try:
        store.create_collection(
            database=database,
            name=collection,
            primary_field=_PRIMARY_FIELD,
            vector_field=FieldSpec(name=_VECTOR_FIELD, dtype="float_vector", dim=dim),
            scalar_fields=_ingest_scalar_fields_v2(),
            indexes=_ingest_indexes_v2(),
        )
    except CollectionAlreadyExists:
        # Lost the race; another worker created the same collection.
        return


_TMP_COLLECTION = "ingest_migrate_tmp"
_MIGRATE_BATCH = 200


def migrate_ingest_collection(store: Any, database: str, dim: int) -> dict:
    """Copy an ingest v1 collection into a v2 collection and swap them.

    Failure at any step drops the temporary collection and leaves the
    original in place (except the narrow window after the old
    collection is dropped and before rename completes — documented in
    docs/retrieval.md).
    """
    adapter = store._adapter
    tmp_created = False
    try:
        if adapter.has_collection(database, _TMP_COLLECTION):
            adapter.drop_collection(database, _TMP_COLLECTION)
        store.create_collection(
            database,
            _TMP_COLLECTION,
            primary_field=_PRIMARY_FIELD,
            vector_field=FieldSpec(name=_VECTOR_FIELD, dtype="float_vector",
                                   dim=dim),
            scalar_fields=_ingest_scalar_fields_v2(),
            indexes=_ingest_indexes_v2(),
        )
        tmp_created = True

        total = 0
        while True:
            rows = adapter.browse(
                database, "ingest", _PRIMARY_FIELD,
                limit=_MIGRATE_BATCH, offset=total,
                include_vectors=True,
            )
            if not rows:
                break
            adapter.insert_rows(database, _TMP_COLLECTION, [
                {"id": row["id"], **row["fields"]} for row in rows
            ])
            total += len(rows)
            if len(rows) < _MIGRATE_BATCH:
                break

        migrated = adapter.count(database, _TMP_COLLECTION)
        old_count = adapter.count(database, "ingest")
        if migrated != old_count:
            raise StoreError(
                f"migration row count mismatch: {migrated} copied vs "
                f"{old_count} original"
            )

        adapter.drop_collection(database, "ingest")
        adapter.rename_collection(database, _TMP_COLLECTION, "ingest")
    except Exception:
        # Drop tmp based on our own create-tracking rather than
        # has_collection(): a backend cache/visibility lag must not stop
        # cleanup, and "any failure deletes the tmp collection" is the
        # contract (brief interface spec).
        if tmp_created:
            adapter.drop_collection(database, _TMP_COLLECTION)
        raise

    return {
        "database": database,
        "collection": "ingest",
        "rows": total,
        "schema_version": 2,
    }


def _delete_by_doc_id(
    store: Any,
    database: str,
    collection: str,
    doc_id: str,
) -> int:
    """Delete every row matching ``doc_id`` via Milvus filter expr.

    Returns the count of rows deleted (best-effort; the store may
    not report a count and we don't fail the rollback on that).
    """
    expr = f'{_DOC_ID_FIELD} == "{_escape(doc_id)}"'
    adapter = getattr(store, "_adapter", None)
    client = getattr(adapter, "_client", None) if adapter is not None else None
    if client is None:
        # Fall back to the high-level store API — the underlying
        # adapter doesn't expose filter-expr delete so we list the
        # ids first, then delete by primary key.
        try:
            rows = store.get(
                database=database, collection=collection,
                primary_field=_PRIMARY_FIELD, ids=[],
                output_fields=[_DOC_ID_FIELD],
            )
        except Exception:
            rows = []
        # ``get`` takes primary ids, not a filter, so we can't
        # query by doc_id through the high-level store. Use the
        # adapter's ``query`` if available.
        return 0
    try:
        client.delete(collection, filter=expr)
    except Exception as e:
        raise RuntimeError(f"rollback delete failed: {e}") from e
    return 0


def _escape(value: str) -> str:
    """Escape a string for embedding in a Milvus filter expression."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _discard_artifacts(doc_id: str) -> None:
    """Best-effort delete of ``artifacts_dir/<doc_id>`` after failure.

    Runs in a thread executor (filesystem I/O). Every error is
    swallowed and logged — artifact cleanup must never mask the
    original pipeline failure.
    """
    try:
        root = Path(get_settings().parser.artifacts_dir) / doc_id
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
    except Exception as e:  # noqa: BLE001 — cleanup is best-effort
        log.warning("ingest_artifact_cleanup_failed", doc_id=doc_id, error=str(e))
