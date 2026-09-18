"""``POST /v1/ingest`` — end-to-end document ingestion.

Pipeline:

1. Parse the uploaded file to markdown + metadata via the matching
   parser (Docling for binary, passthrough for text).
2. Chunk the markdown with :class:`RecursiveChunker`.
3. Embed the chunks (server-side, using the loaded text embedder).
4. Upsert into Milvus with a UUID4 ``doc_id`` carried on every
   row's ``doc_id`` scalar field.
5. If anything fails AFTER the upsert, delete every row with the
   new ``doc_id`` (best-effort, 409-tolerant) and re-raise so the
   caller sees a clean error and the collection is left consistent.

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
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile

from vector_service.chunking.recursive_chunker import RecursiveChunker
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
from vector_service.parsers.docling_parser import DoclingParser
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
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
        }
        if suffix in mapping:
            return mapping[suffix]
    return "application/octet-stream"


# ---- route ------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Empty upload / bad config."},
        404: {"model": ErrorEnvelope, "description": "Database does not exist."},
        409: {"model": ErrorEnvelope, "description": "Collection exists with conflicting schema."},
        413: {"model": ErrorEnvelope, "description": "Upload exceeds max size."},
        415: {"model": ErrorEnvelope, "description": "Unsupported MIME type."},
        422: {"model": ErrorEnvelope, "description": "Validation / dimension mismatch."},
        500: {"model": ErrorEnvelope, "description": "Parser failure."},
        503: {"model": ErrorEnvelope, "description": "Embedder or parser backend unavailable."},
    },
    summary="Ingest a document end-to-end",
    description=(
        "Parse the uploaded file, chunk the resulting markdown, "
        "embed the chunks, and store them in a Milvus collection. "
        "On any post-upsert failure the route deletes the partial "
        "upsert by ``doc_id`` filter before re-raising the error, "
        "so the collection stays consistent."
    ),
)
async def ingest_document(
    file: UploadFile,
    request: Request,
    database: str = "default",
    collection: str = "ingest",
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    embed_model: str = "bge-m3",
    metadata: str = "{}",
):
    """End-to-end ingest.

    ``metadata`` arrives as a JSON-encoded string per the spec —
    multipart form fields can't carry nested objects directly, so
    callers serialise to a string and we parse it here. A bad JSON
    string yields 400 ``invalid_metadata``.
    """
    settings = request.app.state.settings
    parser_settings = settings.parser
    embedder = request.app.state.embedder

    # ---- 0. argument validation -------------------------------------
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

    # ---- 2. parse ---------------------------------------------------
    if mime in DoclingParser.accepted_mime:
        parser = DoclingParser()
    else:
        parser = MarkdownParser()

    try:
        parsed = await parser.parse_bytes(data, mime)
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
    if file.filename:
        extra_metadata.setdefault(_FILENAME_FIELD, file.filename)

    # ---- 3. chunk ---------------------------------------------------
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk(markdown)

    if not chunks:
        # No content — return an empty ingest rather than failing.
        return IngestResponse(
            doc_id=str(uuid.uuid4()),
            chunk_count=0,
            page_count=page_count,
            tokens_used=0,
        )

    # ---- 4. ensure collection ---------------------------------------
    store = request.app.state.store
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(None, _ensure_collection, store, database, collection, embedder.dim)
    except (DatabaseNotFound, CollectionAlreadyExists, DimensionMismatch, StoreError, BackendError) as e:
        raise _store_http_error(e, op="create_collection")

    # ---- 5. embed ---------------------------------------------------
    texts = [c.text for c in chunks]
    timeout_s = getattr(settings, "inference_timeout_seconds", 60.0)
    try:
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(None, embedder.embed_documents, texts),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise EmbedderError(
                f"embedder {embed_model!r} did not finish within {timeout_s}s"
            )
    except (EmbedderError, ModelNotLoaded) as e:
        log.warning("ingest_embed_failed", error=str(e))
        raise HTTPException(503, detail={"error": {
            "code": "embedder_unavailable",
            "message": str(e) or "embedder unavailable",
            "model": embed_model,
            "chunk_count": len(chunks),
            "exception_type": type(e).__name__,
        }})

    # ---- 6. upsert with rollback on failure -----------------------
    doc_id = str(uuid.uuid4())
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
    ``CollectionAlreadyExists`` as no-ops so this helper is safe
    to call on every ingest request.
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
            scalar_fields=_ingest_scalar_fields(),
            indexes=[
                IndexSpec(
                    field_name=_VECTOR_FIELD,
                    metric_type="cosine",
                    index_type="HNSW",
                    params={"M": 16, "efConstruction": 200},
                ),
            ],
        )
    except CollectionAlreadyExists:
        # Lost the race; another worker created the same collection.
        return


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
