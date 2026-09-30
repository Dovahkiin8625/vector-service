"""Shared ingest pipeline core — parse → chunk → embed → upsert.

The synchronous ``/v1/ingest`` routes were removed; documents enter
through ``POST /v1/jobs/ingest`` (see :mod:`vector_service.api.jobs`)
and are executed by the background worker. This module holds the
pieces both sides reuse:

- :func:`_prepare_ingest` — argument/MIME/size validation and upload
  spooling (embedder-state checks are optional there, so a submission
  can precede the model being loaded);
- :class:`PreparedIngest`, :func:`_run_ingest_pipeline` and
  :func:`_produce_chunks` — the end-to-end pipeline;
- :func:`_ensure_collection`, :func:`_delete_by_doc_id` and
  :func:`_discard_artifacts` — thin-index bootstrap / rollback.

Progress events delivered through the pipeline's ``emit`` sink::

    {"type": "stage", "stage": "parse" | "chunk" | "embed" | "upsert"}
    {"type": "progress", "stage": "parse", "page": 3, "total": 12}

``progress`` events tick per completed page during the parse stage
for paginated binary documents (PDF/DOCX/PPTX/HTML); text uploads
emit none.

Architecture: the SQLite **corpus is the system of record**; Milvus is
a thin, rebuildable **derived index**. The pipeline parses the upload
to markdown, chunks it (optional contextual prefix), embeds dense
vectors, writes document + chunks to SQLite BEFORE any vectors, fits
client-side BM25 and encodes sparse vectors, then ensures the thin
collection and upserts.

On failure at any stage the job is marked ``failed`` and cleanup runs:
corpus delete (cascades the index registry), Milvus delete by
``doc_id`` (best-effort), and artifact-folder discard.

The thin collection schema:

- ``id`` (VARCHAR(64), primary key) — ``{doc_id}_{chunk_index}``
- ``vector`` (FLOAT_VECTOR, dim=embedder.dim)
- ``doc_id`` (VARCHAR(64)) — filter / rollback key
- ``chunk_index`` (INT64) — sort order within a document
- ``sparse`` (SPARSE_FLOAT_VECTOR) — client-side BM25 vector
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

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
from vector_service.core.threadpools import (
    run_in_model,
    run_in_sqlite,
    run_in_store,
)
from vector_service.corpus import (
    JOB_CHUNKING,
    JOB_EMBEDDING,
    JOB_PARSING,
    JOB_UPSERTING,
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
    IndexEntry,
)
from vector_service.parsers.docling_parser import DoclingParser, get_docling_parser
from vector_service.parsers.markdown_parser import MarkdownParser
from vector_service.schemas.ingest import IngestResponse
from vector_service.stores.base import FieldSpec, IndexSpec

if TYPE_CHECKING:
    from fastapi import UploadFile

log = get_logger(__name__)

# Fixed thin collection schema for the derived index.
_PRIMARY_FIELD = "id"
_VECTOR_FIELD = "vector"
_DOC_ID_FIELD = "doc_id"
_CHUNK_INDEX_FIELD = "chunk_index"
_SPARSE_FIELD = "sparse"

# Model tags recorded in the chunk_indexes registry.
_SPARSE_MODEL = "bm25-jieba"

# Document-level metadata keys honored from the ``metadata`` form field
# / parser output; unknown keys are ignored (the corpus has fixed columns).
_TITLE_FIELD = "title"
_AUTHOR_FIELD = "author"
_PAGE_COUNT_FIELD = "page_count"
_FILENAME_FIELD = "filename"

# Transitional v1->v2 collection migration (removed when the retrieval
# rewrite lands; kept so the HEAD retrieval route still resolves).
_TMP_COLLECTION = "ingest_migrate_tmp"
_MIGRATE_BATCH = 200


def _ingest_scalar_fields() -> list[FieldSpec]:
    """Scalar + sparse fields for the thin collection.

    The sparse field is declared alongside the scalars because
    ``create_collection`` treats only ``vector`` / extra vectors
    specially; it is still a vector field.
    """
    return [
        FieldSpec(name=_PRIMARY_FIELD, dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name=_DOC_ID_FIELD, dtype="varchar", max_length=64),
        FieldSpec(name=_CHUNK_INDEX_FIELD, dtype="int64"),
        FieldSpec(name=_SPARSE_FIELD, dtype="sparse_float_vector"),
    ]


def _ingest_indexes() -> list[IndexSpec]:
    """Dense HNSW index plus the sparse inverted (IP) index."""
    return [
        IndexSpec(
            field_name=_VECTOR_FIELD,
            metric_type="cosine",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200},
        ),
        IndexSpec(
            field_name=_SPARSE_FIELD,
            metric_type="ip",
            index_type="SPARSE_INVERTED_INDEX",
        ),
    ]


# The thin schema *is* the former "v2" schema; aliases kept for the
# migration routine below until the retrieval rewrite removes both.
_ingest_scalar_fields_v2 = _ingest_scalar_fields
_ingest_indexes_v2 = _ingest_indexes


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


# ---- helpers ----------------------------------------------------------


def _store_http_error(e: Exception, op: str) -> HTTPException:
    """Map a store-layer exception onto the canonical envelope."""
    if isinstance(e, DatabaseNotFound):
        return HTTPException(
            404,
            detail={
                "error": {
                    "code": "database_not_found",
                    "message": str(e),
                    "name": getattr(e, "name", None),
                }
            },
        )
    if isinstance(e, CollectionNotFound):
        return HTTPException(
            404,
            detail={
                "error": {
                    "code": "collection_not_found",
                    "message": str(e),
                }
            },
        )
    if isinstance(e, CollectionAlreadyExists):
        return HTTPException(
            409,
            detail={
                "error": {
                    "code": "collection_exists",
                    "message": str(e),
                }
            },
        )
    if isinstance(e, DimensionMismatch):
        return HTTPException(
            422,
            detail={
                "error": {
                    "code": "dimension_mismatch",
                    "message": str(e),
                    "expected": getattr(e, "expected", None),
                    "got": getattr(e, "got", None),
                }
            },
        )
    if isinstance(e, BackendError):
        return HTTPException(
            503,
            detail={
                "error": {
                    "code": "store_unavailable",
                    "message": str(e) or "vector store backend unavailable",
                    "op": op,
                    "exception_type": type(e).__name__,
                }
            },
        )
    if isinstance(e, StoreError):
        return HTTPException(
            422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": str(e) or "invalid request",
                    "op": op,
                    "exception_type": type(e).__name__,
                }
            },
        )
    return HTTPException(
        503,
        detail={
            "error": {
                "code": "store_unavailable",
                "message": str(e) or "vector store backend unavailable",
                "op": op,
                "exception_type": type(e).__name__,
            }
        },
    )


def _resolve_mime(filename: str | None, content_type: str | None) -> str:
    """Pick the MIME type to dispatch on. Mirrors ``/v1/parse``."""
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base:
            return base
    if filename:
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

# Async sink for progress events. The worker persists progress ticks
# (and, later, fans them out over SSE).
Emit = Callable[[dict[str, Any]], Awaitable[None]]

# Async gate consulted at every stage boundary; returns whether a
# cancellation was requested.
ShouldCancel = Callable[[], Awaitable[bool]]


class JobCancelled(Exception):
    """Raised internally when a stage boundary sees the cancel flag.

    Caught inside the pipeline: rollback runs, the job is marked
    ``cancelled``, and the exception re-raises so the worker knows the
    terminal state was already written.
    """


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
    check_embedder: bool = True,
) -> PreparedIngest:
    """Validate arguments and spool the upload into memory.

    Covers pipeline steps 0 and 1 — everything that can fail before any
    work starts. Callers run this synchronously so pre-flight failures
    come back as normal JSON error envelopes. ``check_embedder`` is
    turned off by the jobs submission route: the embedder may be loaded
    by the time the background worker runs, so embedder-state checks are
    deferred to execution time.
    """
    parser_settings = settings.parser

    # ---- 0. argument validation -------------------------------------
    # Raises 400 invalid_profile for unknown profile values.
    profile = _validate_profile(profile)

    if strategy not in list_strategies():
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "invalid_strategy",
                    "message": (
                        f"unknown chunking strategy {strategy!r}; expected "
                        f"one of {', '.join(list_strategies())}"
                    ),
                    "strategy": strategy,
                    "allowed": list_strategies(),
                }
            },
        )

    try:
        parsed_options = json.loads(chunk_options) if chunk_options else {}
        if not isinstance(parsed_options, dict):
            raise ValueError("chunk_options must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "invalid_chunk_options",
                    "message": f"chunk_options must be a JSON object: {e}",
                }
            },
        )

    # LLM-backed features need the external chat backend.
    if strategy == "llm" or add_context:
        if not is_llm_configured(settings):
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "llm_unavailable",
                        "message": (
                            "LLM is not configured; set VS_LLM__BASE_URL and "
                            "VS_LLM__MODEL first"
                        ),
                    }
                },
            )

    try:
        extra_metadata = json.loads(metadata) if metadata else {}
        if not isinstance(extra_metadata, dict):
            raise ValueError("metadata must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "invalid_metadata",
                    "message": f"metadata must be a JSON object: {e}",
                }
            },
        )

    if chunk_size < 1 or chunk_size > 8192:
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "invalid_chunk_size",
                    "message": f"chunk_size must be in [1, 8192], got {chunk_size}",
                    "chunk_size": chunk_size,
                }
            },
        )
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "invalid_chunk_overlap",
                    "message": (
                        f"chunk_overlap must be in [0, chunk_size), got "
                        f"{chunk_overlap} for chunk_size {chunk_size}"
                    ),
                    "chunk_overlap": chunk_overlap,
                    "chunk_size": chunk_size,
                }
            },
        )

    if check_embedder:
        if embedder is None:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "embedder_unavailable",
                        "message": (
                            f"text embedder is not loaded; "
                            f"call POST /v1/models/{embed_model}/load first"
                        ),
                        "model": embed_model,
                    }
                },
            )

        # Resolve the live embedder — match the inference routes' contract.
        if embedder.model_name != embed_model:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "embedder_unavailable",
                        "message": (
                            f"embedder {embed_model!r} is not loaded; the "
                            "lifespan step initialised a different model"
                        ),
                        "model": embed_model,
                        "loaded": embedder.model_name,
                    }
                },
            )

    # ---- 1. read upload ---------------------------------------------
    mime = _resolve_mime(file.filename, file.content_type)
    if mime not in (
        *DoclingParser.accepted_mime,
        *MarkdownParser.accepted_mime,
    ):
        raise HTTPException(
            415,
            detail={
                "error": {
                    "code": "unsupported_mime",
                    "message": f"unsupported MIME type {mime!r}",
                    "got": mime,
                }
            },
        )

    max_bytes = parser_settings.max_file_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            detail={
                "error": {
                    "code": "file_too_large",
                    "message": (
                        f"upload exceeds {parser_settings.max_file_size_mb} MB cap"
                    ),
                    "max_bytes": max_bytes,
                }
            },
        )
    if not data:
        raise HTTPException(
            400,
            detail={
                "error": {
                    "code": "empty_file",
                    "message": "uploaded file is empty",
                }
            },
        )

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
    repo: CorpusRepository,
    bm25: Any,
    embedder: Any,
    database: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    inference_timeout_seconds: float,
    emit: Emit,
    job_id: str,
    doc_id: str,
    should_cancel: ShouldCancel,
    report_parse_progress: bool = True,
) -> IngestResponse:
    """Parse -> chunk -> embed -> corpus write -> sparse -> upsert.

    ``job_id`` / ``doc_id`` come from the queued row the worker
    claimed; the job is NOT created or failed inside the pipeline —
    the caller decides retry vs terminal failure after rollback.

    Stage transitions persist via ``repo.mark_job``; ``emit`` carries
    stage/progress events to the worker sink. When
    ``report_parse_progress`` is set, per-page parser ticks are
    forwarded as ``progress`` events. The ``should_cancel`` gate is
    consulted at every stage boundary: a set flag rolls back, marks
    the job cancelled and raises :class:`JobCancelled`. Every other
    failure mode rolls back and raises :class:`HTTPException`.
    """
    data = prepared.data
    mime = prepared.mime
    profile = prepared.profile
    extra_metadata = prepared.extra_metadata
    loop = asyncio.get_running_loop()

    # Whether the document has been committed to the corpus — decides
    # how far rollback needs to reach.
    corpus_stored = False

    async def _rollback() -> None:
        """Undo whatever landed: corpus row, thin-index rows, artifacts."""
        if corpus_stored:
            try:
                await run_in_sqlite(repo.delete_document, doc_id)
            except Exception as e:  # noqa: BLE001 — original error wins
                log.warning(
                    "ingest_rollback_corpus_failed",
                    doc_id=doc_id,
                    error=str(e),
                )
            try:
                await run_in_store(
                    _delete_by_doc_id,
                    store,
                    database,
                    collection,
                    doc_id,
                )
            except Exception as e:  # noqa: BLE001
                log.error(
                    "ingest_rollback_milvus_failed",
                    doc_id=doc_id,
                    error=str(e),
                )
            # The failed doc may already have been folded into BM25 stats
            # (step 6) — invalidate every physical stats set the binding
            # points at so the next query refits over surviving leaves.
            try:
                refs = [collection]
                binding = await run_in_sqlite(
                    repo.get_binding, database, collection
                )
                if binding is not None:
                    refs.extend(
                        ref
                        for ref in (binding["active_ref"], binding["canary_ref"])
                        if ref and ref not in refs
                    )
                for ref in refs:
                    # Filesystem cleanup (stats file) — default executor.
                    await asyncio.to_thread(
                        bm25.discard, database, ref
                    )
            except Exception as e:  # noqa: BLE001 — original error wins
                log.warning(
                    "ingest_rollback_bm25_failed",
                    doc_id=doc_id,
                    error=str(e),
                )
        await asyncio.to_thread(_discard_artifacts, doc_id)

    async def _abort(exc: HTTPException) -> None:
        """Roll back whatever landed and re-raise the failure."""
        await _rollback()
        raise exc

    async def _check_cancel() -> None:
        """Stage-boundary gate; roll back and bail when cancelling."""
        if await should_cancel():
            await _rollback()
            await run_in_sqlite(repo.mark_cancelled, job_id)
            raise JobCancelled(job_id)

    try:
        # ---- 2. parse -----------------------------------------------
        await _check_cancel()
        # Persist the stage BEFORE emitting: the worker's emit sink
        # nudges SSE subscribers, who reload this row, so the stage must
        # already be committed when the nudge lands.
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_PARSING)
        )
        await emit({"type": "stage", "stage": "parse"})
        if mime in DoclingParser.accepted_mime:
            # Process-wide singleton: a fresh DocumentConverter per
            # request costs ~10-30s of model loading; the lifespan
            # warmup populates the same instance.
            parser = get_docling_parser()
        else:
            parser = MarkdownParser()

        # Page ticks fire on the Docling worker thread; hop onto the
        # event loop and turn each into a progress event. Counter +
        # drain loop (same pattern as /v1/parse/stream) guarantee
        # every tick is queued before the chunk-stage event.
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

            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(_fire()))

        on_progress = _on_parse_progress if report_parse_progress else None
        try:
            parsed = await parser.parse_bytes(
                data,
                mime,
                on_progress=on_progress,
                profile=profile,
                artifact_stem=doc_id,
            )
            while tick_flushed < tick_scheduled:
                await asyncio.sleep(0)
        except RuntimeError as e:
            log.warning("parser_failed", mime=mime, error=str(e))
            await _abort(
                HTTPException(
                    500,
                    detail={
                        "error": {
                            "code": "parser_failed",
                            "message": str(e) or "parser failed",
                            "mime": mime,
                            "exception_type": type(e).__name__,
                        }
                    },
                )
            )
        except Exception as e:
            log.warning("parser_unavailable", mime=mime, error=str(e))
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "parser_unavailable",
                            "message": str(e) or "parser backend unavailable",
                            "mime": mime,
                            "exception_type": type(e).__name__,
                        }
                    },
                )
            )

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

        # ---- 3. chunk -----------------------------------------------
        # All blocking work — tiktoken, sentence embeddings for
        # semantic, httpx chat calls for llm/contextualization — runs
        # in a worker thread so the event loop stays live.
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_CHUNKING)
        )
        await emit({"type": "stage", "stage": "chunk"})
        chunks = await run_in_model(
            _produce_chunks,
            markdown,
            prepared,
            settings,
            embedder,
            chunk_size,
            chunk_overlap,
        )

        if not chunks:
            # No content — finish the job empty and drop the images.
            await asyncio.to_thread(_discard_artifacts, doc_id)
            await run_in_sqlite(
                functools.partial(
                    repo.finish_job,
                    job_id,
                    chunk_count=0,
                    page_count=page_count,
                    tokens_used=0,
                ),
            )
            return IngestResponse(
                doc_id=doc_id,
                chunk_count=0,
                page_count=page_count,
                tokens_used=0,
            )

        # ---- 4. embed (dense) ---------------------------------------
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_EMBEDDING)
        )
        await emit({"type": "stage", "stage": "embed"})
        # Context prefixes (when enrichment ran) participate in embedding.
        texts = [embed_text(c) for c in chunks]
        try:
            vectors = await asyncio.wait_for(
                run_in_model(embedder.embed_documents, texts),
                timeout=inference_timeout_seconds,
            )
        except asyncio.TimeoutError:
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": (
                                f"embedder {embed_model!r} did not finish within "
                                f"{inference_timeout_seconds}s"
                            ),
                            "model": embed_model,
                        }
                    },
                )
            )
        except (EmbedderError, ModelNotLoaded) as e:
            log.warning("ingest_embed_failed", error=str(e))
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": str(e) or "embedder unavailable",
                            "model": embed_model,
                            "chunk_count": len(chunks),
                            "exception_type": type(e).__name__,
                        }
                    },
                )
            )
        except Exception as e:
            log.exception("ingest_embed_surprise")
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": str(e) or "embedder unavailable",
                            "model": embed_model,
                        }
                    },
                )
            )

        # ---- 5. commit content to the corpus ------------------------
        # Content lands BEFORE vectors: every derived row below is a
        # derivation of something the corpus now holds.
        now = time.time()
        document = DocumentRecord(
            doc_id=doc_id,
            database=database,
            collection=collection,
            filename=prepared.filename,
            mime=mime,
            content_hash=hashlib.sha256(data).hexdigest(),
            title=_as_optional_str(extra_metadata.get(_TITLE_FIELD)),
            author=_as_optional_str(extra_metadata.get(_AUTHOR_FIELD)),
            page_count=_as_optional_int(extra_metadata.get(_PAGE_COUNT_FIELD)),
            created_ts=now,
        )
        # Flat leaf rows — §1 has no section/document parents. Ids are
        # ``{doc_id}_{ordinal}`` and chunk_index is the source order.
        chunk_records = [
            ChunkRecord(
                chunk_id=f"{doc_id}_{i}",
                doc_id=doc_id,
                database=database,
                collection=collection,
                chunk_index=i,
                text=c.text,
                section_header=c.section_header,
                page_number=c.page_number,
                token_count=c.token_count,
                context=c.context,
                summary="",
                parent_id=None,
                level="chunk",
                char_start=None,
                char_end=None,
                created_ts=now,
            )
            for i, c in enumerate(chunks)
        ]
        await run_in_sqlite(repo.store_document, document, chunk_records)
        corpus_stored = True

        # ---- 6. client-side BM25 ------------------------------------
        # Refit stats over the corpus (now including the new chunks),
        # persist, then encode this document's sparse rows.
        try:
            await run_in_model(
                bm25.fit_for_ingest, repo, database, collection
            )
            sparse_vectors = await run_in_model(
                bm25.encode_documents,
                database,
                collection,
                [c.text for c in chunks],
            )
        except Exception as e:
            log.exception("ingest_sparse_encode_failed")
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "sparse_encode_failed",
                            "message": str(e) or "BM25 encoding failed",
                        }
                    },
                )
            )

        # ---- 7. ensure collection + upsert --------------------------
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_UPSERTING)
        )
        await emit({"type": "stage", "stage": "upsert"})
        try:
            await run_in_store(
                functools.partial(
                    _ensure_collection,
                    store,
                    database,
                    collection,
                    embedder.dim,
                ),
            )
        except (
            DatabaseNotFound,
            CollectionAlreadyExists,
            DimensionMismatch,
            StoreError,
            BackendError,
        ) as e:
            await _abort(_store_http_error(e, op="create_collection"))

        # Every row is a leaf in the flat schema.
        chunk_ids = [c.chunk_id for c in chunk_records]
        fields = [
            {
                _DOC_ID_FIELD: doc_id,
                _CHUNK_INDEX_FIELD: int(c.chunk_index),
            }
            for c in chunk_records
        ]

        try:
            await run_in_store(
                functools.partial(
                    store.upsert,
                    database,
                    collection,
                    _PRIMARY_FIELD,
                    _VECTOR_FIELD,
                    chunk_ids,
                    vectors,
                    fields,
                    sparse_vectors={_SPARSE_FIELD: sparse_vectors},
                ),
            )
        except (
            DatabaseNotFound,
            CollectionNotFound,
            DimensionMismatch,
            StoreError,
            BackendError,
        ) as upsert_err:
            await _abort(_store_http_error(upsert_err, op="upsert"))
        except Exception as upsert_err:
            log.exception("ingest_upsert_surprise")
            await _abort(
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "store_unavailable",
                            "message": str(upsert_err)
                            or "vector store backend unavailable",
                            "op": "upsert",
                            "exception_type": type(upsert_err).__name__,
                        }
                    },
                )
            )

        # ---- 8. index registry + job done ---------------------------
        registry_now = time.time()
        entries: list[IndexEntry] = []
        for chunk_id in chunk_ids:
            entries.append(
                IndexEntry(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    database=database,
                    collection=collection,
                    index_kind="dense",
                    model=embed_model,
                    index_ref=collection,
                    created_ts=registry_now,
                )
            )
            entries.append(
                IndexEntry(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    database=database,
                    collection=collection,
                    index_kind="sparse",
                    model=_SPARSE_MODEL,
                    index_ref=collection,
                    created_ts=registry_now,
                )
            )
        await run_in_sqlite(repo.record_indexes, entries)

        tokens_used = sum(c.token_count for c in chunks)
        await run_in_sqlite(
            functools.partial(
                repo.finish_job,
                job_id,
                chunk_count=len(chunks),
                page_count=page_count,
                tokens_used=tokens_used,
            ),
        )
        log.info(
            "ingest_done",
            doc_id=doc_id,
            job_id=job_id,
            collection=f"{database}/{collection}",
            chunk_count=len(chunks),
            tokens_used=tokens_used,
        )
        return IngestResponse(
            doc_id=doc_id,
            chunk_count=len(chunks),
            page_count=page_count,
            tokens_used=tokens_used,
        )
    except HTTPException:
        # Raised by _abort; propagate without re-aborting.
        raise
    except JobCancelled:
        # Already rolled back and marked cancelled inside the gate.
        raise
    except Exception as e:  # pragma: no cover — safety net
        log.exception("ingest_pipeline_failed")
        await _abort(
            HTTPException(
                500,
                detail={
                    "error": {
                        "code": "internal_error",
                        "message": str(e) or "internal error",
                    }
                },
            )
        )
        # _abort always raises; unreachable, keeps type checkers calm.
        raise


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


# ---- store-side helpers ----------------------------------------------


def ensure_database(store: Any, database: str) -> None:
    """Create ``database`` unless the store already lists it.

    A 409 from a racing worker is treated as success.
    """
    try:
        existing_dbs = store.list_databases()
    except (BackendError, StoreError):
        raise
    except Exception:
        existing_dbs = []

    if database in existing_dbs:
        return
    try:
        store.create_database(database)
    except Exception as e:
        if "already" not in str(e).lower():
            raise


def create_thin_collection(
    store: Any, database: str, collection: str, dim: int
) -> None:
    """Create one physical collection with the fixed thin schema.

    Unlike :func:`_ensure_collection` this is not idempotent — the
    caller (index rebuild) has dropped a stale same-named collection
    itself.
    """
    try:
        store.create_collection(
            database=database,
            name=collection,
            primary_field=_PRIMARY_FIELD,
            vector_field=FieldSpec(name=_VECTOR_FIELD, dtype="float_vector", dim=dim),
            scalar_fields=_ingest_scalar_fields(),
            indexes=_ingest_indexes(),
        )
    except CollectionAlreadyExists:
        # Lost the race; another worker created the same collection.
        return


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
    ensure_database(store, database)

    # Ensure collection exists with the thin schema.
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
            raise DimensionMismatch(
                f"collection {collection!r} has dim {info.dim} but "
                f"embedder outputs dim {dim}",
                expected=info.dim,
                got=dim,
            )
        return

    create_thin_collection(store, database, collection, dim)


def migrate_ingest_collection(store: Any, database: str, dim: int) -> dict:
    """Copy an ingest v1 collection into a v2 collection and swap them.

    Failure at any step drops the temporary collection and leaves the
    original in place (except the narrow window after the old
    collection is dropped and before rename completes).
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
        # contract.
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
    try:
        store._adapter._client.delete(collection, filter=expr)
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
