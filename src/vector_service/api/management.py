"""Vector store management endpoints.

All routes are database-scoped:

    /v1/databases
    /v1/databases/{db}
    /v1/databases/{db}/collections
    /v1/databases/{db}/collections/{coll}
    /v1/databases/{db}/collections/{coll}/vectors...
    /v1/databases/{db}/collections/{coll}/search

Operations are implemented in terms of the universal :class:`VectorStore`
interface; the production backend talks directly to Milvus via
:mod:`pymilvus` (see ``vector_service.stores.milvus``).

Collection schema is **fully caller-defined**: scalar fields, vector
field, and indexes are all provided by the caller at create time, and
``upsert`` / ``get`` / ``search`` / ``delete`` carry the relevant
field names in their request bodies.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    EmbedderError,
    ImageDecodeError,
    ImageEmbedderError,
    ImageTooLarge,
    ModelNotLoaded,
    ModelNotLoadedForImages,
    StoreError,
    UnsupportedMime,
)
from vector_service.core.logging import get_logger
from vector_service.core.threadpools import (
    run_in_model,
    run_in_sqlite,
    run_in_store,
)
from vector_service.core.metrics import (
    STORE_COLLECTIONS,
    STORE_DATABASES,
    STORE_OP_DURATION_SECONDS,
    STORE_VECTORS_TOTAL,
)
from vector_service.embeddings.image_decoding import (
    decode_batch_or_422,
    decode_image,
    fail_envelope_422,
)
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.management import (
    BrowseRequest,
    BrowseResponse,
    CollectionInfoResponse,
    CreateCollectionRequest,
    CreateDatabaseRequest,
    DatabaseInfoResponse,
    DatabaseListResponse,
    DeleteVectorsRequest,
    FieldSummary,
    GetVectorsRequest,
    GetVectorsResponse,
    GetVectorItem,
    HitResponse,
    IndexManageRequest,
    IndexSummary,
    SearchRequest,
    SearchResponse,
    UpsertVectorsRequest,
)
from vector_service.stores.base import FieldSpec, IndexSpec

router = APIRouter(prefix="/v1", tags=["management"])
log = get_logger(__name__)


# ---- helpers ----

def _http_from_store_error(e: Exception) -> HTTPException:
    """Map a store-layer exception to the canonical HTTP error envelope.

    The ``exception_type`` extra key lets callers tell apart errors raised
    by this service (StoreError subclasses) from unexpected native
    backend failures, without having to parse the message string.

    Order matters: ``BackendError`` is a subclass of ``StoreError``, so it
    must be checked first to map to 503 (backend down) rather than 422
    (invalid request).
    """
    etype = type(e).__name__
    if isinstance(e, DatabaseNotFound):
        return HTTPException(
            404,
            detail={"error": {
                "code": "database_not_found",
                "message": str(e) or f"database {e.name!r} does not exist",
                "name": getattr(e, "name", None),
                "exception_type": etype,
            }},
        )
    if isinstance(e, DatabaseAlreadyExists):
        return HTTPException(
            409,
            detail={"error": {
                "code": "database_exists",
                "message": str(e) or f"database {e.name!r} already exists",
                "name": getattr(e, "name", None),
                "exception_type": etype,
            }},
        )
    if isinstance(e, CollectionNotFound):
        return HTTPException(404, detail={"error": {
            "code": "collection_not_found",
            "message": str(e) or "collection not found",
            "exception_type": etype,
        }})
    if isinstance(e, CollectionAlreadyExists):
        return HTTPException(409, detail={"error": {
            "code": "collection_exists",
            "message": str(e) or "collection already exists",
            "exception_type": etype,
        }})
    if isinstance(e, DimensionMismatch):
        return HTTPException(
            422,
            detail={"error": {
                "code": "dimension_mismatch",
                "message": str(e) or "vector dimension does not match the collection",
                "expected": e.expected, "got": e.got,
                "exception_type": etype,
            }},
        )
    if isinstance(e, BackendError):
        return HTTPException(503, detail={"error": {
            "code": "store_unavailable",
            "message": str(e) or "vector store backend unavailable",
            "exception_type": etype,
        }})
    if isinstance(e, StoreError):
        return HTTPException(422, detail={"error": {
            "code": "invalid_request",
            "message": str(e) or "invalid request",
            "exception_type": etype,
        }})
    return HTTPException(503, detail={"error": {
        "code": "store_unavailable",
        "message": str(e) or "vector store backend unavailable",
        "exception_type": etype,
    }})


def _timed(op: str, backend: str, database: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    status = "ok"
    try:
        return fn(*args, **kwargs)
    except Exception:
        status = "error"
        raise
    finally:
        STORE_OP_DURATION_SECONDS.labels(
            op=op, backend=backend, database=database, status=status
        ).observe(time.perf_counter() - t0)


async def _timed_async(op: str, backend: str, database: str, fn, *args, **kwargs):
    def _call() -> Any:
        return _timed(op, backend, database, fn, *args, **kwargs)

    return await run_in_store(_call)


def _backend(store) -> str:
    return store.backend_name


async def _chunk_ids_under_filter(
    store: Any, database: str, collection: str, primary_field: str,
    filter_expr: str, *, page_size: int = 200,
) -> list[str]:
    """Resolve a filter expression to matching row ids via the store.

    Pages ``store.browse`` over the matching rows. Used before a bulk
    delete on a corpus-backed collection so the same chunks can be
    removed from the system of record afterwards (post-delete nothing
    remains to resolve). Filters match individual chunks, not documents,
    so sibling chunks are deliberately left alone.
    """
    chunk_ids: list[str] = []
    offset = 0
    while True:
        rows = await run_in_store(
            functools.partial(
                store.browse,
                database, collection, primary_field,
                limit=page_size, offset=offset, filter_expr=filter_expr,
            ),
        )
        if not rows:
            break
        chunk_ids.extend(row["id"] for row in rows if "id" in row)
        if len(rows) < page_size:
            break
        offset += page_size
    return chunk_ids


# ---- image-embedder helpers ----

def _resolve_image_embedder_or_404(request: Request, model_id: str):
    """Validate ``model_id`` and return the loaded image embedder.

    Raises ``HTTPException`` directly:

    - 404 ``model_not_found`` if the id is not registered.
    - 503 ``image_embedder_unavailable`` if no embedder instance is
      attached to ``app.state`` (lifespan failed to load it) OR if
      the queried id does not match the embedder that is currently
      loaded (a registered-but-not-loaded backend).

    Returns the live ``ImageEmbedder`` instance on success.
    """
    try:
        get_image_embedder_class(model_id)
    except ImageEmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "model_not_found",
            "message": str(e) or f"unknown model {model_id!r}",
            "model": model_id,
            "exception_type": type(e).__name__,
        }})
    embedder = request.app.state.image_embedder
    # Constraint: the service only ever keeps one image embedder loaded
    # at a time. A request naming a *registered* but *not currently
    # loaded* model id must be told the embedder is unavailable rather
    # than silently falling through to whichever backend happens to be
    # in app.state. Mirrors the guard in ``api/models.py``.
    if embedder is None or embedder.model_name != model_id:
        raise HTTPException(status_code=503, detail={"error": {
            "code": "image_embedder_unavailable",
            "message": (
                f"image embedder {model_id!r} is not loaded; "
                "the lifespan step did not initialise it"
            ),
            "model": model_id,
        }})
    return embedder


def _decode_image_or_422(request: Request, b64: str, mime: str, *, index: int | None = None):
    """Decode a single base64 image, mapping errors to 422 envelopes.

    Used by both the upsert batch and the single search-image paths.
    Per-item failures during a batch decode are aggregated by the caller
    (``_decode_images_batch``); here we surface a single-image failure
    directly with a top-level code.
    """
    settings = request.app.state.settings.image_embedding
    try:
        return decode_image(
            b64, mime,
            max_bytes=settings.max_image_bytes,
            allowed_mime=set(settings.allowed_mime),
        )
    except UnsupportedMime as e:
        extras: dict = {"got": e.got, "allowed": e.allowed}
        code = "unsupported_mime"
        message = str(e) or code
    except ImageTooLarge as e:
        extras = {"got": e.got, "max": e.max}
        code = "image_too_large"
        message = str(e) or code
    except ImageDecodeError as e:
        extras = {}
        code = "image_decode_failed"
        message = str(e) or code
    if index is not None:
        extras["index"] = index
    raise HTTPException(status_code=422, detail={"error": {
        "code": code,
        "message": message,
        **extras,
    }})


def _decode_images_batch(request: Request, b64s: list[str], mimes: list[str]):
    """Decode a list of images; on any failure raise 422 with the same
    envelope shape used by ``POST /v1/image_embeddings``.
    """
    settings = request.app.state.settings.image_embedding
    decoded, failures = decode_batch_or_422(
        list(zip(b64s, mimes)),
        extract=lambda pair: (pair[0], pair[1]),
        max_bytes=settings.max_image_bytes,
        allowed_mime=set(settings.allowed_mime),
    )
    if failures:
        raise HTTPException(status_code=422, detail=fail_envelope_422(failures))
    return decoded


# ---- databases ----

@router.get(
    "/databases",
    response_model=DatabaseListResponse,
    summary="List databases",
    description="Return every database known to the connected vector store.",
)
async def list_databases(request: Request):
    store = request.app.state.store
    names = await _timed_async("db_list", _backend(store), "-", store.list_databases)
    STORE_DATABASES.labels(backend=_backend(store)).set(len(names))
    return DatabaseListResponse(databases=names)


@router.post(
    "/databases",
    status_code=201,
    response_model=DatabaseInfoResponse,
    responses={
        409: {"model": ErrorEnvelope, "description": "Database already exists."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Create a database",
)
async def create_database(body: CreateDatabaseRequest, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async(
            "db_create", _backend(store), body.name,
            store.create_database, body.name,
        )
    except (DatabaseAlreadyExists, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_DATABASES.labels(backend=_backend(store)).inc()
    return DatabaseInfoResponse(name=info.name, metadata=info.metadata)


@router.get(
    "/databases/{name}",
    response_model=DatabaseInfoResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Database does not exist."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Inspect a database",
)
async def get_database(name: str, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async("db_info", _backend(store), name, store.database_info, name)
    except (DatabaseNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return DatabaseInfoResponse(name=info.name, metadata=info.metadata)


@router.delete(
    "/databases/{name}",
    responses={
        404: {"model": ErrorEnvelope, "description": "Database does not exist."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Drop a database",
    description="Delete a database and every collection inside it. Cannot be undone.",
)
async def drop_database(name: str, request: Request):
    store = request.app.state.store
    repo = request.app.state.corpus
    bm25 = request.app.state.bm25
    # Capture logical collections + bindings before the drop so every
    # derived BM25 state file (including rebuilt physical refs) can be
    # discarded afterwards.
    corpus_collections = await run_in_sqlite(repo.corpus_collections, name)
    bindings = await run_in_sqlite(repo.list_bindings, name)
    refs = set(corpus_collections)
    for binding in bindings:
        refs.add(binding["active_ref"])
        if binding["canary_ref"]:
            refs.add(binding["canary_ref"])
    try:
        await _timed_async("db_drop", _backend(store), name, store.drop_database, name)
    except (DatabaseNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    orphaned = await run_in_sqlite(repo.delete_for_database, name)
    for digest in orphaned:
        await asyncio.to_thread(
            request.app.state.blob_store.delete, digest
        )
    for ref in refs:
        await asyncio.to_thread(bm25.discard, name, ref)
    STORE_DATABASES.labels(backend=_backend(store)).dec()
    return {"deleted": name}


# ---- collections ----

@router.get(
    "/databases/{db}/collections",
    summary="List collections in a database",
)
async def list_collections(db: str, request: Request):
    store = request.app.state.store
    try:
        names = await _timed_async(
            "coll_list", _backend(store), db, store.list_collections, db,
        )
    except (DatabaseNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=_backend(store), database=db).set(len(names))
    return {"collections": names}


@router.post(
    "/databases/{db}/collections",
    status_code=201,
    response_model=CollectionInfoResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Database does not exist."},
        409: {"model": ErrorEnvelope, "description": "Collection already exists."},
        422: {"model": ErrorEnvelope, "description": "Schema validation / dimension mismatch."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Create a collection",
    description=(
        "Create a collection with a fully caller-defined schema: scalar "
        "fields (one of which is the VARCHAR primary key), exactly one "
        "FLOAT_VECTOR field, and any indexes covering the vector field. "
        "Defaults to an HNSW index with the requested metric if "
        "`index_params` is omitted."
    ),
)
async def create_collection(db: str, body: CreateCollectionRequest, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async(
            "coll_create", _backend(store), db,
            store.create_collection, db, body.name,
            body.primary_field,
            FieldSpec(
                name=body.vector_field.name,
                dtype="float_vector",
                dim=body.vector_field.dim,
            ),
            [
                FieldSpec(
                    name=f.name,
                    dtype=f.dtype,
                    is_primary=f.is_primary,
                    max_length=f.max_length,
                    nullable=f.nullable,
                    default_value=f.default_value,
                )
                for f in body.scalar_fields
            ],
            [
                IndexSpec(
                    field_name=ip.field_name,
                    metric_type=ip.metric_type,
                    index_type=ip.index_type,
                    params=ip.params,
                )
                for ip in body.index_params
            ],
        )
    except (DatabaseNotFound, CollectionAlreadyExists, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=_backend(store), database=db).inc()
    # The store now returns the same ``fields`` / ``indexes`` payload that
    # ``GET .../collections/{name}`` would have produced, so the POST
    # response stays consistent with later detail calls.
    fields = [FieldSummary(**f) for f in (info.fields or [])]
    indexes = [IndexSummary(**i) for i in (info.indexes or [])]
    return CollectionInfoResponse(
        database=info.database,
        name=info.name,
        dim=info.dim,
        metric=info.metric,
        count=info.count,
        primary_field=info.primary_field,
        vector_field=info.vector_field,
        fields=fields,
        indexes=indexes,
        metadata=info.metadata,
    )


@router.delete(
    "/databases/{db}/collections/{name}",
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Drop a collection",
)
async def drop_collection(db: str, name: str, request: Request):
    store = request.app.state.store
    repo = request.app.state.corpus
    bm25 = request.app.state.bm25
    is_corpus_collection = await run_in_sqlite(
        repo.is_corpus_collection, db, name
    )
    # A corpus collection may live behind one or two physical names
    # (active + canary); without a binding physical == logical.
    binding = None
    if is_corpus_collection:
        binding = await run_in_sqlite(repo.get_binding, db, name)
    if binding is None:
        refs = [name]
    else:
        refs = [str(binding["active_ref"])]
        if binding["canary_ref"]:
            refs.append(str(binding["canary_ref"]))

    for ref_index, ref in enumerate(refs):
        try:
            await _timed_async(
                "coll_drop", _backend(store), db,
                store.drop_collection, db, ref,
            )
        except CollectionNotFound:
            # A canary binding may point at an already-gone ref; but the
            # primary (first) ref missing is a real 404 for the caller.
            if ref_index == 0:
                raise _http_from_store_error(
                    CollectionNotFound(ref)
                )
        except (DatabaseNotFound, StoreError, BackendError) as e:
            raise _http_from_store_error(e)
    if is_corpus_collection:
        orphaned = await run_in_sqlite(repo.delete_for_collection, db, name)
        for digest in orphaned:
            await asyncio.to_thread(
                request.app.state.blob_store.delete, digest
            )
        for ref in refs:
            await asyncio.to_thread(bm25.discard, db, ref)
    STORE_COLLECTIONS.labels(backend=_backend(store), database=db).dec()
    return {"deleted": name}


@router.get(
    "/databases/{db}/collections/{name}",
    response_model=CollectionInfoResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Inspect a collection",
)
async def get_collection(db: str, name: str, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async(
            "coll_info", _backend(store), db, store.collection_info, db, name,
        )
    except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return CollectionInfoResponse(
        database=info.database,
        name=info.name,
        dim=info.dim,
        metric=info.metric,
        count=info.count,
        primary_field=info.primary_field,
        vector_field=info.vector_field,
        fields=[FieldSummary(**f) for f in (info.fields or [])],
        indexes=[IndexSummary(**i) for i in (info.indexes or [])],
        metadata=info.metadata,
    )


# ---- index management ----

@router.post(
    "/databases/{db}/collections/{name}/index",
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Invalid index params (bad metric / unknown field)."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Create or rebuild a vector index",
    description=(
        "Create a new vector index on the collection's vector field, "
        "or replace the existing one if it already exists. Use this "
        "endpoint to tune ``HNSW`` ``M`` / ``efConstruction`` or "
        "switch between ``HNSW`` / ``IVF_FLAT`` / ``DISKANN`` without "
        "re-creating the collection."
    ),
)
async def create_collection_index(
    db: str, name: str, body: IndexManageRequest, request: Request,
):
    store = request.app.state.store
    try:
        await _timed_async(
            "index_create", _backend(store), db, store.create_index,
            db, name,
            field_name=body.field_name,
            metric_type=body.metric_type,
            index_type=body.index_type,
            params=body.params,
        )
    except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return {"rebuilt": body.field_name}


@router.delete(
    "/databases/{db}/collections/{name}/index",
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Invalid field name."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Drop a vector index",
    description=(
        "Drop the index on the named vector field. A missing index "
        "is treated as a no-op (idempotent) so the dashboard "
        "\"delete index\" button is safe to re-click."
    ),
)
async def drop_collection_index(
    db: str,
    name: str,
    field_name: str,
    request: Request,
):
    store = request.app.state.store
    try:
        await _timed_async(
            "index_drop", _backend(store), db, store.drop_index,
            db, name, field_name=field_name,
        )
    except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return {"dropped": field_name}


# ---- vectors ----

@router.put(
    "/databases/{db}/collections/{name}/vectors",
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Validation / dimension mismatch."},
        503: {"model": ErrorEnvelope, "description": "Embedder or vector store unavailable."},
    },
    summary="Upsert vectors",
    description=(
        "Insert or update rows. Provide `texts` to embed server-side, "
        "or pre-computed `vectors`. Per-row scalar field values go in "
        "`fields` (aligned with `ids`)."
    ),
)
async def upsert_vectors(db: str, name: str, body: UpsertVectorsRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder
    settings = request.app.state.settings

    op_label = "upsert"
    if body.texts is not None:
        if embedder is None:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": (
                    "text embedder is not loaded; "
                    "call POST /v1/models/{id}/load first"
                ),
                "text_count": len(body.texts),
            }})
        timeout_s = settings.inference_timeout_seconds
        try:
            try:
                vectors = await asyncio.wait_for(
                    run_in_model(embedder.embed_documents, body.texts),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                raise EmbedderError(
                    f"embedder did not finish within {timeout_s}s"
                )
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": str(e) or "embedder unavailable",
                "text_count": len(body.texts),
                "exception_type": type(e).__name__,
            }})
    elif body.images is not None:
        # Image branch: validate the model id, decode every payload
        # (returning 422 with ``failed_indices`` on any per-row failure),
        # then run the registered image embedder. Errors from the
        # embedder surface as 503 image_embedder_unavailable, mirroring
        # the text branch's handling of EmbedderError.
        op_label = "upsert_image"
        image_embedder = _resolve_image_embedder_or_404(request, body.model)
        decoded = _decode_images_batch(request, body.images, body.image_mimes)
        timeout_s = settings.inference_timeout_seconds
        try:
            try:
                vectors = await asyncio.wait_for(
                    run_in_model(image_embedder.embed_images, decoded),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                raise ImageEmbedderError(
                    f"image embedder did not finish within {timeout_s}s"
                )
        except (ImageEmbedderError, ModelNotLoadedForImages) as e:
            raise HTTPException(503, detail={"error": {
                "code": "image_embedder_unavailable",
                "message": str(e) or "image embedder unavailable",
                "model": body.model,
                "image_count": len(decoded),
                "exception_type": type(e).__name__,
            }})
    else:
        vectors = body.vectors or []

    if len(vectors) != len(body.ids):
        raise HTTPException(
            422,
            detail={"error": {
                "code": "shape_mismatch",
                "message": (
                    f"ids has {len(body.ids)} entries but vectors has "
                    f"{len(vectors)} entries; they must match"
                ),
                "ids_len": len(body.ids),
                "vectors_len": len(vectors),
            }},
        )

    try:
        await _timed_async(
            op_label, _backend(store), db, store.upsert,
            db, name, body.primary_field, body.vector_field,
            body.ids, vectors, body.fields,
        )
    except (DatabaseNotFound, CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(
        op=op_label, backend=_backend(store), database=db
    ).inc(len(body.ids))
    return {"upserted": len(body.ids)}


@router.post(
    "/databases/{db}/collections/{name}/vectors/delete",
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Validation (XOR ids vs filter_expr, or bad filter)."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Delete vectors by id or filter expression",
    description=(
        "Delete rows from a collection. Provide exactly one of:\n\n"
        "- ``ids``: explicit primary-key list.\n"
        "- ``filter_expr``: Milvus-native boolean expression; every "
        "matching row is removed (bulk delete).\n\n"
        "Returns the number of rows actually deleted."
    ),
)
async def delete_vectors(db: str, name: str, body: DeleteVectorsRequest, request: Request):
    store = request.app.state.store
    repo = request.app.state.corpus

    is_corpus_collection = await run_in_sqlite(
        repo.is_corpus_collection, db, name
    )
    # A corpus collection may be mirrored in two physical collections
    # (active + canary) — the same rows must leave each.
    binding = None
    if is_corpus_collection:
        binding = await run_in_sqlite(repo.get_binding, db, name)
    if binding is None:
        physical_refs = [name]
    else:
        physical_refs = [str(binding["active_ref"])]
        if binding["canary_ref"]:
            physical_refs.append(str(binding["canary_ref"]))

    # Filter deletes must resolve their rows BEFORE the rows are gone;
    # ids deletes already name the exact rows. Resolve per physical ref
    # and union: post-delete nothing remains to resolve from.
    ids_by_ref: dict[str, list[str]] = {}
    if is_corpus_collection:
        for ref in physical_refs:
            if body.ids is not None:
                ids_by_ref[ref] = body.ids
            else:
                ids_by_ref[ref] = await _chunk_ids_under_filter(
                    store, db, ref, body.primary_field, body.filter_expr,
                )

    deleted_total = 0
    for ref in physical_refs:
        try:
            if is_corpus_collection:
                deleted = await _timed_async(
                    "delete", _backend(store), db, store.delete,
                    db, ref, body.primary_field, ids_by_ref[ref],
                )
            else:
                deleted = await _timed_async(
                    "delete", _backend(store), db, store.delete,
                    db, ref, body.primary_field, body.ids,
                    filter_expr=body.filter_expr,
                )
        except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
            raise _http_from_store_error(e)
        deleted_total += int(deleted)

    if is_corpus_collection:
        corpus_chunk_ids = sorted({
            chunk_id
            for ids in ids_by_ref.values()
            for chunk_id in ids
        })
        orphaned = await run_in_sqlite(repo.delete_chunks, corpus_chunk_ids)
        for digest in orphaned:
            await asyncio.to_thread(
                request.app.state.blob_store.delete, digest
            )
        # Removing leaves invalidates BM25 statistics in every physical
        # ref — discard so the next query refits over surviving leaves.
        for ref in physical_refs:
            await asyncio.to_thread(
                request.app.state.bm25.discard, db, ref
            )

    STORE_VECTORS_TOTAL.labels(op="delete", backend=_backend(store), database=db).inc(deleted_total)
    return {"deleted": deleted_total}


@router.post(
    "/databases/{db}/collections/{name}/vectors/get",
    response_model=GetVectorsResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Fetch vectors by id",
)
async def get_vectors(db: str, name: str, body: GetVectorsRequest, request: Request):
    store = request.app.state.store
    try:
        items = await _timed_async(
            "get", _backend(store), db, store.get,
            db, name, body.primary_field, body.ids,
        )
    except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return GetVectorsResponse(items=[GetVectorItem(**it) for it in items])


@router.post(
    "/databases/{db}/collections/{name}/rows",
    response_model=BrowseResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Invalid output_fields or filter expression."},
        503: {"model": ErrorEnvelope, "description": "Vector store unavailable."},
    },
    summary="Browse collection rows (paginated list view)",
    description=(
        "List a slice of rows from a collection without specifying "
        "primary keys. Supports offset/limit pagination, an optional "
        "Milvus-native filter expression, and an optional list of "
        "scalar output fields. The vector field is never returned."
    ),
)
async def browse_rows(db: str, name: str, body: BrowseRequest, request: Request):
    store = request.app.state.store
    repo = request.app.state.corpus

    # Corpus-backed collections: content lives in SQLite, not the thin
    # derived index — page over the corpus regardless of Milvus state.
    # filter_expr / output_fields from the body are not applied on this
    # path (the corpus serves its fixed content-field vocabulary); the
    # dashboard sends neither.
    is_corpus_collection = await run_in_sqlite(
        repo.is_corpus_collection, db, name
    )
    if is_corpus_collection:
        rows, total = await run_in_sqlite(
            functools.partial(
                repo.browse, db, name,
                limit=body.limit, offset=body.offset,
            ),
        )
        total = int(total)
    else:
        # ``total`` comes from a live ``count_rows`` query: Milvus's
        # collection metadata keeps counting tombstoned rows until
        # compaction, so the pager would look unrefreshed right after a
        # delete. With ``filter_expr`` the count is restricted to the
        # same filter — the exact denominator for filtered pages. The
        # collection_info read runs first for its not-found/404 side
        # effect. Collection-not-found and invalid filters share the
        # browse call's error envelope.
        try:
            info = await _timed_async(
                "coll_info", _backend(store), db, store.collection_info, db, name,
            )
            total = int(info.count)
            total = int(await _timed_async(
                "count", _backend(store), db, store.count_rows,
                db, name, filter_expr=body.filter_expr,
            ))
            rows = await _timed_async(
                "browse", _backend(store), db, store.browse,
                db, name, body.primary_field,
                limit=body.limit,
                offset=body.offset,
                filter_expr=body.filter_expr,
                output_fields=body.output_fields,
            )
        except (DatabaseNotFound, CollectionNotFound, StoreError, BackendError) as e:
            raise _http_from_store_error(e)

    items = [GetVectorItem(**r) for r in rows]
    returned = len(items)
    has_more = body.offset + returned < total
    return BrowseResponse(
        items=items,
        total=total,
        limit=body.limit,
        offset=body.offset,
        returned=returned,
        has_more=has_more,
    )


@router.post(
    "/databases/{db}/collections/{name}/search",
    response_model=SearchResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Collection or database does not exist."},
        422: {"model": ErrorEnvelope, "description": "Validation / dimension mismatch."},
        503: {"model": ErrorEnvelope, "description": "Embedder or vector store unavailable."},
    },
    summary="Search similar vectors",
    description=(
        "Run k-NN search over a collection. Provide `query_text` to embed "
        "server-side, or pre-computed `query_vector`. Filter via "
        "`filter_expr` (Milvus-native boolean expression)."
    ),
)
async def search(db: str, name: str, body: SearchRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder
    settings = request.app.state.settings

    image_query = body.query_image is not None
    if body.query_text is not None:
        if embedder is None:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": (
                    "text embedder is not loaded; "
                    "call POST /v1/models/{id}/load first"
                ),
            }})
        timeout_s = settings.inference_timeout_seconds
        try:
            try:
                qvec = await asyncio.wait_for(
                    run_in_model(embedder.embed_query, body.query_text),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                raise EmbedderError(
                    f"embedder did not finish within {timeout_s}s"
                )
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {
                "code": "embedder_unavailable",
                "message": str(e) or "embedder unavailable",
                "exception_type": type(e).__name__,
            }})
    elif image_query:
        # Image branch: validate the model id, decode the single query
        # image, then run ``embed_query_image`` on the registered image
        # embedder. Per-image decode failures surface as 422 with the
        # same envelope shape as ``POST /v1/image_embeddings``; embedder
        # failures as 503 image_embedder_unavailable.
        image_embedder = _resolve_image_embedder_or_404(request, body.model)
        qimg = _decode_image_or_422(
            request, body.query_image, body.query_image_mime, index=0,
        )
        timeout_s = settings.inference_timeout_seconds
        try:
            try:
                qvec = await asyncio.wait_for(
                    run_in_model(image_embedder.embed_query_image, qimg),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                raise ImageEmbedderError(
                    f"image embedder did not finish within {timeout_s}s"
                )
        except (ImageEmbedderError, ModelNotLoadedForImages) as e:
            raise HTTPException(503, detail={"error": {
                "code": "image_embedder_unavailable",
                "message": str(e) or "image embedder unavailable",
                "model": body.model,
                "exception_type": type(e).__name__,
            }})
    else:
        qvec = body.query_vector or []

    search_op = "search_image" if image_query else "search"
    try:
        hits = await _timed_async(
            search_op, _backend(store), db, store.search,
            db, name, body.vector_field, qvec, body.top_k,
            body.filter_expr, body.output_fields,
        )
    except (DatabaseNotFound, CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    if image_query:
        # Distinct op label for image-query traffic — kept separate from
        # the (currently un-incremented) text/vector search path so
        # dashboards can filter image queries cleanly.
        STORE_VECTORS_TOTAL.labels(
            op="search_image", backend=_backend(store), database=db
        ).inc()
    return SearchResponse(
        hits=[HitResponse(id=h.id, score=h.score, fields=h.fields) for h in hits]
    )
