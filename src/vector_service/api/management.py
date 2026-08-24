"""Vector store management endpoints."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
)
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    STORE_COLLECTIONS,
    STORE_OP_DURATION_SECONDS,
    STORE_VECTORS_TOTAL,
)
from vector_service.schemas.management import (
    CollectionInfoResponse,
    CreateCollectionRequest,
    DeleteVectorsRequest,
    GetVectorItem,
    GetVectorsRequest,
    GetVectorsResponse,
    SearchRequest,
    SearchResponse,
    HitResponse,
    UpsertVectorsRequest,
)

router = APIRouter(tags=["management"])
log = get_logger(__name__)


# ---- helpers ----

def _http_from_store_error(e: Exception) -> HTTPException:
    if isinstance(e, CollectionNotFound):
        return HTTPException(404, detail={"error": {"code": "collection_not_found", "message": str(e)}})
    if isinstance(e, CollectionAlreadyExists):
        return HTTPException(409, detail={"error": {"code": "collection_exists", "message": str(e)}})
    if isinstance(e, DimensionMismatch):
        return HTTPException(422, detail={"error": {
            "code": "dimension_mismatch",
            "message": str(e),
            "expected": e.expected, "got": e.got,
        }})
    return HTTPException(503, detail={"error": {"code": "store_unavailable", "message": str(e)}})


def _timed(op: str, backend: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    status = "ok"
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        status = "error"
        raise
    finally:
        STORE_OP_DURATION_SECONDS.labels(op=op, backend=backend, status=status).observe(time.perf_counter() - t0)


async def _timed_async(op: str, backend: str, fn, *args, **kwargs):
    loop = asyncio.get_running_loop()

    def _call() -> Any:
        return _timed(op, backend, fn, *args, **kwargs)

    return await loop.run_in_executor(None, _call)


# ---- collections ----

@router.get("/collections")
async def list_collections(request: Request):
    store = request.app.state.store
    names = await _timed_async("list", store.backend_name, store.list_collections)
    STORE_COLLECTIONS.labels(backend=store.backend_name).set(len(names))
    return {"collections": names}


@router.post("/collections", status_code=201)
async def create_collection(body: CreateCollectionRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder
    dim = body.dim if body.dim is not None else embedder.dim
    try:
        await _timed_async("create", store.backend_name, store.create_collection,
                           body.name, dim, metric=body.metric, **body.backend_opts)
    except (CollectionAlreadyExists, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=store.backend_name).inc()
    return {"name": body.name, "dim": dim, "metric": body.metric}


@router.delete("/collections/{name}")
async def drop_collection(name: str, request: Request):
    store = request.app.state.store
    try:
        await _timed_async("drop", store.backend_name, store.drop_collection, name)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=store.backend_name).dec()
    return {"deleted": name}


@router.get("/collections/{name}", response_model=CollectionInfoResponse)
async def get_collection(name: str, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async("info", store.backend_name, store.collection_info, name)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return CollectionInfoResponse(**info)


# ---- vectors ----

@router.put("/collections/{name}/vectors")
async def upsert_vectors(name: str, body: UpsertVectorsRequest, request: Request):
    settings = request.app.state.settings
    store = request.app.state.store
    embedder = request.app.state.embedder

    if body.texts is not None:
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(None, embedder.embed_documents, body.texts)
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
    else:
        vectors = body.embeddings or []

    if len(vectors) != len(body.ids):
        raise HTTPException(422, detail={"error": {"code": "shape_mismatch", "message": "vectors/ids length"}})

    try:
        await _timed_async("upsert", store.backend_name, store.upsert,
                           name, body.ids, vectors, body.metadatas)
    except (CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(op="upsert", backend=store.backend_name).inc(len(body.ids))
    return {"upserted": len(body.ids)}


@router.post("/collections/{name}/vectors/delete")
async def delete_vectors(name: str, body: DeleteVectorsRequest, request: Request):
    store = request.app.state.store
    try:
        await _timed_async("delete", store.backend_name, store.delete, name, body.ids)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(op="delete", backend=store.backend_name).inc(len(body.ids))
    return {"deleted": len(body.ids)}


@router.post("/collections/{name}/vectors/get", response_model=GetVectorsResponse)
async def get_vectors(name: str, body: GetVectorsRequest, request: Request):
    store = request.app.state.store
    try:
        items = await _timed_async("get", store.backend_name, store.get, name, body.ids)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return GetVectorsResponse(items=[GetVectorItem(**it) for it in items])


@router.post("/collections/{name}/search", response_model=SearchResponse)
async def search(name: str, body: SearchRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder

    if body.query_text is not None:
        loop = asyncio.get_running_loop()
        try:
            qvec = (await loop.run_in_executor(None, embedder.embed_query, body.query_text))
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
    else:
        qvec = body.query_embedding or []

    try:
        hits = await _timed_async("search", store.backend_name, store.search,
                                  name, qvec, body.top_k, body.filter)
    except (CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return SearchResponse(hits=[HitResponse(id=h.id, score=h.score, metadata=h.metadata) for h in hits])