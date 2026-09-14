"""Health, readiness, metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from vector_service.core.metrics import get_content_type, render_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    """Process liveness — independent of any backend."""
    return {"status": "ok"}


def _embedder_loaded(embedder) -> bool:
    """True iff the embedder has finished its eager load().

    Looks for the `_impl` attribute set by concrete embedders after a
    successful load. Falls back to True for embedders that don't expose
    an internal handle (e.g. lightweight fakes in tests).
    """
    if embedder is None:
        return False
    if hasattr(embedder, "_impl"):
        return embedder._impl is not None
    return True


def _reranker_loaded(request: Request) -> bool:
    """True iff the reranker has finished its eager load().

    Looks for the `_impl` attribute set by concrete rerankers after a
    successful load. Diagnostic only — does NOT gate `/readyz` status.
    """
    reranker = getattr(request.app.state, "reranker", None)
    return reranker is not None and getattr(reranker, "_impl", None) is not None


def _image_embedder_loaded(request: Request) -> bool:
    """True iff the image embedder has finished its eager load().

    Looks for the `_impl` / `_model` attributes set by concrete image
    embedders after a successful load. Falls back to True for embedders
    that don't expose an internal handle (e.g. lightweight fakes).
    """
    img = getattr(request.app.state, "image_embedder", None)
    if img is None:
        return False
    for attr in ("_impl", "_model"):
        if hasattr(img, attr):
            return getattr(img, attr) is not None
    return True


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: process alive AND vector store reachable.

    Models are loaded on demand (default ``auto_load=false``); a fresh
    process is still ``ready`` even when every family is unloaded —
    inference will return 503 ``*_unavailable`` until an operator
    calls ``POST /v1/models/{id}/load``. The response body keeps
    per-family load status for observability so a single probe
    surfaces both the store health and the model roster state.
    """
    store = getattr(request.app.state, "store", None)
    embedder = getattr(request.app.state, "embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)

    # Probe the store with a cheap list call. Fail open if it errors
    # so /readyz stays informative during transient outages.
    store_ok = False
    if store is not None:
        try:
            store.list_databases()
            store_ok = True
        except Exception:
            store_ok = False

    embedder_state = "loaded" if _embedder_loaded(embedder) else "not_loaded"
    image_state = "loaded" if _image_embedder_loaded(request) else "not_loaded"
    reranker_state = "ready" if _reranker_loaded(request) else "not_loaded"
    mm_state = (
        "loaded"
        if getattr(request.app.state, "multimodal_embedder", None)
        and getattr(
            getattr(request.app.state, "multimodal_embedder", None), "_model", None
        ) is not None
        else "not_loaded"
    )

    if store is None:
        status = "not_ready"
        code = 503
    elif store_ok:
        status = "ready"
        code = 200
    else:
        status = "degraded"
        code = 503

    body = (
        '{"status":"' + status
        + '","store":"' + ("ok" if store_ok else "down")
        + '","embedder":"' + embedder_state
        + '","image_embedder":"' + image_state
        + '","multimodal_embedder":"' + mm_state
        + '","reranker":"' + reranker_state
        + '"}'
    )
    return Response(content=body, status_code=code, media_type="application/json")


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        content=render_metrics(),
        media_type=get_content_type(),
    )