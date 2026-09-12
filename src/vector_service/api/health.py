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


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness: embedder loaded AND vector store reachable."""
    embedder = getattr(request.app.state, "embedder", None)
    store = getattr(request.app.state, "store", None)
    embedder_ok = _embedder_loaded(embedder)
    if store is None:
        body = (
            '{"status":"not_ready","store":"down","embedder":"'
            + ("loaded" if embedder_ok else "not_loaded")
            + '","reranker":"'
            + ("ready" if _reranker_loaded(request) else "not_loaded")
            + '"}'
        )
        return Response(content=body, status_code=503, media_type="application/json")
    # Probe the store with a cheap list call. Fail open if it errors so
    # the service still serves embeddings during transient store outages.
    store_ok = True
    try:
        store.list_databases()
    except Exception:
        store_ok = False

    overall_ok = embedder_ok and store_ok
    status = "ready" if overall_ok else (
        "degraded" if embedder_ok and not store_ok else "not_ready"
    )
    store_state = "ok" if store_ok else "down"
    embedder_state = "loaded" if embedder_ok else "not_loaded"
    reranker_state = "ready" if _reranker_loaded(request) else "not_loaded"
    code = 200 if overall_ok else 503
    body = (
        '{"status":"' + status
        + '","store":"' + store_state
        + '","embedder":"' + embedder_state
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