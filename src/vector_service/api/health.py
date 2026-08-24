"""Health, readiness, metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from vector_service.core.metrics import get_content_type, render_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    store = getattr(request.app.state, "store", None)
    if embedder is None or store is None:
        return Response(
            content='{"status":"not_ready"}',
            status_code=503,
            media_type="application/json",
        )
    return {"status": "ready"}


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        content=render_metrics(),
        media_type=get_content_type(),
    )