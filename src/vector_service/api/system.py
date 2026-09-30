"""Service-level overview endpoint: ``GET /v1/system/status``.

The dashboard's home panel reads from this endpoint to render the
service + machine summary (registered models, loaded models per
family, vector-store connectivity, db count, CPU / memory / GPU
sensors). The same endpoint is useful to humans via ``curl`` and to
external probes — the schema is stable.

Design notes:

- All reads come from ``app.state`` and the store; nothing here holds
  its own state, so the endpoint is safe to poll at high rates.
- The store probe is fail-open: an unreachable Milvus must not 5xx the
  status endpoint, otherwise a degraded cluster can't even diagnose
  itself. We surface ``store.status="down"`` and the exception text
  under ``store.error`` so the UI can render a red banner.
- GPU / CPU / memory are delegated to
  :mod:`vector_service.core.system_metrics` so both the HTTP route and
  any future in-process consumer share one implementation.
- ``uptime_seconds`` is measured from ``app.state.startup_ts`` which
  the lifespan writes once at boot.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vector_service import __version__
from vector_service.core.parser_status import parser_status
from vector_service.core.system_metrics import system_snapshot
from vector_service.core.threadpools import run_in_store
from vector_service.embeddings.image_registry import list_image_embedder_names
from vector_service.embeddings.multimodal_registry import (
    list_multimodal_embedder_names,
)
from vector_service.embeddings.registry import list_embedder_names
from vector_service.rerankers.registry import list_reranker_names

router = APIRouter(prefix="/v1", tags=["system"])


def _model_summary(request: Request) -> dict[str, Any]:
    """Per-family registered + loaded summary.

    ``loaded`` carries the model_name currently held on ``app.state``
    (or ``None`` if the slot is empty); ``dimensions`` is included so
    the dashboard can render the dim-dots signature without a second
    round-trip to ``GET /v1/models``.
    """
    embedder = request.app.state.embedder
    image_embedder = request.app.state.image_embedder
    multimodal_embedder = request.app.state.multimodal_embedder
    reranker = request.app.state.reranker

    def _state(instance: Any) -> dict[str, Any]:
        if instance is None:
            return {"loaded": None, "dimensions": None}
        return {
            "loaded": getattr(instance, "model_name", None),
            "dimensions": getattr(instance, "dim", None),
        }

    families = {
        "embedder": (list_embedder_names(), _state(embedder)),
        "image_embedder": (list_image_embedder_names(), _state(image_embedder)),
        "multimodal_embedder": (
            list_multimodal_embedder_names(),
            _state(multimodal_embedder),
        ),
        "reranker": (list_reranker_names(), _state(reranker)),
    }
    return {
        family: {
            "registered": registered,
            "loaded_id": state["loaded"],
            "dimensions": state["dimensions"],
        }
        for family, (registered, state) in families.items()
    }


def _parser_summary(request: Request) -> dict[str, Any]:
    """Per-profile Docling parser status (warm profiles + components).

    Prefer the parser the lifespan warmed on ``app.state``; fall back
    to the process singleton. Fail-open like the store probe.
    """
    parser = request.app.state.parser
    if parser is None:
        from vector_service.parsers.docling_parser import get_docling_parser

        parser = get_docling_parser()
    try:
        return parser_status(parser)
    except Exception as e:  # noqa: BLE001 — status must stay fail-open
        return {"backend": "docling", "status": "down", "error": str(e)}


def _store_summary(request: Request) -> dict[str, Any]:
    """Connection state + database count for the configured vector store.

    ``status`` is one of:

    - ``"ok"`` — store reachable, databases listed successfully;
    - ``"down"`` — probe raised; the original exception text is kept
      under ``error`` so the operator can pin the failure without
      grepping server logs.
    """
    store = request.app.state.store
    settings = request.app.state.settings
    summary: dict[str, Any] = {
        "backend": store.backend_name,
        "uri": store.uri,
        "status": "ok",
        "databases": [],
        "database_count": 0,
        "error": None,
        "configured_backend": settings.vector_store_backend,
    }
    try:
        dbs = store.list_databases()
        summary["databases"] = list(dbs)
        summary["database_count"] = len(dbs)
    except Exception as e:  # noqa: BLE001 — must fail open
        summary["status"] = "down"
        summary["error"] = "{}: {}".format(type(e).__name__, e)
    return summary


def _thread_pool_summary(request: Request) -> dict[str, Any] | None:
    """Per-pool workers/admission/inflight counters from the lifespan."""
    thread_pools = getattr(request.app.state, "thread_pools", None)
    if thread_pools is None:
        return None
    return thread_pools.describe()


@router.get(
    "/system/status",
    summary="Service + machine overview",
    description=(
        "Aggregated dashboard payload: service version, embedding / store "
        "backends, uptime, registered + loaded model counts, per-profile "
        "Docling parser status, vector-store connection status, database "
        "list, isolated thread-pool counters, and live machine metrics "
        "(CPU, memory, disk, GPU). "
        "Fail-open — a degraded subsystem is reported as a `status: down` "
        "field, not a 5xx."
    ),
)
async def get_system_status(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    snapshot = system_snapshot()
    startup_ts = float(request.app.state.startup_ts)
    uptime = max(0.0, time.time() - startup_ts)

    payload = {
        "service": {
            "version": __version__,
            "embedding_backend": settings.embedding_backend,
            "vector_store_backend": settings.vector_store_backend,
            "startup_ts": startup_ts,
            "uptime_seconds": uptime,
        },
        "models": _model_summary(request),
        "parser": _parser_summary(request),
        "store": await run_in_store(_store_summary, request),
        "thread_pools": _thread_pool_summary(request),
        "system": snapshot,
    }
    return JSONResponse(content=payload)