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
  the lifespan writes once at boot. We fall back to "—" when the
  attribute is missing (e.g. in tests that bypass lifespan).
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from vector_service import __version__
from vector_service.core.system_metrics import system_snapshot
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
    embedder = getattr(request.app.state, "embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    multimodal_embedder = getattr(request.app.state, "multimodal_embedder", None)
    reranker = getattr(request.app.state, "reranker", None)

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


def _store_summary(request: Request) -> dict[str, Any]:
    """Connection state + database count for the configured vector store.

    ``status`` is one of:

    - ``"ok"`` — store reachable, databases listed successfully;
    - ``"down"`` — probe raised; the original exception text is kept
      under ``error`` so the operator can pin the failure without
      grepping server logs;
    - ``"not_attached"`` — no store on ``app.state`` (e.g. lifespan
      not yet started in a unit test).
    """
    store = getattr(request.app.state, "store", None)
    settings = getattr(request.app.state, "settings", None)
    backend = getattr(store, "backend_name", "unknown")
    summary: dict[str, Any] = {
        "backend": backend,
        "uri": getattr(store, "uri", None),
        "status": "not_attached",
        "databases": [],
        "database_count": 0,
        "error": None,
    }
    if store is None:
        return summary
    try:
        dbs = store.list_databases()
        summary["status"] = "ok"
        summary["databases"] = list(dbs)
        summary["database_count"] = len(dbs)
    except Exception as e:  # noqa: BLE001 — must fail open
        summary["status"] = "down"
        summary["error"] = "{}: {}".format(type(e).__name__, e)
    if settings is not None:
        summary["configured_backend"] = getattr(
            settings, "vector_store_backend", None
        )
    return summary


@router.get(
    "/system/status",
    summary="Service + machine overview",
    description=(
        "Aggregated dashboard payload: service version, embedding / store "
        "backends, uptime, registered + loaded model counts, vector-store "
        "connection status, database list, and live machine metrics "
        "(CPU, memory, disk, GPU). Fail-open — a degraded subsystem is "
        "reported as a `status: down` field, not a 5xx."
    ),
)
def get_system_status(request: Request) -> JSONResponse:
    settings = getattr(request.app.state, "settings", None)
    snapshot = system_snapshot()
    startup_ts = float(getattr(request.app.state, "startup_ts", time.time()))
    uptime = max(0.0, time.time() - startup_ts)

    payload = {
        "service": {
            "version": __version__,
            "embedding_backend": getattr(settings, "embedding_backend", None),
            "vector_store_backend": getattr(
                settings, "vector_store_backend", None
            ),
            "startup_ts": startup_ts,
            "uptime_seconds": uptime,
        },
        "models": _model_summary(request),
        "store": _store_summary(request),
        "system": snapshot,
    }
    return JSONResponse(content=payload)