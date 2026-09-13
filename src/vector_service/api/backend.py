"""Backend escape hatch endpoints (debug-only for raw/call)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.schemas.management import BackendInfo

router = APIRouter(prefix="/backend", tags=["backend"])


@router.get(
    "/raw",
    response_model=BackendInfo,
    summary="Inspect active vector-store backend",
    description=(
        "Return backend name and connection URI. Useful for operators; "
        "not part of the supported public surface."
    ),
)
def backend_raw(request: Request):
    store = request.app.state.store
    info: dict[str, Any] = {
        "backend": store.backend_name,
        "uri": getattr(store, "uri", None),
        "native_methods": [
            "list_databases", "create_database", "drop_database", "database_info",
            "list_collections", "create_collection", "drop_collection", "collection_info",
            "upsert", "delete", "get", "search",
        ],
    }
    return {"backend": store.backend_name, "info": info}


@router.post(
    "/raw/call",
    include_in_schema=False,
    summary="Debug-only raw backend call (hidden from public docs)",
)
async def backend_raw_call(request: Request):
    settings = request.app.state.settings
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")
    body = await request.json()
    op = (body or {}).get("op")
    args = (body or {}).get("args") or {}
    store = request.app.state.store

    # Map a small set of read-only passthrough ops.
    try:
        if op == "list_databases":
            return {"result": store.list_databases()}
        if op == "list_collections":
            return {"result": store.list_collections(args["database"])}
        if op == "database_info":
            return {"result": store.database_info(args["name"]).__dict__}
        if op == "collection_info":
            return {"result": store.collection_info(args["database"], args["name"]).__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail=f"unsupported op {op!r}")
