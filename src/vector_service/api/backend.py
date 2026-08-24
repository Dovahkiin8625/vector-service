"""Backend escape hatch endpoints (debug-only for raw/call)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/backend", tags=["backend"])


@router.get("/raw")
def backend_raw(request: Request):
    store = request.app.state.store
    backend = store.backend
    info: dict[str, Any] = {
        "backend": store.backend_name,
        "uri": getattr(backend, "uri", None),
        "native_methods": [m for m in dir(backend) if not m.startswith("_")][:20],
    }
    return {"backend": store.backend_name, "info": info}


@router.post("/raw/call")
async def backend_raw_call(request: Request):
    settings = request.app.state.settings
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")
    body = await request.json()
    op = (body or {}).get("op")
    if op == "list_collections":
        return {"result": request.app.state.store.list_collections()}
    raise HTTPException(status_code=400, detail=f"unsupported op {op!r}")