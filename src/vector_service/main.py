"""FastAPI application factory."""
from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vector_service import __version__
from vector_service.api.backend import router as backend_router
from vector_service.api.embeddings import router as embeddings_router
from vector_service.api.health import router as health_router
from vector_service.api.management import router as management_router
from vector_service.core.errors import VectorServiceError
from vector_service.core.lifespan import lifespan
from vector_service.core.logging import get_logger, request_id_var
from vector_service.core.middleware import RequestIDMiddleware

log = get_logger(__name__)


def _err(code: str, message: str, status: int, extra: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id_var.get(),
                "extra": extra or {},
            }
        },
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="vector-service",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(VectorServiceError)
    async def _vs_error_handler(request: Request, exc: VectorServiceError):
        # 业务异常的 HTTP 状态码由具体路由处理；
        # 这里兜底（不应被命中）
        log.error("unhandled_business_error", error=str(exc))
        return _err("internal", "internal error", 500)

    @app.exception_handler(HTTPException)
    async def _http_error_handler(request: Request, exc: HTTPException):
        # Routes raise HTTPException with detail={"error": {...}}; rewrap
        # into the canonical error envelope so clients see one shape.
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail and isinstance(detail["error"], dict):
            inner = detail["error"]
            extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
            return _err(inner.get("code", "error"), inner.get("message", str(exc.detail)),
                        exc.status_code, extras)
        return _err("error", str(detail), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        # exc.errors() may contain non-JSON-serializable objects in `ctx`;
        # coerce to strings so the response renders cleanly.
        safe_errors = []
        for e in exc.errors():
            safe = {k: v for k, v in e.items() if k != "ctx"}
            ctx = e.get("ctx")
            if isinstance(ctx, dict):
                safe["ctx"] = {k: str(v) for k, v in ctx.items()}
            safe_errors.append(safe)
        return _err("invalid_request", "validation error", 422, {"errors": safe_errors})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled_exception", error=str(exc))
        return _err("internal", "internal error", 500)

    app.include_router(health_router)
    app.include_router(embeddings_router)
    app.include_router(management_router)
    app.include_router(backend_router)

    return app


app = create_app()


def run() -> None:
    from vector_service.core.config import get_settings
    s = get_settings()
    uvicorn.run(
        "vector_service.main:app",
        host=s.host,
        port=s.port,
        workers=s.workers,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    run()