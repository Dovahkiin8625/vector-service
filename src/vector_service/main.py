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
from vector_service.api.image_embeddings import router as image_embeddings_router
from vector_service.api.multimodal_embeddings import router as multimodal_embeddings_router
from vector_service.api.management import router as management_router
from vector_service.api.models import router as models_router
from vector_service.api.dashboard import router as dashboard_router
from vector_service.api.rerank import router as rerank_router
from vector_service.api.system import router as system_router
from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    RerankerError,
    RerankerNotLoaded,
    VectorServiceError,
)
from vector_service.core.lifespan import lifespan
from vector_service.core.logging import get_logger, request_id_var
from vector_service.core.middleware import RequestIDMiddleware

log = get_logger(__name__)


def _err(
    code: str,
    message: str,
    status: int,
    extra: dict | None = None,
    *,
    exc: BaseException | None = None,
) -> JSONResponse:
    """Build a canonical error envelope.

    ``message`` is always returned verbatim so callers can see the actual
    reason. When an ``exc`` is supplied we attach ``exception_type`` (and,
    where available, ``exception_cause``) to ``extra`` so operators can
    pinpoint the failure without grepping server logs.
    """
    payload_extra: dict[str, Any] = dict(extra or {})
    if exc is not None:
        payload_extra.setdefault("exception_type", type(exc).__name__)
        cause = exc.__cause__ or exc.__context__
        if cause is not None and cause is not exc:
            payload_extra.setdefault(
                "exception_cause", f"{type(cause).__name__}: {cause}"
            )
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id_var.get(),
                "extra": payload_extra,
            }
        },
    )


OPENAPI_TAGS = [
    {
        "name": "health",
        "description": (
            "Liveness, readiness, and Prometheus metrics. Use `/healthz` "
            "for liveness probes and `/readyz` for readiness probes — "
            "`/readyz` returns 503 until both the embedder and the "
            "vector store are initialized."
        ),
    },
    {
        "name": "model",
        "description": (
            "Model registry endpoints shared between the embeddings and "
            "rerank subsystems. `GET /v1/models` lists every registered "
            "backend (embedders and rerankers) under a single shape — "
            "discriminate with the `type` field. "
            "`GET /v1/models/{id}` looks up a single backend by id."
        ),
    },
    {
        "name": "embeddings",
        "description": (
            "OpenAI-compatible text embedding endpoints under `/v1`. "
            "`POST /v1/embeddings` returns dense vectors for a registered "
            "embedder model."
        ),
    },
    {
        "name": "image_embeddings",
        "description": (
            "Image vectorization under `/v1`. `POST /v1/image_embeddings` "
            "returns dense vectors for a registered image embedder model "
            "from base64-encoded image inputs."
        ),
    },
    {
        "name": "multimodal_embeddings",
        "description": (
            "Cross-modal (text + image) vectorization under `/v1`. "
            "`POST /v1/multimodal_embeddings` accepts a mixed list of Chinese "
            "texts and base64-encoded images; both land in the same shared "
            "vector space — useful for text-search-image and "
            "image-search-text retrieval."
        ),
    },
    {
        "name": "rerank",
        "description": (
            "Cross-encoder reranking. `POST /v1/rerank` takes a query "
            "and a list of documents and returns documents reordered "
            "by relevance. Registered reranker backends are listed via "
            "`GET /v1/models` (filter by `type=reranker`)."
        ),
    },
    {
        "name": "management",
        "description": (
            "Multi-database vector-store CRUD: create / drop / inspect "
            "databases, then collections within a database, upsert / "
            "delete / fetch vectors, and run similarity search. Backed "
            "directly by Milvus. All errors share the same envelope: "
            "see `ErrorEnvelope`."
        ),
    },
    {
        "name": "backend",
        "description": (
            "Backend introspection escape hatch (debug builds only). "
            "`POST /backend/raw/call` is hidden from the public schema."
        ),
    },
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="vector-service",
        version=__version__,
        description=(
            "Production-ready FastAPI vector service.\n\n"
            "- **OpenAI-compatible embeddings** under `/v1`.\n"
            "- **Multi-database vector-store management** under "
            "`/v1/databases` and `/v1/databases/{db}/collections`.\n"
            "- **Direct Milvus backend** via `pymilvus` — no proxy hop.\n"
            "- **Observability**: Prometheus metrics at `/metrics` plus "
            "structured JSON logs.\n\n"
            "Browse the interactive reference at [`/docs`](#) (Swagger UI) "
            "or [`/redoc`](#) (ReDoc), or fetch the raw OpenAPI document "
            "from `/openapi.json`."
        ),
        contact={
            "name": "vector-service",
            "url": "https://example.invalid/vector-service",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {"url": "http://localhost:8080", "description": "Local development"},
            {"url": "/", "description": "Current host"},
        ],
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(DatabaseNotFound)
    async def _db_not_found(request: Request, exc: DatabaseNotFound):
        return _err(
            "database_not_found", str(exc), 404, {"name": getattr(exc, "name", None)}
        )

    @app.exception_handler(DatabaseAlreadyExists)
    async def _db_exists(request: Request, exc: DatabaseAlreadyExists):
        return _err(
            "database_exists", str(exc), 409, {"name": getattr(exc, "name", None)}
        )

    @app.exception_handler(CollectionNotFound)
    async def _coll_not_found(request: Request, exc: CollectionNotFound):
        return _err("collection_not_found", str(exc), 404)

    @app.exception_handler(CollectionAlreadyExists)
    async def _coll_exists(request: Request, exc: CollectionAlreadyExists):
        return _err("collection_exists", str(exc), 409)

    @app.exception_handler(DimensionMismatch)
    async def _dim_mismatch(request: Request, exc: DimensionMismatch):
        return _err(
            "dimension_mismatch",
            str(exc),
            422,
            {"expected": exc.expected, "got": exc.got},
        )

    @app.exception_handler(BackendError)
    async def _backend_err(request: Request, exc: BackendError):
        log.warning("backend_error", error=str(exc))
        return _err(
            "store_unavailable",
            str(exc) or "vector store backend unavailable",
            503,
            exc=exc,
        )

    @app.exception_handler(RerankerNotLoaded)
    async def _rerank_not_loaded(request: Request, exc: RerankerNotLoaded):
        return _err(
            "reranker_not_loaded",
            str(exc) or "reranker not loaded",
            503,
            exc=exc,
        )

    @app.exception_handler(RerankerError)
    async def _rerank_error(request: Request, exc: RerankerError):
        log.warning("reranker_error", error=str(exc))
        return _err(
            "reranker_error",
            str(exc) or "reranker failed",
            503,
            exc=exc,
        )

    @app.exception_handler(VectorServiceError)
    async def _vs_error_handler(request: Request, exc: VectorServiceError):
        # 兜底：未被上面具体 handler 命中的业务异常
        log.error("unhandled_business_error", error=str(exc))
        return _err("internal", str(exc) or "internal error", 500, exc=exc)

    @app.exception_handler(HTTPException)
    async def _http_error_handler(request: Request, exc: HTTPException):
        # Routes raise HTTPException with detail={"error": {...}}; rewrap
        # into the canonical error envelope so clients see one shape.
        detail = exc.detail
        if (
            isinstance(detail, dict)
            and "error" in detail
            and isinstance(detail["error"], dict)
        ):
            inner = detail["error"]
            extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
            return _err(
                inner.get("code", "error"),
                inner.get("message", str(exc.detail)),
                exc.status_code,
                extras,
            )
        return _err("error", str(detail), exc.status_code, exc=exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        # exc.errors() may contain non-JSON-serializable objects in `ctx`;
        # coerce to strings so the response renders cleanly.
        safe_errors = []
        first_msg = ""
        for e in exc.errors():
            safe = {k: v for k, v in e.items() if k != "ctx"}
            ctx = e.get("ctx")
            if isinstance(ctx, dict):
                safe["ctx"] = {k: str(v) for k, v in ctx.items()}
            safe_errors.append(safe)
            if not first_msg:
                loc = ".".join(str(p) for p in e.get("loc", []) if p != "body")
                err_type = e.get("type", "invalid")
                raw_msg = e.get("msg", "")
                first_msg = f"{loc}: {raw_msg}" if loc else f"{err_type}: {raw_msg}"
        return _err(
            "invalid_request",
            first_msg or "validation error",
            422,
            {"errors": safe_errors},
            exc=exc,
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled_exception", error=str(exc))
        return _err("internal", str(exc) or "internal error", 500, exc=exc)

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(embeddings_router)
    app.include_router(rerank_router)
    app.include_router(system_router)
    app.include_router(management_router)
    app.include_router(backend_router)
    app.include_router(dashboard_router)
    app.include_router(image_embeddings_router)
    app.include_router(multimodal_embeddings_router)

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
