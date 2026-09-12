"""Rerank API: ``POST /v1/rerank`` and ``GET /v1/rerank/models``."""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.config import Settings
from vector_service.core.errors import RerankerError, RerankerNotLoaded
from vector_service.core.logging import get_logger, request_id_var
from vector_service.core.metrics import RERANK_DURATION_SECONDS, RERANK_REQUESTS_TOTAL
from vector_service.rerankers.base import Reranker
from vector_service.rerankers.registry import list_reranker_names
from vector_service.schemas.rerank import (
    RerankModelsResponse,
    RerankRequest,
    RerankResponse,
    RerankResultItem,
    RerankerInfo,
)

if TYPE_CHECKING:  # pragma: no cover
    pass

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["rerank"])


def _bad_request(code: str, message: str, extra: dict | None = None) -> HTTPException:
    """Build a 422 HTTPException with the canonical error envelope."""
    detail: dict = {"error": {"code": code, "message": message}}
    if extra:
        detail["error"].update(extra)
    return HTTPException(status_code=422, detail=detail)


@router.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest, request: Request) -> RerankResponse:
    settings: Settings = request.app.state.settings
    reranker: Reranker | None = getattr(request.app.state, "reranker", None)
    model = req.model or settings.reranker.backend

    # ---- registry check (404) --------------------------------------
    if model not in list_reranker_names():
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "model_not_found",
                               "message": f"reranker {model!r} is not registered",
                               "registered": list_reranker_names()}},
        )

    # ---- instance check (503) --------------------------------------
    if reranker is None or getattr(reranker, "_impl", None) is None:
        raise RerankerNotLoaded(f"reranker {model!r} is not loaded")

    # ---- input limits (422) ----------------------------------------
    r_settings = settings.reranker
    n_docs = len(req.documents)
    if n_docs > r_settings.max_documents_per_request:
        raise _bad_request(
            "too_many_documents",
            f"{n_docs} documents > max {r_settings.max_documents_per_request}",
            {"got": n_docs, "max": r_settings.max_documents_per_request},
        )
    for i, d in enumerate(req.documents):
        if len(d) > r_settings.max_chars_per_doc:
            raise _bad_request(
                "document_too_long",
                f"documents[{i}] length {len(d)} > max {r_settings.max_chars_per_doc}",
                {"index": i, "got": len(d), "max": r_settings.max_chars_per_doc},
            )
    if len(req.query) > r_settings.max_query_chars:
        raise _bad_request(
            "query_too_long",
            f"query length {len(req.query)} > max {r_settings.max_query_chars}",
            {"got": len(req.query), "max": r_settings.max_query_chars},
        )
    top_n = req.top_n if req.top_n is not None else r_settings.top_n_default
    if top_n > r_settings.max_top_n:
        raise _bad_request(
            "invalid_top_n",
            f"top_n {top_n} > max {r_settings.max_top_n}",
            {"got": top_n, "max": r_settings.max_top_n},
        )

    # ---- inference -------------------------------------------------
    RERANK_REQUESTS_TOTAL.labels(model, "received").inc()
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    try:
        hits = await loop.run_in_executor(
            None, reranker.rerank, req.query, req.documents, top_n
        )
    except RerankerError:
        RERANK_REQUESTS_TOTAL.labels(model, "error").inc()
        raise
    except Exception as exc:  # defensive wrap
        RERANK_REQUESTS_TOTAL.labels(model, "error").inc()
        log.warning(
            "rerank_failed",
            model=model,
            n_docs=n_docs,
            top_n=top_n,
            error_type=type(exc).__name__,
            error=str(exc),
            request_id=request_id_var.get(),
        )
        raise RerankerError(str(exc) or "rerank failed") from exc

    dt = time.perf_counter() - t0
    RERANK_DURATION_SECONDS.labels(model).observe(dt)
    RERANK_REQUESTS_TOTAL.labels(model, "ok").inc()
    log.info(
        "rerank_completed",
        model=model,
        n_docs=n_docs,
        top_n=top_n,
        latency_ms=int(dt * 1000),
        request_id=request_id_var.get(),
    )

    return RerankResponse(
        model=model,
        results=[RerankResultItem(index=h.index, score=h.score) for h in hits],
        request_id=request_id_var.get(),
    )


@router.get("/rerank/models", response_model=RerankModelsResponse)
async def list_rerank_models() -> RerankModelsResponse:
    """List registered reranker backends (does not require a loaded instance)."""
    return RerankModelsResponse(
        data=[RerankerInfo(name=n) for n in list_reranker_names()]
    )
