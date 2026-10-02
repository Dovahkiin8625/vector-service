"""Retrieval API: multi-channel retrieval, NDJSON stream, capabilities."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vector_service.chunking.llm_chunker import is_llm_configured
from vector_service.core.errors import (
    EmbedderError,
    RerankerError,
    RerankerNotLoaded,
    StoreError,
)
from vector_service.retrieval.pipeline import RetrievalPipeline
from vector_service.api.ingest import _store_http_error
from vector_service.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    to_result,
)

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _pipeline(request: Request) -> RetrievalPipeline:
    state = request.app.state
    return RetrievalPipeline(
        settings=state.settings,
        store=state.store,
        embedder=getattr(state, "embedder", None),
        reranker=getattr(state, "reranker", None),
        repo=getattr(state, "corpus", None),
        bm25=getattr(state, "bm25", None),
    )


def _error_event(status: int, code: str, message: str, exc: Exception) -> dict:
    """Terminal stream ``error`` event using the canonical error envelope."""
    return {
        "type": "error",
        "status": status,
        "error": {
            "code": code,
            "message": message,
            "exception_type": type(exc).__name__,
        },
    }


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, request: Request) -> RetrievalResponse:
    result = await _pipeline(request).retrieve(body)
    return to_result(result)


@router.post("/retrieval/stream")
async def retrieve_stream(body: RetrievalRequest, request: Request):
    """NDJSON stage events + terminal result.

    Pre-flight validation runs before the StreamingResponse is
    created, so bad params / missing models come back as ordinary JSON
    error envelopes rather than mid-stream errors.
    """
    pipeline = _pipeline(request)
    prevalidated = await pipeline.validate(body)

    async def event_stream():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def runner():
            try:
                result = await pipeline.retrieve(
                    body,
                    emit=lambda event: queue.put(event),
                    prevalidated=prevalidated,
                )
                queue.put_nowait({
                    "type": "result",
                    **to_result(result).model_dump(),
                })
            except HTTPException as e:
                queue.put_nowait({
                    "type": "error", "status": e.status_code, **e.detail,
                })
            except StoreError as e:
                # Mirror ingest._store_http_error (BackendError → 503
                # store_unavailable; NotFound family → 404; ...) rather
                # than re-coding the status table here.
                mapped = _store_http_error(e, op="retrieve")
                queue.put_nowait({
                    "type": "error", "status": mapped.status_code,
                    **mapped.detail,
                })
            except RerankerNotLoaded as e:
                queue.put_nowait(_error_event(
                    503, "reranker_not_loaded",
                    str(e) or "reranker not loaded", e,
                ))
            except RerankerError as e:
                queue.put_nowait(_error_event(
                    503, "reranker_error",
                    str(e) or "reranker failed", e,
                ))
            except EmbedderError as e:
                # MMR-stage embedding failure — same code as the ingest
                # pipeline's embed-stage envelope (embedder_unavailable).
                queue.put_nowait(_error_event(
                    503, "embedder_unavailable",
                    str(e) or "embedder unavailable", e,
                ))
            except Exception as e:  # noqa: BLE001 — last-resort terminal event
                queue.put_nowait(_error_event(500, "internal", str(e), e))
            finally:
                queue.put_nowait(sentinel)

        asyncio.create_task(runner())
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/retrieval/capabilities")
def capabilities(request: Request, database: str = "default") -> dict:
    """LLM availability for the retrieval panel (database is informational)."""
    _ = database
    return {
        "llm_configured": is_llm_configured(request.app.state.settings),
    }
