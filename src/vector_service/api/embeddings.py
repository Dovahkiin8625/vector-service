"""OpenAI-compatible embedding endpoint: ``POST /v1/embeddings``.

Model discovery (`GET /v1/models`, `GET /v1/models/{id}`) lives on its
own router in :mod:`vector_service.api.models` under the `model` tag,
since the registry is shared with the reranker subsystem.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import EmbedderError, ModelNotLoaded
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    EMBEDDING_DURATION_SECONDS,
    EMBEDDING_REQUESTS_TOTAL,
    EMBEDDING_TOKENS_TOTAL,
    MODEL_LOADED,
)
from vector_service.embeddings.registry import get_embedder_class
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.openai import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

router = APIRouter(prefix="/v1", tags=["embeddings"])
log = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    return max(1, -(-len(text) // 4))  # ceil(len/4)


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown model id."},
        422: {"model": ErrorEnvelope, "description": "Input validation failed."},
        503: {"model": ErrorEnvelope, "description": "Embedder unavailable."},
    },
    summary="Create embeddings",
    description=(
        "Embed one or more text inputs using the registered embedder. "
        "OpenAI-compatible: accepts a single string or a list of strings."
    ),
)
async def create_embeddings(body: EmbeddingRequest, request: Request):
    settings = request.app.state.settings
    embedder = request.app.state.embedder

    # 校验 model
    try:
        get_embedder_class(body.model)
    except EmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "model_not_found",
            "message": str(e) or f"unknown model {body.model!r}",
            "model": body.model,
        }})

    texts = [body.input] if isinstance(body.input, str) else list(body.input)

    # 限制校验
    if len(texts) > settings.embedding_max_texts_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": "too_many_texts",
                "message": (
                    f"got {len(texts)} texts but the per-request limit is "
                    f"{settings.embedding_max_texts_per_request}"
                ),
                "max": settings.embedding_max_texts_per_request,
                "got": len(texts),
            }},
        )
    for i, t in enumerate(texts):
        if len(t) > settings.embedding_max_chars_per_text:
            raise HTTPException(
                status_code=422,
                detail={"error": {
                    "code": "text_too_long",
                    "message": (
                        f"text[{i}] has {len(t)} chars; "
                        f"per-text limit is {settings.embedding_max_chars_per_text}"
                    ),
                    "max": settings.embedding_max_chars_per_text,
                    "got": len(t),
                    "index": i,
                }},
            )

    total_tokens = sum(_estimate_tokens(t) for t in texts)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"
    try:
        vectors = await loop.run_in_executor(None, embedder.embed_documents, texts)
    except (EmbedderError, ModelNotLoaded) as e:
        status = "error"
        raise HTTPException(status_code=503, detail={"error": {
            "code": "embedder_unavailable",
            "message": str(e) or f"embedder {body.model!r} unavailable",
            "model": body.model,
            "text_count": len(texts),
            "exception_type": type(e).__name__,
        }})
    finally:
        EMBEDDING_DURATION_SECONDS.labels(model=body.model, status=status).observe(time.perf_counter() - t0)
        EMBEDDING_REQUESTS_TOTAL.labels(model=body.model, status=status).inc()
        EMBEDDING_TOKENS_TOTAL.labels(model=body.model).inc(total_tokens)

    data = [
        EmbeddingData(index=i, embedding=v)
        for i, v in enumerate(vectors)
    ]
    log.info(
        "embedding_request",
        model=body.model,
        text_count=len(texts),
        tokens=total_tokens,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        status=status,
    )
    return EmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
