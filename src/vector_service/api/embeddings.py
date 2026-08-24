"""OpenAI-compatible embedding endpoints."""
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
from vector_service.embeddings.registry import (
    EMBEDDER_REGISTRY,
    get_embedder_class,
    list_embedder_names,
)
from vector_service.schemas.openai import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    Model,
    ModelList,
)

router = APIRouter(prefix="/v1", tags=["embeddings"])
log = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    return max(1, -(-len(text) // 4))  # ceil(len/4)


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(body: EmbeddingRequest, request: Request):
    settings = request.app.state.settings
    embedder = request.app.state.embedder

    # 校验 model
    try:
        get_embedder_class(body.model)
    except EmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {"code": "model_not_found", "message": str(e)}})

    texts = [body.input] if isinstance(body.input, str) else list(body.input)

    # 限制校验
    if len(texts) > settings.embedding_max_texts_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "too_many_texts",
                              "message": f"max {settings.embedding_max_texts_per_request}",
                              "max": settings.embedding_max_texts_per_request}},
        )
    for i, t in enumerate(texts):
        if len(t) > settings.embedding_max_chars_per_text:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "text_too_long",
                                  "message": f"max {settings.embedding_max_chars_per_text} chars",
                                  "index": i}},
            )

    total_tokens = sum(_estimate_tokens(t) for t in texts)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"
    try:
        vectors = await loop.run_in_executor(None, embedder.embed_documents, texts)
    except (EmbedderError, ModelNotLoaded) as e:
        status = "error"
        raise HTTPException(status_code=503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
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


@router.get("/models", response_model=ModelList)
def list_models(request: Request):
    embedder = request.app.state.embedder
    models = []
    for name in list_embedder_names():
        try:
            dim = embedder.dim if (embedder and name == embedder.model_name) else None
        except Exception:
            dim = None
        models.append(Model(id=name, dimensions=dim))
    return ModelList(data=models)


@router.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str, request: Request):
    if model_id not in EMBEDDER_REGISTRY:
        raise HTTPException(status_code=404, detail={"error": {"code": "model_not_found", "message": model_id}})
    embedder = request.app.state.embedder
    dim = embedder.dim if (embedder and embedder.model_name == model_id) else None
    return Model(id=model_id, dimensions=dim)