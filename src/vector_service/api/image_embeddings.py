"""``POST /v1/image_embeddings`` — image vectorization endpoint."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    ImageDecodeError,
    ImageEmbedderError,
    ImageTooLarge,
    ModelNotLoadedForImages,
    UnsupportedMime,
)
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    IMAGE_EMBEDDING_DURATION_SECONDS,
    IMAGE_EMBEDDING_INPUTS_TOTAL,
    IMAGE_EMBEDDING_REQUESTS_TOTAL,
)
from vector_service.embeddings.image_decoding import (
    decode_batch_or_422,
    decode_image,
    fail_envelope_422,
)
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.image_embeddings import (
    ImageEmbeddingData,
    ImageEmbeddingRequest,
    ImageEmbeddingResponse,
)
from vector_service.schemas.openai import EmbeddingUsage

router = APIRouter(prefix="/v1", tags=["image_embeddings"])
log = get_logger(__name__)


def _decode_all(items, *, max_bytes, allowed_mime):
    """Decode every input; raise a 422 with failed_indices on any failure."""
    decoded, failures = decode_batch_or_422(
        items,
        extract=lambda item: (item.data, item.mime),
        max_bytes=max_bytes, allowed_mime=allowed_mime,
    )
    if failures:
        raise HTTPException(status_code=422, detail=fail_envelope_422(failures))
    return decoded


@router.post(
    "/image_embeddings",
    response_model=ImageEmbeddingResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown image embedder id."},
        422: {"model": ErrorEnvelope, "description": "Image decoding / validation failure."},
        503: {"model": ErrorEnvelope, "description": "Image embedder unavailable."},
    },
    summary="Create image embeddings",
    description=(
        "Embed one or more base64-encoded images using a registered image "
        "embedder. Model id must be one of the image_embedder entries "
        "returned by `GET /v1/models`."
    ),
)
async def create_image_embeddings(body: ImageEmbeddingRequest, request: Request):
    settings = request.app.state.settings.image_embedding

    # Validate model id.
    try:
        get_image_embedder_class(body.model)
    except ImageEmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {
            "code": "model_not_found",
            "message": str(e) or f"unknown model {body.model!r}",
            "model": body.model,
        }})

    # Resolve the live embedder. A None here means the lifespan step
    # never produced one (weights download failed, model dir missing,
    # etc.) — per spec the route should surface that as
    # 503 image_embedder_unavailable, never a 500 AttributeError.
    embedder = getattr(request.app.state, "image_embedder", None)
    if embedder is None:
        raise HTTPException(status_code=503, detail={"error": {
            "code": "image_embedder_unavailable",
            "message": (
                f"image embedder {body.model!r} is not loaded; "
                "the lifespan step did not initialise it"
            ),
            "model": body.model,
        }})

    items = body.input if isinstance(body.input, list) else [body.input]
    if len(items) > settings.max_images_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": "too_many_images",
                "message": (
                    f"got {len(items)} images but the per-request limit is "
                    f"{settings.max_images_per_request}"
                ),
                "max": settings.max_images_per_request,
                "got": len(items),
            }},
        )

    decoded = _decode_all(
        items,
        max_bytes=settings.max_image_bytes,
        allowed_mime=set(settings.allowed_mime),
    )

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"
    timeout_s = getattr(settings, "inference_timeout_seconds", 60.0)
    try:
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(None, embedder.embed_images, decoded),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            status = "timeout"
            raise ImageEmbedderError(
                f"image embedder {body.model!r} did not finish within {timeout_s}s"
            )
    except (ImageEmbedderError, ModelNotLoadedForImages) as e:
        status = "error"
        raise HTTPException(status_code=503, detail={"error": {
            "code": "image_embedder_unavailable",
            "message": str(e) or f"image embedder {body.model!r} unavailable",
            "model": body.model,
            "image_count": len(decoded),
            "exception_type": type(e).__name__,
        }})
    finally:
        IMAGE_EMBEDDING_DURATION_SECONDS.labels(model=body.model, status=status).observe(time.perf_counter() - t0)
        IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(model=body.model, status=status).inc()
        IMAGE_EMBEDDING_INPUTS_TOTAL.labels(model=body.model).inc(len(decoded))

    data = [ImageEmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)]
    log.info(
        "image_embedding_request",
        model=body.model,
        image_count=len(decoded),
        duration_ms=int((time.perf_counter() - t0) * 1000),
        status=status,
    )
    return ImageEmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(
            prompt_tokens=len(decoded),
            total_tokens=len(decoded),
        ),
    )
