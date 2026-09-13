"""``POST /v1/multimodal_embeddings`` — cross-modal text+image vectorization.

Each input item is either ``{"text": "..."}`` (Chinese text) or
``{"image": {"data": "<base64>", "mime": "..."}}``. The endpoint batches
by modality internally — one ``embed_text`` call for all text items,
one ``embed_images`` call for all image items — then reassembles results
in input order. Both modalities land in the same shared vector space
(typically a projection head output), so cosine similarity across
modalities is meaningful: this is the foundation for text-search-image
and image-search-text retrieval.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    ImageDecodeError,
    ImageTooLarge,
    ModelNotLoadedForImages,
    MultimodalEmbedderError,
    UnsupportedMime,
)
from vector_service.core.logging import get_logger
from vector_service.embeddings.image_base import ImageInput
from vector_service.embeddings.image_decoding import decode_image
from vector_service.embeddings.multimodal_registry import (
    get_multimodal_embedder_class,
)
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.multimodal_embeddings import (
    MultimodalEmbeddingData,
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
)
from vector_service.schemas.openai import EmbeddingUsage

router = APIRouter(prefix="/v1", tags=["multimodal_embeddings"])
log = get_logger(__name__)


def _decode_images(
    payloads, *, max_bytes: int, allowed_mime: set[str]
) -> list[ImageInput]:
    """Decode every image payload; raise 422 with failed_indices on any failure."""
    decoded = []
    failed = []
    for i, p in enumerate(payloads):
        try:
            decoded.append(decode_image(p.data, p.mime, max_bytes=max_bytes, allowed_mime=allowed_mime))
        except UnsupportedMime as e:
            failed.append((i, "unsupported_mime", str(e), {"got": e.got, "allowed": e.allowed}))
        except ImageTooLarge as e:
            failed.append((i, "image_too_large", str(e), {"got": e.got, "max": e.max}))
        except ImageDecodeError as e:
            failed.append((i, "image_decode_failed", str(e), {}))
    if failed:
        code = failed[0][1]
        if not all(f[1] == code for f in failed):
            code = "image_decode_failed"  # mixed failures collapse
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": code,
                "message": failed[0][2],
                "failed_indices": [f[0] for f in failed],
                "failures": [
                    {"index": f[0], "code": f[1], "message": f[2], **f[3]}
                    for f in failed
                ],
            }},
        )
    return decoded


@router.post(
    "/multimodal_embeddings",
    response_model=MultimodalEmbeddingResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown multimodal embedder id."},
        422: {"model": ErrorEnvelope, "description": "Validation / decoding failure."},
        503: {"model": ErrorEnvelope, "description": "Multimodal embedder unavailable."},
    },
    summary="Create multimodal embeddings",
    description=(
        "Embed a mixed list of Chinese texts and base64-encoded images using "
        "a registered multimodal embedder (e.g. Chinese-CLIP). Text and image "
        "vectors live in the same shared space — cosine similarity is "
        "meaningful across modalities, enabling text-search-image and "
        "image-search-text retrieval."
    ),
)
async def create_multimodal_embeddings(
    body: MultimodalEmbeddingRequest, request: Request
):
    settings = request.app.state.settings.multimodal_embedding

    # Validate model id.
    try:
        get_multimodal_embedder_class(body.model)
    except MultimodalEmbedderError as e:
        raise HTTPException(
            status_code=404,
            detail={"error": {
                "code": "model_not_found",
                "message": str(e) or f"unknown model {body.model!r}",
                "model": body.model,
            }},
        )

    embedder = getattr(request.app.state, "multimodal_embedder", None)
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail={"error": {
                "code": "multimodal_embedder_unavailable",
                "message": (
                    f"multimodal embedder {body.model!r} is not loaded; "
                    "the lifespan step did not initialise it"
                ),
                "model": body.model,
            }},
        )

    items = body.input if isinstance(body.input, list) else [body.input]
    if not items:
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": "empty_input",
                "message": "input must contain at least one item",
            }},
        )
    if len(items) > settings.max_items_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {
                "code": "too_many_items",
                "message": (
                    f"got {len(items)} items but the per-request limit is "
                    f"{settings.max_items_per_request}"
                ),
                "max": settings.max_items_per_request,
                "got": len(items),
            }},
        )

    # Bucket by modality, preserving original indices.
    text_indices: list[int] = []
    text_payloads: list[str] = []
    image_indices: list[int] = []
    image_payloads = []
    for i, item in enumerate(items):
        if item.text is not None:
            if len(item.text) > settings.max_text_chars:
                raise HTTPException(
                    status_code=422,
                    detail={"error": {
                        "code": "text_too_long",
                        "message": (
                            f"item {i} text length {len(item.text)} exceeds "
                            f"max_text_chars={settings.max_text_chars}"
                        ),
                        "max": settings.max_text_chars,
                        "got": len(item.text),
                    }},
                )
            text_indices.append(i)
            text_payloads.append(item.text)
        else:
            image_indices.append(i)
            image_payloads.append(item.image)

    decoded_images = _decode_images(
        image_payloads,
        max_bytes=settings.max_image_bytes,
        allowed_mime=set(settings.allowed_mime),
    )

    # Allocate output slots in input order; fill them per modality.
    out_vectors: list[list[float] | None] = [None] * len(items)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"

    try:
        if text_payloads:
            text_vecs = await loop.run_in_executor(
                None, embedder.embed_text, text_payloads
            )
            for idx, vec in zip(text_indices, text_vecs):
                out_vectors[idx] = vec
        if decoded_images:
            image_vecs = await loop.run_in_executor(
                None, embedder.embed_images, decoded_images
            )
            for idx, vec in zip(image_indices, image_vecs):
                out_vectors[idx] = vec
    except (MultimodalEmbedderError, ModelNotLoadedForImages) as e:
        status = "error"
        raise HTTPException(
            status_code=503,
            detail={"error": {
                "code": "multimodal_embedder_unavailable",
                "message": str(e) or f"multimodal embedder {body.model!r} unavailable",
                "model": body.model,
                "item_count": len(items),
                "exception_type": type(e).__name__,
            }},
        )
    finally:
        from vector_service.core.metrics import (
            MULTIMODAL_EMBEDDING_DURATION_SECONDS,
            MULTIMODAL_EMBEDDING_INPUTS_TOTAL,
            MULTIMODAL_EMBEDDING_REQUESTS_TOTAL,
        )

        MULTIMODAL_EMBEDDING_DURATION_SECONDS.labels(model=body.model, status=status).observe(
            time.perf_counter() - t0
        )
        MULTIMODAL_EMBEDDING_REQUESTS_TOTAL.labels(model=body.model, status=status).inc()
        MULTIMODAL_EMBEDDING_INPUTS_TOTAL.labels(model=body.model).inc(len(items))

    # out_vectors is fully populated at this point (one path runs if the
    # other is empty). Defensive None-check — surfaces a programming bug
    # if any slot was missed.
    for i, v in enumerate(out_vectors):
        if v is None:
            raise HTTPException(
                status_code=500,
                detail={"error": {
                    "code": "internal",
                    "message": f"missing embedding for index {i}",
                }},
            )

    data = [MultimodalEmbeddingData(index=i, embedding=v) for i, v in enumerate(out_vectors)]
    log.info(
        "multimodal_embedding_request",
        model=body.model,
        item_count=len(items),
        text_count=len(text_payloads),
        image_count=len(decoded_images),
        duration_ms=int((time.perf_counter() - t0) * 1000),
        status=status,
    )
    return MultimodalEmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(
            prompt_tokens=len(items),
            total_tokens=len(items),
        ),
    )