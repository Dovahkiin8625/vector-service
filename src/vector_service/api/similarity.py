"""Similarity endpoints under ``/v1``.

Three POST handlers, one router tagged ``similarity``:

* ``/v1/text_similarity``       — score query text against N candidates.
* ``/v1/image_similarity``      — score query image against N candidate images.
* ``/v1/multimodal_similarity`` — score query against N candidates where
  each item may be text or image (multimodal embedder's shared space).

All three share the same response envelope (``SimilarityResponse``) and
reuse the live embedders via ``app.state.<family>`` — no new model
families are introduced, so ``GET /v1/models`` is untouched. The metric
literal mirrors the one already used by the vector-store schemas
(``cosine`` / ``ip`` / ``l2``), keeping scores consistent between this
endpoint and search results.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    EmbedderError,
    ImageEmbedderError,
    ModelNotLoadedForSimilarity,
    MultimodalEmbedderError,
    SimilarityError,
)
from vector_service.core.logging import get_logger
from vector_service.embeddings.image_decoding import (
    decode_batch_or_422,
    fail_envelope_422,
)
from vector_service.embeddings.image_base import ImageInput
from vector_service.embeddings.registry import get_embedder_class
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.embeddings.multimodal_registry import get_multimodal_embedder_class
from vector_service.embeddings.similarity import arg_sort, pairwise_scores, validate_metric
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.similarity import (
    ImageSimilarityRequest,
    MultimodalSimilarityRequest,
    SimilarityResponse,
    SimilarityResultItem,
    TextSimilarityRequest,
)

router = APIRouter(prefix="/v1", tags=["similarity"])
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _http_error(status: int, code: str, message: str, **extra) -> HTTPException:
    """Build an HTTPException with the canonical error envelope shape."""
    detail: dict = {"error": {"code": code, "message": message}}
    detail["error"].update(extra)
    return HTTPException(status_code=status, detail=detail)


async def _dispatch_embed(
    loop: asyncio.AbstractEventLoop,
    fn,
    *args,
    timeout_s: float,
):
    """Wrap ``fn(*args)`` in ``run_in_executor`` + ``wait_for`` and translate
    timeouts into ``SimilarityError``. Mirrors the inference-route idiom
    used elsewhere in the API."""
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, fn, *args),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as e:
        raise SimilarityError(
            f"similarity call did not finish within {timeout_s}s"
        ) from e


# ---------------------------------------------------------------------------
# Text-text
# ---------------------------------------------------------------------------


@router.post(
    "/text_similarity",
    response_model=SimilarityResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown text-embedder id."},
        422: {"model": ErrorEnvelope, "description": "Validation failed."},
        503: {"model": ErrorEnvelope, "description": "Text embedder unavailable."},
    },
    summary="Score a query text against candidate texts",
    description=(
        "Embed the query and each candidate with the registered text "
        "embedder, then compute a pairwise similarity using the chosen "
        "metric (cosine / ip / l2). Results are sorted most-similar-first "
        "for cosine and ip, closest-first for l2."
    ),
)
async def text_similarity(body: TextSimilarityRequest, request: Request) -> SimilarityResponse:
    # ---- 404: unknown model id ---------------------------------------
    try:
        get_embedder_class(body.model)
    except EmbedderError as e:
        raise _http_error(
            404, "model_not_found",
            str(e) or f"unknown text-embedder model {body.model!r}",
            model=body.model,
        )

    # ---- 503: embedder not loaded -----------------------------------
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:
        raise ModelNotLoadedForSimilarity(
            f"text embedder {body.model!r} is not loaded"
        )

    metric = validate_metric(body.metric)
    timeout_s = getattr(request.app.state.settings, "inference_timeout_seconds", 60.0)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        # Embed query + candidates in a single batch — same dim, same
        # preprocessing, no asymmetry between sides.
        all_texts = [body.query, *body.documents]
        vecs = await _dispatch_embed(
            loop, embedder.embed_documents, all_texts,
            timeout_s=timeout_s,
        )
    except SimilarityError as e:
        log.warning("text_similarity_error", model=body.model, error=str(e))
        raise _http_error(
            503, "similarity_unavailable",
            str(e) or "text similarity call failed",
            model=body.model,
            exception_type=type(e).__name__,
        )

    query_vec = vecs[0]
    doc_vecs = vecs[1:]
    scores = pairwise_scores(query_vec, doc_vecs, metric)
    order = arg_sort(scores, metric)
    results = [
        SimilarityResultItem(index=i, score=scores[i]) for i in order
    ]

    log.info(
        "text_similarity_completed",
        model=body.model,
        metric=metric,
        n_candidates=len(body.documents),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return SimilarityResponse(model=body.model, metric=metric, results=results)


# ---------------------------------------------------------------------------
# Image-image
# ---------------------------------------------------------------------------


@router.post(
    "/image_similarity",
    response_model=SimilarityResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown image-embedder id."},
        422: {"model": ErrorEnvelope, "description": "Image decoding / validation failure."},
        503: {"model": ErrorEnvelope, "description": "Image embedder unavailable."},
    },
    summary="Score a query image against candidate images",
    description=(
        "Embed the query image and each candidate image with the "
        "registered image embedder, then compute pairwise similarity. "
        "Decoding uses the same base64 / MIME / size validation as "
        "``POST /v1/image_embeddings``."
    ),
)
async def image_similarity(body: ImageSimilarityRequest, request: Request) -> SimilarityResponse:
    i_settings = request.app.state.settings.image_embedding

    # ---- 404: unknown model id ---------------------------------------
    try:
        get_image_embedder_class(body.model)
    except ImageEmbedderError as e:
        raise _http_error(
            404, "model_not_found",
            str(e) or f"unknown image-embedder model {body.model!r}",
            model=body.model,
        )

    # ---- 503: embedder not loaded -----------------------------------
    embedder = getattr(request.app.state, "image_embedder", None)
    if embedder is None:
        raise ModelNotLoadedForSimilarity(
            f"image embedder {body.model!r} is not loaded"
        )

    metric = validate_metric(body.metric)

    # ---- 422: decode base64 + validate MIME/size --------------------
    items = [body.query, *body.documents]
    decoded, failures = decode_batch_or_422(
        items,
        extract=lambda it: (it.data, it.mime),
        max_bytes=i_settings.max_image_bytes,
        allowed_mime=set(i_settings.allowed_mime),
    )
    if failures:
        # fail_envelope_422 returns either the specific code (e.g.
        # ``unsupported_mime`` when all failures share it) or the generic
        # ``image_decode_failed`` fallback for mixed failures.
        env = fail_envelope_422(failures)["error"]
        code = env.get("code", "image_decode_failed")
        message = env.get("message", failures[0][2])
        extras = {k: v for k, v in env.items() if k not in ("code", "message")}
        raise _http_error(422, code, message, **extras)

    timeout_s = getattr(request.app.state.settings, "inference_timeout_seconds", 60.0)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        vecs = await _dispatch_embed(
            loop, embedder.embed_images, decoded,
            timeout_s=timeout_s,
        )
    except SimilarityError as e:
        log.warning("image_similarity_error", model=body.model, error=str(e))
        raise _http_error(
            503, "similarity_unavailable",
            str(e) or "image similarity call failed",
            model=body.model,
            exception_type=type(e).__name__,
        )

    query_vec = vecs[0]
    doc_vecs = vecs[1:]
    scores = pairwise_scores(query_vec, doc_vecs, metric)
    order = arg_sort(scores, metric)
    results = [
        SimilarityResultItem(index=i, score=scores[i]) for i in order
    ]

    log.info(
        "image_similarity_completed",
        model=body.model,
        metric=metric,
        n_candidates=len(body.documents),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return SimilarityResponse(model=body.model, metric=metric, results=results)


# ---------------------------------------------------------------------------
# Multimodal (text + image, shared space)
# ---------------------------------------------------------------------------


@router.post(
    "/multimodal_similarity",
    response_model=SimilarityResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown multimodal-embedder id."},
        422: {"model": ErrorEnvelope, "description": "Validation / image decoding failure."},
        503: {"model": ErrorEnvelope, "description": "Multimodal embedder unavailable."},
    },
    summary="Score a query against candidates where each item is text or image",
    description=(
        "Embed the query and each candidate using the multimodal embedder "
        "(e.g. Chinese-CLIP). Text and image inputs land in the same shared "
        "vector space — cosine similarity is meaningful across modalities, "
        "so ``{text}`` vs ``{image}`` is a valid query/candidate pairing. "
        "Bucketing by modality mirrors ``POST /v1/multimodal_embeddings``."
    ),
)
async def multimodal_similarity(body: MultimodalSimilarityRequest, request: Request) -> SimilarityResponse:
    m_settings = request.app.state.settings.multimodal_embedding

    # ---- 404: unknown model id ---------------------------------------
    try:
        get_multimodal_embedder_class(body.model)
    except MultimodalEmbedderError as e:
        raise _http_error(
            404, "model_not_found",
            str(e) or f"unknown multimodal-embedder model {body.model!r}",
            model=body.model,
        )

    # ---- 503: embedder not loaded -----------------------------------
    embedder = getattr(request.app.state, "multimodal_embedder", None)
    if embedder is None:
        raise ModelNotLoadedForSimilarity(
            f"multimodal embedder {body.model!r} is not loaded"
        )

    metric = validate_metric(body.metric)

    all_items = [body.query, *body.documents]

    # Bucket by modality, preserving original indices.
    text_indices: list[int] = []
    text_payloads: list[str] = []
    image_indices: list[int] = []
    image_payloads = []
    for i, item in enumerate(all_items):
        if item.text is not None:
            if len(item.text) > m_settings.max_text_chars:
                raise _http_error(
                    422, "text_too_long",
                    f"item {i} text length {len(item.text)} exceeds max_text_chars={m_settings.max_text_chars}",
                    index=i,
                    got=len(item.text),
                    max=m_settings.max_text_chars,
                )
            text_indices.append(i)
            text_payloads.append(item.text)
        else:
            image_indices.append(i)
            image_payloads.append(item.image)

    decoded_images: list[ImageInput] = []
    if image_payloads:
        decoded, failures = decode_batch_or_422(
            image_payloads,
            extract=lambda p: (p.data, p.mime),
            max_bytes=m_settings.max_image_bytes,
            allowed_mime=set(m_settings.allowed_mime),
        )
        if failures:
            env = fail_envelope_422(failures)["error"]
            code = env.get("code", "image_decode_failed")
            message = env.get("message", failures[0][2])
            extras = {k: v for k, v in env.items() if k not in ("code", "message")}
            raise _http_error(422, code, message, **extras)
        decoded_images = decoded

    out_vectors: list[list[float] | None] = [None] * len(all_items)
    timeout_s = getattr(request.app.state.settings, "inference_timeout_seconds", 60.0)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()

    try:
        if text_payloads:
            text_vecs = await _dispatch_embed(
                loop, embedder.embed_text, text_payloads,
                timeout_s=timeout_s,
            )
            for idx, vec in zip(text_indices, text_vecs):
                out_vectors[idx] = vec
        if decoded_images:
            image_vecs = await _dispatch_embed(
                loop, embedder.embed_images, decoded_images,
                timeout_s=timeout_s,
            )
            for idx, vec in zip(image_indices, image_vecs):
                out_vectors[idx] = vec
    except SimilarityError as e:
        log.warning("multimodal_similarity_error", model=body.model, error=str(e))
        raise _http_error(
            503, "similarity_unavailable",
            str(e) or "multimodal similarity call failed",
            model=body.model,
            exception_type=type(e).__name__,
        )

    # Defensive: every slot must be filled (one of the two paths always
    # runs if both are present, so this only trips on a programmer bug).
    for i, v in enumerate(out_vectors):
        if v is None:
            raise _http_error(500, "internal", f"missing embedding for index {i}")

    query_vec = out_vectors[0]
    doc_vecs = out_vectors[1:]
    scores = pairwise_scores(query_vec, doc_vecs, metric)
    order = arg_sort(scores, metric)
    results = [
        SimilarityResultItem(index=i, score=scores[i]) for i in order
    ]

    log.info(
        "multimodal_similarity_completed",
        model=body.model,
        metric=metric,
        n_candidates=len(body.documents),
        text_count=len(text_payloads),
        image_count=len(decoded_images),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return SimilarityResponse(model=body.model, metric=metric, results=results)