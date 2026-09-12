"""Application lifespan: load model + open vector store."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import (
    ImageEmbedderError,
    ModelNotLoaded,
    ModelNotLoadedForImages,
    RerankerError,
    RerankerNotLoaded,
)
from vector_service.core.logging import get_logger, setup_logging
from vector_service.core.metrics import MODEL_LOADED, VS_INFO
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.embeddings.registry import get_embedder_class
from vector_service.rerankers.base import Reranker
from vector_service.rerankers.registry import get_reranker_class
from vector_service.stores.registry import build_store

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


def build_embedder(settings: Settings):
    cls = get_embedder_class(settings.embedding_backend)
    return cls(settings=settings)


def build_reranker(settings: Settings) -> Reranker:
    """Construct a reranker instance for ``settings.reranker.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_reranker_class(settings.reranker.backend)
    return cls(settings=settings)


def build_image_embedder(settings: Settings) -> ImageEmbedder:
    """Construct an image embedder instance for ``settings.image_embedding.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_image_embedder_class(settings.image_embedding.backend)
    return cls(settings=settings.image_embedding)


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    settings = get_settings()
    setup_logging(settings.log_format, settings.log_level)
    VS_INFO.labels(
        version="0.2.0",
        embedding_backend=settings.embedding_backend,
        vector_store_backend=settings.vector_store_backend,
    ).set(1)

    embedder = build_embedder(settings)
    store = build_store(settings)

    # Eagerly open the vector-store connection so /readyz can report a
    # truthful state. If Milvus is unreachable at startup we still want
    # the process to come up so embeddings keep working; surface the
    # error in the log and let /readyz return 503.
    try:
        if hasattr(store, "_ensure_connected"):
            store._ensure_connected()
            log.info("store_connected", backend=store.backend_name, uri=getattr(store, "uri", ""))
        else:
            log.info("store_opened", backend=store.backend_name, uri=getattr(store, "uri", ""))
    except Exception as e:
        log.warning("store_connect_failed", backend=getattr(store, "backend_name", "?"), error=str(e))

    # Eagerly load the embedder so the first user request isn't held up
    # by model download + weight load + warmup. A failed load is logged
    # and surfaced via /readyz (503), but doesn't kill the process —
    # this lets operators inspect the state on a half-broken host.
    try:
        embedder.load()
    except ModelNotLoaded as e:
        log.error("model_load_failed", model=embedder.model_name, error=str(e))
    else:
        device = getattr(embedder, "_device", "unknown")
        MODEL_LOADED.labels(kind="embedder").set(1)
        log.info("model_loaded", model=embedder.model_name, device=device, dim=embedder.dim)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store

    # ---- reranker (loaded last; not on the /search hot path) ----
    # Missing/invalid VS_RERANKER__BACKEND is treated as a startup
    # error: re-raise so lifespan exits non-zero. /readyz is diagnostic
    # only — embedder + store still gate the overall ready signal.
    try:
        reranker = build_reranker(settings)
        reranker.load()
        app.state.reranker = reranker
        MODEL_LOADED.labels(kind="reranker").set(1)
        log.info(
            "reranker_loaded",
            model=reranker.model_name,
            backend=settings.reranker.backend,
        )
    except (RerankerNotLoaded, RerankerError, KeyError) as exc:
        MODEL_LOADED.labels(kind="reranker").set(0)
        log.error(
            "reranker_load_failed",
            backend=settings.reranker.backend,
            error=str(exc),
        )
        raise

    # ---- image embedder (parallel to text embedder) ----
    # A failed image-embedder load is logged + surfaced via /readyz (503),
    # but does NOT kill the process — same fail-open policy as the text
    # embedder.
    try:
        image_embedder = build_image_embedder(settings)
        image_embedder.load()
        app.state.image_embedder = image_embedder
        MODEL_LOADED.labels(kind="image_embedder").set(1)
        log.info(
            "image_embedder_loaded",
            model=image_embedder.model_name,
            device=getattr(image_embedder, "_device", "unknown"),
            dim=image_embedder.dim,
        )
    except (ModelNotLoadedForImages, ImageEmbedderError) as exc:
        MODEL_LOADED.labels(kind="image_embedder").set(0)
        log.error(
            "image_embedder_load_failed",
            backend=settings.image_embedding.backend,
            error=str(exc),
        )

    try:
        yield
    finally:
        store.close()
        log.info("shutdown")
