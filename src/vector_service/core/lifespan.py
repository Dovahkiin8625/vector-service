"""Application lifespan: load model + open vector store."""
from __future__ import annotations

from contextlib import asynccontextmanager
import time
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
from vector_service.core.model_lifecycle import attach_default_slots
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.embeddings.multimodal_base import MultimodalEmbedder
from vector_service.embeddings.multimodal_registry import (
    get_multimodal_embedder_class,
)
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


def build_multimodal_embedder(settings: Settings) -> MultimodalEmbedder:
    """Construct a multimodal embedder for ``settings.multimodal_embedding.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_multimodal_embedder_class(settings.multimodal_embedding.backend)
    return cls(settings=settings.multimodal_embedding)


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    settings = get_settings()
    setup_logging(settings.log_format, settings.log_level)
    VS_INFO.labels(
        version="0.2.0",
        embedding_backend=settings.embedding_backend,
        vector_store_backend=settings.vector_store_backend,
    ).set(1)

    app.state.settings = settings
    app.state.startup_ts = time.time()
    app.state.store = build_store(settings)
    store = app.state.store

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

    # Attach the per-family ``ModelSlot`` holders BEFORE any optional
    # loads so the hot-reload routes always see the slots. With the
    # default ``auto_load=false`` for every family the slots start
    # empty and ``app.state.<family>`` stays ``None`` until an
    # operator calls ``POST /v1/models/{id}/load``.
    attach_default_slots(app, settings=settings)
    app.state.embedder = None
    app.state.reranker = None
    app.state.image_embedder = None
    app.state.multimodal_embedder = None
    MODEL_LOADED.labels(kind="embedder").set(0)
    MODEL_LOADED.labels(kind="reranker").set(0)
    MODEL_LOADED.labels(kind="image_embedder").set(0)
    MODEL_LOADED.labels(kind="multimodal_embedder").set(0)

    # ---- text embedder (opt-in eager load) -----------------------
    # ``embedding_auto_load`` defaults to ``False`` so a fresh process
    # starts in a zero-state: no model constructed, no slot populated,
    # /v1/embeddings returns 503. Operators trigger the load via the
    # dashboard or ``POST /v1/models/{id}/load``. The fail-open
    # contract from the original behaviour is preserved — a failed
    # eager load is logged + surfaced via /readyz but does NOT kill
    # the process.
    if settings.embedding_auto_load:
        try:
            embedder = build_embedder(settings)
            embedder.load()
        except ModelNotLoaded as e:
            log.error("model_load_failed", error=str(e))
        except Exception as e:
            log.error("model_load_unexpected", error=str(e), exception_type=type(e).__name__)
        else:
            app.state.embedder = embedder
            app.state._slot_embedder.set_instance(embedder)
            MODEL_LOADED.labels(kind="embedder").set(1)
            device = getattr(embedder, "_device", "unknown")
            log.info(
                "model_loaded",
                model=embedder.model_name, device=device, dim=embedder.dim,
            )

    # ---- reranker (opt-in eager load) ---------------------------
    # Missing/invalid VS_RERANKER__BACKEND is treated as a startup
    # error: re-raise so lifespan exits non-zero ONLY when auto_load
    # is on (without auto_load the reranker simply isn't built).
    if settings.reranker.auto_load:
        try:
            reranker = build_reranker(settings)
            reranker.load()
        except (RerankerNotLoaded, RerankerError, KeyError) as exc:
            log.error(
                "reranker_load_failed",
                backend=settings.reranker.backend,
                error=str(exc),
            )
            raise
        else:
            app.state.reranker = reranker
            app.state._slot_reranker.set_instance(reranker)
            MODEL_LOADED.labels(kind="reranker").set(1)
            log.info(
                "reranker_loaded",
                model=reranker.model_name,
                backend=settings.reranker.backend,
            )

    # ---- image embedder (opt-in eager load, fail-open) ----------
    if settings.image_embedding.auto_load:
        try:
            image_embedder = build_image_embedder(settings)
            image_embedder.load()
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.error(
                "image_embedder_load_failed",
                backend=settings.image_embedding.backend,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            app.state.image_embedder = image_embedder
            app.state._slot_image.set_instance(image_embedder)
            MODEL_LOADED.labels(kind="image_embedder").set(1)
            log.info(
                "image_embedder_loaded",
                model=image_embedder.model_name,
                device=getattr(image_embedder, "_device", "unknown"),
                dim=image_embedder.dim,
            )

    # ---- multimodal embedder (opt-in eager load, fail-open) -----
    if settings.multimodal_embedding.auto_load:
        try:
            multimodal_embedder = build_multimodal_embedder(settings)
            multimodal_embedder.load()
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.error(
                "multimodal_embedder_load_failed",
                backend=settings.multimodal_embedding.backend,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            app.state.multimodal_embedder = multimodal_embedder
            app.state._slot_multimodal.set_instance(multimodal_embedder)
            MODEL_LOADED.labels(kind="multimodal_embedder").set(1)
            log.info(
                "multimodal_embedder_loaded",
                model=multimodal_embedder.model_name,
                device=getattr(multimodal_embedder, "_device", "unknown"),
                dim=multimodal_embedder.dim,
            )

    try:
        yield
    finally:
        store.close()
        log.info("shutdown")
