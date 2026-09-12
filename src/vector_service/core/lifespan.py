"""Application lifespan: load model + open vector store."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import ModelNotLoaded
from vector_service.core.logging import get_logger, setup_logging
from vector_service.core.metrics import MODEL_LOADED, VS_INFO
from vector_service.embeddings.registry import get_embedder_class
from vector_service.stores.registry import build_store

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


def build_embedder(settings: Settings):
    cls = get_embedder_class(settings.embedding_backend)
    return cls(settings=settings)


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

    try:
        yield
    finally:
        store.close()
        log.info("shutdown")
