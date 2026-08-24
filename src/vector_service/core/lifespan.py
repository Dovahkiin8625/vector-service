"""Application lifespan: load model + open store."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
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
        version="0.1.0",
        embedding_backend=settings.embedding_backend,
        vector_store_backend=settings.vector_store_backend,
    ).set(1)

    embedder = build_embedder(settings)
    store = build_store(settings)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store

    device = getattr(embedder, "_device", "unknown")
    MODEL_LOADED.labels(model=embedder.model_name, device=device).set(1)
    log.info("model_loaded", model=embedder.model_name, device=device, dim=embedder.dim)
    log.info("store_opened", backend=store.backend_name)

    try:
        yield
    finally:
        store.close()
        log.info("shutdown")