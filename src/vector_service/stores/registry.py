"""Vector store registry.

vector-service ships one production :class:`VectorStore` implementation:
:class:`vector_service.stores.milvus.MilvusStore`, which talks directly to
a Milvus server via :mod:`pymilvus`. To add another backend, drop a new
module under ``stores/`` implementing the :class:`VectorStore` ABC and
branch on ``settings.vector_store_backend`` here.
"""
from __future__ import annotations

from vector_service.core.config import Settings
from vector_service.core.errors import StoreError
from vector_service.stores.base import VectorStore


def build_store(settings: Settings) -> VectorStore:
    """Construct the configured vector-store backend."""
    backend = (settings.vector_store_backend or "").lower()
    if backend == "milvus":
        # Import lazily so importing this module doesn't require pymilvus
        # to be installed (e.g. when only running the embedder / docs).
        from vector_service.stores.milvus import MilvusStore

        return MilvusStore(settings=settings)
    raise StoreError(
        f"unknown vector_store_backend {settings.vector_store_backend!r}; "
        f"only 'milvus' is supported"
    )
