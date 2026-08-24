"""Vector store registry."""
from __future__ import annotations

from vector_service.core.config import Settings
from vector_service.core.errors import StoreError
from vector_service.stores.base import VectorStore
from vector_service.stores.milvus_lite import MilvusLiteStore

STORES_REGISTRY: dict[str, type[VectorStore]] = {
    "milvus_lite": MilvusLiteStore,
}


def get_store_class(name: str) -> type[VectorStore]:
    if name not in STORES_REGISTRY:
        raise StoreError(f"unknown vector store backend: {name!r}")
    return STORES_REGISTRY[name]


def build_store(settings: Settings) -> VectorStore:
    cls = get_store_class(settings.vector_store_backend)
    return cls(settings=settings)