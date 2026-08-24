"""Embedder registry."""
from __future__ import annotations

from vector_service.core.errors import EmbedderError
from vector_service.embeddings.base import Embedder
from vector_service.embeddings.bge_m3 import BGEM3Embedder

EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {
    "bge-m3": BGEM3Embedder,
}


def get_embedder_class(name: str) -> type[Embedder]:
    if name not in EMBEDDER_REGISTRY:
        raise EmbedderError(f"unknown embedding model: {name!r}")
    return EMBEDDER_REGISTRY[name]


def list_embedder_names() -> list[str]:
    return list(EMBEDDER_REGISTRY.keys())