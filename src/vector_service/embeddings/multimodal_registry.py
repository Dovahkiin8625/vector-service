"""Multimodal embedder registry."""
from __future__ import annotations

from vector_service.core.errors import MultimodalEmbedderError
from vector_service.embeddings.chinese_clip_multimodal import (
    ChineseCLIPMultimodalEmbedder,
)
from vector_service.embeddings.multimodal_base import MultimodalEmbedder


MULTIMODAL_EMBEDDER_REGISTRY: dict[str, type[MultimodalEmbedder]] = {
    "chinese-clip-vit-base-patch16": ChineseCLIPMultimodalEmbedder,
}


def get_multimodal_embedder_class(name: str) -> type[MultimodalEmbedder]:
    if name not in MULTIMODAL_EMBEDDER_REGISTRY:
        raise MultimodalEmbedderError(f"unknown multimodal embedding model: {name!r}")
    return MULTIMODAL_EMBEDDER_REGISTRY[name]


def list_multimodal_embedder_names() -> list[str]:
    return list(MULTIMODAL_EMBEDDER_REGISTRY.keys())