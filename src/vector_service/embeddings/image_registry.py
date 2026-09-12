"""Image embedder registry."""
from __future__ import annotations

from vector_service.core.errors import ImageEmbedderError
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

IMAGE_EMBEDDER_REGISTRY: dict[str, type[ImageEmbedder]] = {
    "openclip-vit-l-14": OpenCLIPVitL14ImageEmbedder,
}


def get_image_embedder_class(name: str) -> type[ImageEmbedder]:
    if name not in IMAGE_EMBEDDER_REGISTRY:
        raise ImageEmbedderError(f"unknown image embedding model: {name!r}")
    return IMAGE_EMBEDDER_REGISTRY[name]


def list_image_embedder_names() -> list[str]:
    return list(IMAGE_EMBEDDER_REGISTRY.keys())
