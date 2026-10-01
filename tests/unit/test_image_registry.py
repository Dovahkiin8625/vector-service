"""Image embedder registry lookup."""
from __future__ import annotations

import pytest

from vector_service.core.errors import ImageEmbedderError
from vector_service.embeddings.image_registry import (
    IMAGE_EMBEDDER_REGISTRY,
    get_image_embedder_class,
    list_image_embedder_names,
)


def test_registry_contains_openclip_vit_l_14():
    assert "openclip-vit-l-14" in IMAGE_EMBEDDER_REGISTRY


def test_get_image_embedder_class_returns_registered_class():
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    cls = get_image_embedder_class("openclip-vit-l-14")
    assert cls is OpenCLIPVitL14ImageEmbedder


def test_get_image_embedder_class_unknown_raises():
    with pytest.raises(ImageEmbedderError) as ei:
        get_image_embedder_class("does-not-exist")
    assert "does-not-exist" in str(ei.value)


def test_list_image_embedder_names_is_non_empty():
    names = list_image_embedder_names()
    assert isinstance(names, list)
    assert "openclip-vit-l-14" in names
