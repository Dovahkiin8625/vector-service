"""ABC contract for ImageEmbedder implementations."""
from __future__ import annotations

import pytest

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


def test_image_input_is_immutable():
    item = ImageInput(data=b"\x00", mime="image/png")
    with pytest.raises((AttributeError, TypeError)):
        item.data = b"\x01"  # type: ignore[misc]


def test_image_input_carries_bytes_and_mime():
    item = ImageInput(data=b"abc", mime="image/jpeg")
    assert item.data == b"abc"
    assert item.mime == "image/jpeg"


def test_image_embedder_base_load_is_abstract():
    with pytest.raises(TypeError):
        ImageEmbedder()  # type: ignore[abstract]


def test_image_embedder_base_subclass_must_implement_methods():
    class _Half(ImageEmbedder):
        dim = 768
        model_name = "half"

        def load(self):
            return None

        # Missing embed_images and embed_query_image

    with pytest.raises(TypeError):
        _Half()  # type: ignore[abstract]


def test_subclass_can_be_loaded_and_called():
    class _Fake(ImageEmbedder):
        dim = 768
        model_name = "fake"

        def __init__(self):
            self.load_called = 0

        def load(self):
            self.load_called += 1

        def embed_images(self, images):
            return [[0.0] * 768 for _ in images]

        def embed_query_image(self, image):
            return [0.0] * 768

    e = _Fake()
    e.load()
    assert e.load_called == 1
    assert len(e.embed_images([ImageInput(b"x", "image/png"), ImageInput(b"y", "image/png")])) == 2
    assert len(e.embed_query_image(ImageInput(b"x", "image/png"))) == 768


def test_subclass_load_failure_propagates_model_not_loaded_for_images():
    class _Boom(ImageEmbedder):
        dim = 768
        model_name = "boom"

        def load(self):
            raise ModelNotLoadedForImages("weights missing")

        def embed_images(self, images): return []
        def embed_query_image(self, image): return []

    with pytest.raises(ModelNotLoadedForImages):
        _Boom().load()
