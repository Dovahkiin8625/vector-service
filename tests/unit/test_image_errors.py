"""Exception classes for image embedding subsystem."""
from __future__ import annotations

import pytest

from vector_service.core.errors import (
    EmbedderError,
    ImageDecodeError,
    ImageEmbedderError,
    ImageTooLarge,
    ModelNotLoaded,
    ModelNotLoadedForImages,
    UnsupportedMime,
    VectorServiceError,
)


def test_image_embedder_error_inherits_vector_service_error():
    assert issubclass(ImageEmbedderError, VectorServiceError)


def test_model_not_loaded_for_images_inherits_image_embedder_error():
    assert issubclass(ModelNotLoadedForImages, ImageEmbedderError)
    assert issubclass(ModelNotLoadedForImages, ModelNotLoaded) is False  # distinct from text


def test_image_decode_error_inherits_vector_service_error():
    assert issubclass(ImageDecodeError, VectorServiceError)


def test_unsupported_mime_carries_got_and_allowed():
    e = UnsupportedMime("nope", got="image/bmp", allowed=["image/jpeg", "image/png"])
    assert e.got == "image/bmp"
    assert e.allowed == ["image/jpeg", "image/png"]
    assert str(e) == "nope"


def test_image_too_large_carries_got_and_max():
    e = ImageTooLarge("big", got=2_000_000, max=1_000_000)
    assert e.got == 2_000_000
    assert e.max == 1_000_000


def test_image_embedder_error_caught_as_embedder_error_for_dispatch():
    """Image embedder errors are siblings, NOT children of EmbedderError —
    routes dispatch on the specific subclass."""
    assert not issubclass(ImageEmbedderError, EmbedderError)
