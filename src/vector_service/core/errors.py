"""Exception hierarchy. Translates to HTTP in the API layer."""
from __future__ import annotations

from typing import Any


class VectorServiceError(Exception):
    """Base for all errors raised by vector-service."""


class EmbedderError(VectorServiceError):
    """Embedding model inference or availability failure."""


class ModelNotLoaded(EmbedderError):
    """Model failed to load at startup or is no longer available."""


class StoreError(VectorServiceError):
    """Vector store backend failure."""


class DatabaseNotFound(StoreError):
    """Operation referenced a non-existent database."""

    def __init__(self, message: str, *, name: str | None = None):
        super().__init__(message)
        self.name = name


class DatabaseAlreadyExists(StoreError):
    """Creation would clobber an existing database."""

    def __init__(self, message: str, *, name: str | None = None):
        super().__init__(message)
        self.name = name


class CollectionNotFound(StoreError):
    """Operation referenced a non-existent collection."""


class CollectionAlreadyExists(StoreError):
    """Creation would clobber an existing collection."""


class DimensionMismatch(StoreError):
    """Vector dimension does not match the collection's."""

    def __init__(self, message: str, *, expected: int | None = None, got: int | None = None):
        super().__init__(message)
        self.expected = expected
        self.got = got


class BackendError(StoreError):
    """Wrapped native backend exception."""


class RerankerError(VectorServiceError):
    """Base class for reranker failures."""


class RerankerNotLoaded(RerankerError):
    """Reranker weights are not loaded (lifespan failed or skipped)."""


class ImageEmbedderError(VectorServiceError):
    """Base class for image embedding failures (sibling of EmbedderError)."""


class ModelNotLoadedForImages(ImageEmbedderError):
    """Image model failed to load at startup or is no longer available.

    Intentionally NOT a subclass of ModelNotLoaded — the text and image
    embedder lifecycles are independent, and a missing image model should
    not look like a text-embedder failure to dispatchers.
    """


class ImageDecodeError(VectorServiceError):
    """Base class for image decoding failures (422 image_decode_failed family)."""


class UnsupportedMime(ImageDecodeError):
    """MIME type was not in the allow-list."""

    def __init__(self, message: str, *, got: str, allowed: list[str]):
        super().__init__(message)
        self.got = got
        self.allowed = allowed


class ImageTooLarge(ImageDecodeError):
    """Decoded image bytes exceeded the configured maximum."""

    def __init__(self, message: str, *, got: int, max: int):
        super().__init__(message)
        self.got = got
        self.max = max
