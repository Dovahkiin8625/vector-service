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
