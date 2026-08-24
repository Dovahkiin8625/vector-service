"""VectorStore abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base for vector store backends.

    Concrete subclasses must implement all abstract methods and set
    `backend_name` (str). The `backend` property exposes the native
    client/connection for backend-specific operations.
    """

    backend_name: str

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dim: int,
        *,
        metric: str = "cosine",
        **backend_opts: Any,
    ) -> None:
        """Create a collection with the given dimension and metric.

        Raises:
            CollectionAlreadyExists: if a collection with this name exists.
            StoreError: on backend failure.
        """

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Raises:
            CollectionNotFound: if no such collection.
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return all collection names in this store."""

    @abstractmethod
    def collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata: at least `dim`, `metric`, `count`."""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors with given ids and metadata.

        Raises:
            CollectionNotFound, DimensionMismatch, StoreError.
        """

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by id. Silently ignores missing ids."""

    @abstractmethod
    def get(self, collection: str, ids: list[str]) -> list[dict]:
        """Fetch vectors by id.

        Returns a list of `{id, vector, metadata}` dicts. Missing ids
        are skipped (not error).
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[Hit]:
        """Top-k nearest neighbors.

        Raises:
            CollectionNotFound, DimensionMismatch, StoreError.
        """

    @property
    def backend(self) -> Any:
        """Native client/connection. Use only for backend-specific ops."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources. Default: no-op."""