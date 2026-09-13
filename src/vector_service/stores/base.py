"""VectorStore universal interface.

This module defines the **universal** contract every vector-store backend
exposes to vector-service. The production backend shipped in ``stores/``
is :class:`vector_service.stores.milvus.MilvusStore`, which talks
directly to a Milvus server via :mod:`pymilvus`. To add another backend,
drop a new module under ``stores/`` implementing this ABC and register it
in :func:`vector_service.stores.registry.build_store`.

Two-level hierarchy:

- **Database** — a named, isolated namespace (analogous to a schema /
  project / database in different vector stores).
- **Collection** — a vector index living inside a single database. All
  vector operations are scoped by ``(database, collection)``.

All methods are synchronous and blocking. Async dispatch is the caller's
responsibility (use :func:`asyncio.get_running_loop().run_in_executor`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    id: str
    score: float
    fields: dict = field(default_factory=dict)


@dataclass
class DatabaseInfo:
    name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CollectionInfo:
    database: str
    name: str
    dim: int
    metric: str
    count: int
    primary_field: str = "id"
    vector_field: str = "vector"
    metadata: dict = field(default_factory=dict)


@dataclass
class FieldSpec:
    """One field in a collection schema.

    ``dtype`` is a Milvus DataType name (``varchar`` / ``int64`` / ``float``
    / ... / ``float_vector``). ``dim`` is required for ``float_vector``.
    ``is_primary`` marks the (single) primary key; ``max_length`` is
    required for ``varchar``. The vector field is identified by
    ``dtype == "float_vector"``.
    """

    name: str
    dtype: str
    is_primary: bool = False
    dim: int | None = None
    max_length: int | None = None
    nullable: bool = False
    default_value: Any | None = None


@dataclass
class IndexSpec:
    """Index parameters for one vector field."""

    field_name: str
    metric_type: str = "cosine"
    index_type: str = "HNSW"
    params: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Universal vector-store interface.

    Production backend: :class:`vector_service.stores.milvus.MilvusStore`.
    """

    backend_name: str = "generic"

    # ---- databases ----

    @abstractmethod
    def list_databases(self) -> list[str]:
        """Return all database names known to the store."""

    @abstractmethod
    def create_database(self, name: str, **backend_opts: Any) -> DatabaseInfo:
        """Create a new database. Raises :class:`DatabaseAlreadyExists`."""

    @abstractmethod
    def drop_database(self, name: str) -> None:
        """Delete a database and every collection inside it. Raises :class:`DatabaseNotFound`."""

    @abstractmethod
    def database_info(self, name: str) -> DatabaseInfo:
        """Return metadata for a single database. Raises :class:`DatabaseNotFound`."""

    # ---- collections ----

    @abstractmethod
    def list_collections(self, database: str) -> list[str]:
        """Return all collection names in the given database."""

    @abstractmethod
    def create_collection(
        self,
        database: str,
        name: str,
        primary_field: str,
        vector_field: FieldSpec,
        scalar_fields: list[FieldSpec],
        indexes: list[IndexSpec] | None = None,
    ) -> CollectionInfo:
        """Create a collection with a caller-defined schema.

        Implementations must enforce at minimum:

        - exactly one VARCHAR primary key in ``scalar_fields``
        - exactly one FLOAT_VECTOR entry in ``vector_field``
        - one index covering the vector field (auto-default if ``indexes`` is empty)

        Raises :class:`DatabaseNotFound` or :class:`CollectionAlreadyExists`.
        """

    @abstractmethod
    def drop_collection(self, database: str, name: str) -> None:
        """Delete a collection and all its vectors."""

    @abstractmethod
    def collection_info(self, database: str, name: str) -> CollectionInfo:
        """Return metadata: at least ``dim``, ``metric``, ``count``."""

    # ---- vectors ----

    @abstractmethod
    def upsert(
        self,
        database: str,
        collection: str,
        primary_field: str,
        vector_field: str,
        ids: list[str],
        vectors: list[list[float]],
        fields: list[dict] | None = None,
    ) -> None:
        """Insert or update rows. ``fields`` is per-row scalar values
        keyed by scalar field name (excluding the primary key, which is in
        ``ids``)."""

    @abstractmethod
    def delete(self, database: str, collection: str, primary_field: str, ids: list[str]) -> None:
        """Delete rows by primary key."""

    @abstractmethod
    def get(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str],
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """Fetch rows by primary key. Returns dicts with at least
        ``id`` and any requested ``output_fields``."""

    @abstractmethod
    def search(
        self,
        database: str,
        collection: str,
        vector_field: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[Hit]:
        """Top-k nearest neighbours in the given (database, collection)."""

    # ---- lifecycle ----

    def close(self) -> None:
        """Release backend resources. Default: no-op."""