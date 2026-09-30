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

All methods are synchronous and blocking. Async routes dispatch them via
the isolated store thread pool — ``run_in_store`` in
:mod:`vector_service.core.threadpools` — never inline on the event loop.
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
    # Per-field schema, one entry per scalar + the vector field. Each entry is
    # a dict shaped like
    # ``{"name", "dtype", "is_primary", "dim"?, "max_length"?, ...}``; keys
    # beyond what the consumer needs are silently dropped at the schema
    # boundary (Pydantic v2 ``extra='ignore'``).
    fields: list[dict] = field(default_factory=list)
    # Indexes built on vector fields. Each entry is
    # ``{"field_name", "metric_type", "index_type", "params": dict}``.
    indexes: list[dict] = field(default_factory=list)


@dataclass
class FieldSpec:
    """One field in a collection schema.

    ``dtype`` is a Milvus DataType name (``varchar`` / ``int64`` /
    ``float`` / ... / ``float_vector`` / ``sparse_float_vector``).
    ``dim`` is required for ``float_vector``. ``is_primary`` marks the
    (single) primary key; ``max_length`` is required for ``varchar``.
    The dense vector field is identified by ``dtype == "float_vector"``.

    The derived index stores no text and runs no analyzer: sparse
    vectors are produced client-side and written as data.
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
        extra_vector_fields: list[FieldSpec] | None = None,
    ) -> CollectionInfo:
        """Create a collection with a caller-defined schema.

        Implementations must enforce at minimum:

        - exactly one VARCHAR primary key in ``scalar_fields``
        - one primary FLOAT_VECTOR entry (``vector_field``), plus any
          additional FLOAT_VECTOR entries in ``extra_vector_fields``
        - one index covering the primary vector field (auto-default if
          ``indexes`` is empty)

        Raises :class:`DatabaseNotFound` or :class:`CollectionAlreadyExists`.
        """

    @abstractmethod
    def drop_collection(self, database: str, name: str) -> None:
        """Delete a collection and all its vectors."""

    @abstractmethod
    def collection_info(self, database: str, name: str) -> CollectionInfo:
        """Return metadata: at least ``dim``, ``metric``, ``count``."""

    @abstractmethod
    def create_index(
        self,
        database: str,
        collection: str,
        *,
        field_name: str,
        metric_type: str = "cosine",
        index_type: str = "HNSW",
        params: dict | None = None,
    ) -> None:
        """Create or rebuild a vector index on ``field_name``.

        Backends that don't allow rebuilding in place (e.g. Milvus)
        will internally drop the existing index first. Parameters
        mirror :class:`IndexParamSpec` so a single object can be
        forwarded between the create-time list and the runtime
        manage endpoint.
        """

    @abstractmethod
    def drop_index(
        self,
        database: str,
        collection: str,
        *,
        field_name: str,
    ) -> None:
        """Drop the vector index on ``field_name``.

        Implementations may treat a missing index as a no-op (the
        collection is still queryable, just without acceleration).
        """

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
        extra_vectors: dict[str, list[list[float]]] | None = None,
        sparse_vectors: dict[str, list[dict]] | None = None,
    ) -> None:
        """Insert or update rows. ``fields`` is per-row scalar values
        keyed by scalar field name (excluding the primary key, which is in
        ``ids``). ``extra_vectors`` maps an extra dense vector field name
        to one vector per row; ``sparse_vectors`` maps a sparse vector
        field name to one sparse dict (``{term_id: weight}``) per row."""

    @abstractmethod
    def delete(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str] | None = None,
        *,
        filter_expr: str | None = None,
    ) -> int:
        """Delete rows by primary key list, or by a filter expression.

        Exactly one of ``ids`` / ``filter_expr`` must be provided.
        ``primary_field`` is required even for the filter-based path
        so backend adapters that need to know which scalar to operate
        on (e.g. for write-protect guards) can validate up front.

        Returns the number of rows actually removed.
        """

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
    def browse(
        self,
        database: str,
        collection: str,
        primary_field: str,
        *,
        limit: int = 20,
        offset: int = 0,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """List a slice of rows without specifying primary keys.

        ``filter_expr`` is an optional backend-native boolean expression
        (e.g. ``category == 'mouse' and price < 100`` for Milvus).
        ``output_fields`` selects which scalar fields to materialise; when
        ``None``, every scalar field declared on the collection is
        returned. The vector field is never returned regardless.

        Rows are returned in the backend's natural order; backends that
        support it (Milvus) page via ``offset``. The shape of each dict
        matches :meth:`get`:
        ``{"id": <primary>, "vector": None, "fields": {scalar: value}}``.
        """

    @abstractmethod
    def count_rows(
        self,
        database: str,
        collection: str,
        *,
        filter_expr: str | None = None,
    ) -> int:
        """Return an authoritative, tombstone-aware row count.

        A live ``count(*)`` query reflects deletes immediately, unlike
        the metadata count in :meth:`collection_info` (which Milvus
        does not adjust until compaction). When ``filter_expr`` is
        given, only matching rows are counted — the correct denominator
        for paged browse results.
        """

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

    @abstractmethod
    def search_sparse(
        self,
        database: str,
        collection: str,
        vector_field: str,
        query_sparse: dict,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[Hit]:
        """Search a sparse vector field with a client-encoded sparse query.

        ``query_sparse`` is the ``{term_id: weight}`` vector produced by
        the retrieval-side sparse encoder (client-side BM25); the index
        is a ``SPARSE_INVERTED_INDEX`` with ``IP`` metric.
        """

    # ---- lifecycle ----

    def close(self) -> None:
        """Release backend resources. Default: no-op."""