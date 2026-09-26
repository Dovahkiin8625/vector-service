"""Milvus (standalone / cluster) implementation of VectorStore.

Connects to a Milvus server via :class:`vector_service.stores._milvus_adapter.MilvusAdapter`,
which wraps the post-2.4 ``MilvusClient`` API. This module is the
production backend used when ``VS_VECTOR_STORE_BACKEND=milvus``.

Two-level hierarchy:

- **Database** — Milvus' native database API (2.4+).
- **Collection** — schema is **fully caller-defined**: any number of
  scalar fields (one of which is the VARCHAR primary key), exactly one
  FLOAT_VECTOR field, and any number of indexes (typically one on the
  vector field). The service does not inject any field.

Construction does **not** open the gRPC connection — the handshake is
deferred to the first call (``_ensure_connected``) so the service can
boot when Milvus is briefly unreachable, and so unit tests can
instantiate the store against any URL.
"""
from __future__ import annotations

import re
from typing import Any

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import (
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.stores._milvus_adapter import MilvusAdapter
from vector_service.stores.base import (
    CollectionInfo,
    DatabaseInfo,
    FieldSpec,
    Hit,
    IndexSpec,
    VectorStore,
)

_VALID_METRICS = {"cosine", "ip", "l2", "bm25"}
_VALID_DTYPES = {
    "bool", "int8", "int16", "int32", "int64", "float", "double",
    "varchar", "json", "sparse_float_vector",
}
# System-wide convention for the BM25 sparse field name. The ingest v2
# helper names it ``sparse`` (``_SPARSE_FIELD`` in ``api/ingest.py``);
# this store-level pass needs no view of the scalar dtypes, and the
# adapter re-validates the field's dtype when creating the collection.
_SPARSE_FIELD_NAME = "sparse"
_ID_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_name(value: str, kind: str) -> None:
    if not _ID_RX.match(value):
        raise StoreError(
            f"invalid {kind} name {value!r}: must match {_ID_RX.pattern}"
        )


def _validate_schema(
    primary_field: str,
    vector_field: FieldSpec,
    scalar_fields: list[FieldSpec],
) -> None:
    if vector_field.dtype != "float_vector":
        raise StoreError(
            f"vector_field.dtype must be 'float_vector', got {vector_field.dtype!r}"
        )
    if vector_field.dim is None or vector_field.dim < 1:
        raise StoreError("vector_field.dim must be a positive integer")

    names = [f.name for f in scalar_fields]
    if vector_field.name in names:
        raise StoreError(
            f"vector field name {vector_field.name!r} collides with a scalar field"
        )

    primary = [f for f in scalar_fields if f.is_primary]
    if len(primary) != 1:
        raise StoreError("exactly one scalar field must have is_primary=true")
    if primary[0].name != primary_field:
        raise StoreError(
            f"primary_field={primary_field!r} does not match the is_primary field "
            f"{primary[0].name!r}"
        )
    if primary[0].dtype != "varchar":
        raise StoreError("primary key field must have dtype='varchar'")
    if primary[0].max_length is None or primary[0].max_length < 1:
        raise StoreError("primary key field must set max_length >= 1")

    if len(set(names)) != len(names):
        raise StoreError("scalar field names must be unique")

    for f in scalar_fields:
        if f.enable_analyzer and f.dtype != "varchar":
            raise StoreError(
                f"field {f.name!r}: enable_analyzer requires dtype='varchar'"
            )
        if f.dtype not in _VALID_DTYPES:
            raise StoreError(
                f"unsupported scalar dtype {f.dtype!r}; expected one of "
                f"{sorted(_VALID_DTYPES)}"
            )
        if f.dtype == "varchar" and (f.max_length is None or f.max_length < 1):
            raise StoreError(f"varchar field {f.name!r} must set max_length >= 1")


def _validate_indexes(
    vector_field: FieldSpec,
    indexes: list[IndexSpec],
) -> None:
    if not indexes:
        raise StoreError("at least one index covering the vector field is required")
    for ip in indexes:
        if ip.field_name == vector_field.name:
            allowed = _VALID_METRICS - {"bm25"}
        elif ip.field_name == _SPARSE_FIELD_NAME:
            allowed = {"bm25"}
        else:
            raise StoreError(
                f"index target {ip.field_name!r} is not the vector field "
                f"{vector_field.name!r}"
            )
        if ip.metric_type not in allowed:
            raise StoreError(
                f"index on {ip.field_name!r} got metric_type {ip.metric_type!r}; "
                f"expected one of {sorted(allowed)}"
            )


class MilvusStore(VectorStore):
    """Direct Milvus-backed :class:`VectorStore`.

    All pymilvus specifics live in :class:`MilvusAdapter`. This class is
    a thin mapping from the universal :class:`VectorStore` contract to
    adapter calls, plus the schema/index validation rules that callers
    must satisfy.
    """

    backend_name = "milvus"

    def __init__(
        self,
        uri: str | None = None,
        *,
        user: str | None = None,
        password: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        alias: str = "default",
        settings: Settings | None = None,
    ):
        s = settings or get_settings()
        self._uri = uri or s.milvus_uri
        self._adapter = MilvusAdapter(
            uri=self._uri,
            user=user if user is not None else (s.milvus_user or ""),
            password=password if password is not None else (s.milvus_password or ""),
            token=token if token is not None else (s.milvus_token or ""),
            timeout=float(timeout) if timeout is not None else float(s.milvus_timeout),
            db_name="default",
            alias=alias,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        # Adapter does its own lazy connect; expose it through the same
        # surface so callers (lifespan, /readyz) can probe.
        self._adapter._ensure_connected()  # noqa: SLF001 — internal hook

    def close(self) -> None:
        self._adapter.close()

    @property
    def uri(self) -> str:
        return self._uri

    @property
    def backend(self) -> Any:
        # Kept for backward compatibility with the original
        # ``/backend/raw`` debug introspection endpoint. No callers
        # in the current codebase reach into ``store.backend.*``; the
        # proxy here just delegates to ``self._adapter``.
        return _MilvusBackendProxy(self._adapter)

    # ------------------------------------------------------------------
    # database
    # ------------------------------------------------------------------

    def list_databases(self) -> list[str]:
        return self._adapter.list_databases()

    def create_database(self, name: str, **backend_opts: Any) -> DatabaseInfo:
        _validate_name(name, "database")
        self._adapter.create_database(name)
        return DatabaseInfo(name=name, metadata=backend_opts or {})

    def drop_database(self, name: str) -> None:
        _validate_name(name, "database")
        self._adapter.drop_database(name)

    def database_info(self, name: str) -> DatabaseInfo:
        _validate_name(name, "database")
        if name not in self.list_databases():
            raise DatabaseNotFound(
                f"database {name!r} does not exist", name=name
            )
        return DatabaseInfo(name=name, metadata={})

    # ------------------------------------------------------------------
    # collections
    # ------------------------------------------------------------------

    def list_collections(self, database: str) -> list[str]:
        _validate_name(database, "database")
        return self._adapter.list_collections(database)

    def create_collection(
        self,
        database: str,
        name: str,
        primary_field: str,
        vector_field: FieldSpec,
        scalar_fields: list[FieldSpec],
        indexes: list[IndexSpec] | None = None,
    ) -> CollectionInfo:
        _validate_name(database, "database")
        _validate_name(name, "collection")
        if indexes is None:
            indexes = []
        _validate_schema(primary_field, vector_field, scalar_fields)
        _validate_indexes(vector_field, indexes)

        if database not in self.list_databases():
            raise DatabaseNotFound(
                f"database {database!r} does not exist", name=database
            )

        try:
            self._adapter.create_collection(
                database=database,
                name=name,
                primary_field=primary_field,
                vector_field_name=vector_field.name,
                vector_dim=int(vector_field.dim or 0),
                vector_metric=(indexes[0].metric_type if indexes else "cosine"),
                scalar_fields=[
                    {
                        "name": f.name,
                        "dtype": f.dtype,
                        "is_primary": bool(f.is_primary),
                        "max_length": f.max_length,
                        "nullable": bool(f.nullable),
                        "default_value": f.default_value,
                        "enable_analyzer": bool(f.enable_analyzer),
                        "analyzer": dict(f.analyzer) if f.analyzer else None,
                    }
                    for f in scalar_fields
                ],
                indexes=[
                    {
                        "field_name": ip.field_name,
                        "metric_type": ip.metric_type,
                        "index_type": ip.index_type,
                        "params": dict(ip.params or {}),
                    }
                    for ip in indexes
                ],
            )
        except (CollectionAlreadyExists, StoreError, DatabaseNotFound):
            raise
        except Exception as e:
            raise StoreError(f"create_collection failed: {e}") from e

        primary_index = indexes[0]
        # Build a response-shape that mirrors what ``describe_collection``
        # would have returned for the freshly-created collection, so the
        # caller of POST /collections sees the same fields/indexes payload
        # as GET /collections/{name}.
        fields_payload: list[dict] = [
            {
                "name": f.name,
                "dtype": f.dtype,
                "is_primary": bool(f.is_primary),
                "dim": f.dim,
                "max_length": f.max_length,
                "nullable": bool(f.nullable),
                "default_value": f.default_value,
                "enable_analyzer": bool(f.enable_analyzer),
                "analyzer": dict(f.analyzer) if f.analyzer else None,
            }
            for f in scalar_fields
        ]
        fields_payload.append(
            {
                "name": vector_field.name,
                "dtype": "float_vector",
                "is_primary": False,
                "dim": int(vector_field.dim or 0),
            }
        )
        indexes_payload: list[dict] = [
            {
                "field_name": ip.field_name,
                "metric_type": ip.metric_type,
                "index_type": ip.index_type,
                "params": dict(ip.params or {}),
            }
            for ip in indexes
        ]
        return CollectionInfo(
            database=database,
            name=name,
            dim=int(vector_field.dim or 0),
            metric=primary_index.metric_type,
            count=0,
            primary_field=primary_field,
            vector_field=vector_field.name,
            metadata={},
            fields=fields_payload,
            indexes=indexes_payload,
        )

    def drop_collection(self, database: str, name: str) -> None:
        _validate_name(database, "database")
        _validate_name(name, "collection")
        self._adapter.drop_collection(database, name)

    def collection_info(self, database: str, name: str) -> CollectionInfo:
        _validate_name(database, "database")
        _validate_name(name, "collection")
        schema = self._adapter.describe_collection(database, name)
        return CollectionInfo(
            database=database,
            name=name,
            dim=int(schema["dim"]),
            metric=str(schema["metric"]),
            count=int(schema["count"]),
            primary_field=str(schema["primary_field"]),
            vector_field=str(schema["vector_field"]),
            metadata={},
            fields=list(schema.get("fields") or []),
            indexes=list(schema.get("indexes") or []),
        )

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
        self._adapter.create_index(
            database=database,
            collection=collection,
            field_name=field_name,
            metric_type=metric_type,
            index_type=index_type,
            params=params,
        )

    def drop_index(
        self,
        database: str,
        collection: str,
        *,
        field_name: str,
    ) -> None:
        self._adapter.drop_index(
            database=database,
            collection=collection,
            field_name=field_name,
        )

    # ------------------------------------------------------------------
    # vectors
    # ------------------------------------------------------------------

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
        self._adapter.upsert(
            database=database,
            collection=collection,
            primary_field=primary_field,
            vector_field=vector_field,
            ids=ids,
            vectors=vectors,
            fields=fields,
        )

    def delete(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str] | None = None,
        *,
        filter_expr: str | None = None,
    ) -> int:
        return self._adapter.delete(
            database=database,
            collection=collection,
            primary_field=primary_field,
            ids=ids,
            filter_expr=filter_expr,
        )

    def get(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str],
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        items = self._adapter.get(
            database=database,
            collection=collection,
            primary_field=primary_field,
            ids=ids,
            output_fields=output_fields,
        )
        # Normalise to "vector is null" semantics — Milvus `get` doesn't
        # return raw vectors, so leave that field out.
        return [
            {"id": it["id"], "vector": None, "fields": it["fields"]}
            for it in items
        ]

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
        rows = self._adapter.browse(
            database=database,
            collection=collection,
            primary_field=primary_field,
            limit=limit,
            offset=offset,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        # Same "vector is null" contract as ``get``; the adapter already
        # strips the vector field, so we only need to wrap.
        return [
            {"id": it["id"], "vector": None, "fields": it["fields"]}
            for it in rows
        ]

    def count_rows(
        self,
        database: str,
        collection: str,
        *,
        filter_expr: str | None = None,
    ) -> int:
        """Live ``count(*)`` — see :meth:`MilvusAdapter.count`.

        Unlike ``collection_info().count`` (segment metadata, stale
        until compaction), this reflects deletes immediately and is
        what drives the dashboard browse pager's total.
        """
        return self._adapter.count(
            database=database,
            collection=collection,
            filter_expr=filter_expr,
        )

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
        hits = self._adapter.search(
            database=database,
            collection=collection,
            vector_field=vector_field,
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return [Hit(id=h["id"], score=h["score"], fields=h["fields"]) for h in hits]

    def search_text(
        self,
        database: str,
        collection: str,
        sparse_field: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[Hit]:
        hits = self._adapter.search_text(
            database=database,
            collection=collection,
            sparse_field=sparse_field,
            query_text=query_text,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return [Hit(id=h["id"], score=h["score"], fields=h["fields"]) for h in hits]


class _MilvusBackendProxy:
    """Escape hatch for ``/backend/raw`` introspection."""

    def __init__(self, adapter: MilvusAdapter) -> None:
        self._adapter = adapter

    def list_databases(self) -> list[str]:
        return self._adapter.list_databases()

    def list_collections(self, database: str) -> list[str]:
        return self._adapter.list_collections(database)

    def describe_collection(self, database: str, name: str):
        return self._adapter.describe_collection(database, name)
