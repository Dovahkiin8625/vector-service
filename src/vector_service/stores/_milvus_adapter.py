"""Adapter around pymilvus ``MilvusClient`` (the post-2.4 non-ORM API).

This module is the only place that talks to pymilvus. ``MilvusStore``
delegates every operation here so we can keep the public ``VectorStore``
interface clean and isolate the SDK quirks (error codes, has-collection
pre-checks, scalar/vector type mapping, ``get`` returning string reprs
on 2.4.x, ...) behind a single boundary.

Two pymilvus subsystems are in play:

- ``MilvusClient`` — collection / vector CRUD, search / query, schema
  introspection, ``using_database``. This is the API that pymilvus
  will keep in 3.x.
- ``pymilvus.db`` — database-level operations (``list_database`` /
  ``create_database`` / ``drop_database``). Not yet on ``MilvusClient``
  on 2.4, so we open a separate ORM connection just for those calls.

The adapter enforces its own locking for ``using_database`` because
pymilvus alias / client state is process-global; concurrent FastAPI
worker threads must not stomp on each other.

Schema-cache invariants:
- ``has_collection`` and ``describe_collection`` are read for every
  upsert/get/delete/search — Milvus schema is immutable after
  ``create_collection``, so the cache TTL is set well above the
  expected re-deploy cadence (5 minutes). create_collection /
  drop_collection / drop_database actively invalidate the affected
  entries on success or failure.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    connections,
    db as milvus_db,
)
from pymilvus.exceptions import MilvusException

from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.core.logging import get_logger

# TTL for ``has_collection`` / ``describe_collection`` caches. Milvus
# schema is immutable after ``create_collection`` completes, so the
# only writers to a cached key are the create/drop hooks inside this
# adapter; we still keep a TTL as a defence-in-depth measure (e.g.
# if an operator runs ``ALTER TABLE`` directly against the backend
# outside this service).
_SCHEMA_CACHE_TTL_S = 300.0

log = get_logger(__name__)

# pymilvus error codes (stable across 2.4.x; see pymilvus/exception.py)
_ERR_DB_NOT_FOUND = 800
_ERR_COLLECTION_NOT_FOUND = 800  # collection not found also reports 800 on 2.4
_ERR_DB_ALREADY_EXISTS = 1100  # "database already exist" in 2.4.x
_ERR_NOT_CONNECTED = 1

# Scalar dtype map (public name → pymilvus DataType). Float_vector is
# reserved for the vector field and is built directly via DataType.
_SCALAR_DTYPE = {
    "bool": DataType.BOOL,
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "varchar": DataType.VARCHAR,
    "json": DataType.JSON,
    "sparse_float_vector": DataType.SPARSE_FLOAT_VECTOR,
}

_METRIC = {"cosine": "COSINE", "ip": "IP", "l2": "L2", "bm25": "BM25"}
_INV_METRIC = {v: k for k, v in _METRIC.items()}


class MilvusAdapter:
    """Thin wrapper that owns one ``MilvusClient`` plus an ORM connection.

    Construction only stores configuration. The actual gRPC handshake is
    deferred to :meth:`_ensure_connected`; tests and the service can
    both instantiate against any URL and connect on demand.
    """

    def __init__(
        self,
        uri: str,
        *,
        user: str = "",
        password: str = "",
        token: str = "",
        timeout: float = 30.0,
        db_name: str = "default",
        alias: str = "default",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._token = token
        self._timeout = float(timeout)
        self._alias = alias
        self._bound_db = db_name

        self._client: MilvusClient | None = None
        self._lock = threading.Lock()
        self._orm_connected = False
        # Schema cache: ``(database, collection)`` → ``(expires_at, value)``.
        # Reads happen on every upsert/get/delete/search; writes happen
        # only inside ``create_collection`` / ``drop_collection`` /
        # ``drop_database`` (invalidation). Guarded by ``self._lock`` so
        # concurrent FastAPI workers cannot observe a half-updated entry.
        self._schema_cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._has_cache: dict[tuple[str, str], tuple[float, bool]] = {}

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        kwargs: dict[str, Any] = {"uri": self._uri, "timeout": self._timeout, "db_name": self._bound_db}
        if self._token:
            kwargs["token"] = self._token
        else:
            if self._user:
                kwargs["user"] = self._user
            if self._password:
                kwargs["password"] = self._password
        try:
            self._client = MilvusClient(**kwargs)
        except Exception as e:
            raise BackendError(f"failed to connect to Milvus at {self._uri}: {e}") from e

        # Separate ORM connection for db.* helpers. Sharing with MilvusClient
        # is unsupported because db.* uses the ORM registry, while MilvusClient
        # uses its own connection pool.
        try:
            connections.connect(alias=self._alias, uri=self._uri, timeout=self._timeout,
                                user=self._user or "", password=self._password or "",
                                token=self._token or "")
            self._orm_connected = True
        except Exception:
            self._orm_connected = False

    def _using_db(self, db_name: str) -> None:
        """Bind the client to a database. Serialised — pymilvus alias
        state is process-global and concurrent FastAPI workers must not
        race each other."""
        with self._lock:
            if self._client is not None and self._bound_db == db_name:
                return
            if self._client is not None:
                try:
                    self._client.using_database(db_name)
                    self._bound_db = db_name
                    return
                except MilvusException as e:
                    if e.code == _ERR_DB_NOT_FOUND:
                        raise DatabaseNotFound(
                            f"database {db_name!r} does not exist", name=db_name
                        ) from e
                    raise BackendError(f"failed to switch to database {db_name!r}: {e}") from e
                except Exception as e:
                    raise BackendError(f"failed to switch to database {db_name!r}: {e}") from e

            # client not connected yet — first call gets to set bound_db
            self._bound_db = db_name

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._orm_connected:
            try:
                connections.disconnect(self._alias)
            except Exception:
                pass
            self._orm_connected = False

    # ------------------------------------------------------------------
    # databases
    # ------------------------------------------------------------------

    def list_databases(self) -> list[str]:
        self._ensure_connected()
        try:
            return sorted(milvus_db.list_database(using=self._alias))
        except Exception as e:
            raise BackendError(f"list_databases failed: {e}") from e

    def create_database(self, name: str) -> None:
        self._ensure_connected()
        if name in self.list_databases():
            raise DatabaseAlreadyExists(
                f"database {name!r} already exists", name=name
            )
        try:
            milvus_db.create_database(name, using=self._alias)
        except MilvusException as e:
            msg = str(e).lower()
            if e.code == _ERR_DB_ALREADY_EXISTS or "already exist" in msg:
                raise DatabaseAlreadyExists(
                    f"database {name!r} already exists", name=name
                ) from e
            raise BackendError(f"create_database failed: {e}") from e
        except Exception as e:
            if "already exist" in str(e).lower():
                raise DatabaseAlreadyExists(
                    f"database {name!r} already exists", name=name
                ) from e
            raise BackendError(f"create_database failed: {e}") from e

    def drop_database(self, name: str) -> None:
        self._ensure_connected()
        # pymilvus 2.4's db.drop_database is a no-op when the database
        # doesn't exist; check first so callers get a clean 404.
        if name not in self.list_databases():
            raise DatabaseNotFound(
                f"database {name!r} does not exist", name=name
            )
        # Milvus refuses drop_database on a non-empty database (error
        # 1100, "must drop all collections before drop database"). Pre-
        # clean: switch into the database, list collections, and drop
        # each one. Any failure aborts the whole operation so the
        # caller sees which collection couldn't be deleted and the
        # database stays inspectable for retry.
        self._using_db(name)
        try:
            colls = list(self._client.list_collections())
        except MilvusException as e:
            raise BackendError(f"list_collections failed for {name!r}: {e}") from e
        except Exception as e:
            raise BackendError(f"list_collections failed for {name!r}: {e}") from e
        for c in colls:
            try:
                self._client.drop_collection(c)
            except MilvusException as e:
                raise BackendError(
                    f"drop_collection {c!r} in database {name!r} failed: {e}"
                ) from e
            except Exception as e:
                raise BackendError(
                    f"drop_collection {c!r} in database {name!r} failed: {e}"
                ) from e
        try:
            milvus_db.drop_database(name, using=self._alias)
        except MilvusException as e:
            if e.code == _ERR_DB_NOT_FOUND:
                raise DatabaseNotFound(
                    f"database {name!r} does not exist", name=name
                ) from e
            raise BackendError(f"drop_database failed: {e}") from e
        except Exception as e:
            if "not found" in str(e).lower() or "not exist" in str(e).lower():
                raise DatabaseNotFound(
                    f"database {name!r} does not exist", name=name
                ) from e
            raise BackendError(f"drop_database failed: {e}") from e

    # ------------------------------------------------------------------
    # collections
    # ------------------------------------------------------------------

    def list_collections(self, database: str) -> list[str]:
        self._ensure_connected()
        self._using_db(database)
        try:
            return sorted(self._client.list_collections())
        except MilvusException as e:
            if e.code == _ERR_DB_NOT_FOUND:
                raise DatabaseNotFound(
                    f"database {database!r} does not exist", name=database
                ) from e
            raise BackendError(
                f"list_collections failed for database {database!r}: {e}"
            ) from e
        except Exception as e:
            raise BackendError(
                f"list_collections failed for database {database!r}: {e}"
            ) from e

    def has_collection(self, database: str, name: str) -> bool:
        """``has_collection`` with a TTL cache.

        Milvus collection metadata does not change between
        ``create_collection`` and ``drop_collection``; reading it on
        every upsert/get/delete/search call was responsible for the
        N+1 RPC pattern that dominated the request path. Caching
        here drops those RPCs to zero on the hot path. Cache is
        invalidated by :meth:`_invalidate_collection`.
        """
        key = (database, name)
        now = time.monotonic()
        with self._lock:
            cached = self._has_cache.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]
        self._ensure_connected()
        self._using_db(database)
        try:
            present = bool(self._client.has_collection(name))
        except Exception:
            present = False
        with self._lock:
            self._has_cache[key] = (now + _SCHEMA_CACHE_TTL_S, present)
        return present

    def _invalidate_collection(self, database: str, name: str) -> None:
        """Drop cached metadata for ``(database, name)``."""
        key = (database, name)
        with self._lock:
            self._has_cache.pop(key, None)
            self._schema_cache.pop(key, None)

    def _invalidate_database(self, database: str) -> None:
        """Drop cached metadata for every collection in ``database``."""
        with self._lock:
            for key in list(self._has_cache):
                if key[0] == database:
                    self._has_cache.pop(key, None)
            for key in list(self._schema_cache):
                if key[0] == database:
                    self._schema_cache.pop(key, None)

    def create_collection(
        self,
        database: str,
        name: str,
        primary_field: str,
        vector_field_name: str,
        vector_dim: int,
        vector_metric: str,
        scalar_fields: list[dict[str, Any]],
        indexes: list[dict[str, Any]],
    ) -> None:
        self._ensure_connected()
        self._using_db(database)
        if vector_metric not in _METRIC:
            raise StoreError(
                f"unsupported metric {vector_metric!r}; expected one of {sorted(_METRIC)}"
            )
        sparse_names = {
            f["name"] for f in scalar_fields if f["dtype"] == "sparse_float_vector"
        }
        for ip in indexes:
            target = ip.get("field_name")
            metric = ip.get("metric_type")
            if target == vector_field_name:
                allowed = ("cosine", "ip", "l2")
            elif target in sparse_names:
                allowed = ("bm25",)
            else:
                raise StoreError(
                    f"index target {target!r} is neither the vector field "
                    f"{vector_field_name!r} nor a sparse field {sorted(sparse_names)}"
                )
            if metric not in allowed:
                raise StoreError(
                    f"index on {target!r} got metric {metric!r}; "
                    f"expected one of {allowed}"
                )

        try:
            schema = self._client.create_schema(
                auto_id=False, enable_dynamic_field=False,
            )
            analyzed_varchars: list[str] = []
            for f in scalar_fields:
                dtype = _SCALAR_DTYPE.get(f["dtype"])
                if dtype is None:
                    raise StoreError(f"unsupported scalar dtype {f['dtype']!r}")
                kwargs: dict[str, Any] = {
                    "is_primary": bool(f.get("is_primary", False)),
                }
                if f["dtype"] == "varchar":
                    if not f.get("max_length"):
                        raise StoreError(
                            f"varchar field {f['name']!r} must set max_length"
                        )
                    kwargs["max_length"] = int(f["max_length"])
                    if f.get("enable_analyzer"):
                        kwargs["enable_analyzer"] = True
                        kwargs["analyzer_params"] = dict(f.get("analyzer") or {})
                        kwargs["enable_match"] = True
                        analyzed_varchars.append(f["name"])
                elif f["dtype"] == "sparse_float_vector":
                    if f.get("enable_analyzer"):
                        raise StoreError(
                            f"sparse field {f['name']!r} cannot have enable_analyzer"
                        )
                else:
                    if f.get("enable_analyzer"):
                        raise StoreError(
                            f"field {f['name']!r}: analyzer requires dtype varchar"
                        )
                if f.get("nullable"):
                    kwargs["nullable"] = True
                if f.get("default_value") is not None:
                    kwargs["default_value"] = f["default_value"]
                schema.add_field(f["name"], dtype, **kwargs)
            schema.add_field(vector_field_name, DataType.FLOAT_VECTOR, dim=vector_dim)
            if analyzed_varchars and sparse_names:
                sparse_list = sorted(sparse_names)
                schema.add_function(Function(
                    name=f"bm25_{sparse_list[0]}",
                    function_type=FunctionType.BM25,
                    input_field_names=analyzed_varchars,
                    output_field_names=sparse_list,
                ))
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"failed to build schema: {e}") from e

        try:
            ip = self._client.prepare_index_params()
            for ix in indexes:
                ip.add_index(
                    field_name=ix["field_name"],
                    index_type=ix["index_type"],
                    metric_type=_METRIC[ix["metric_type"]],
                    params=dict(ix.get("params") or {}),
                )
        except Exception as e:
            raise StoreError(f"failed to build index params: {e}") from e

        try:
            self._client.create_collection(
                collection_name=name, schema=schema, index_params=ip,
            )
        except MilvusException as e:
            # The backend is authoritative for the duplicate check.
            self._invalidate_collection(database, name)
            if "already exist" in str(e).lower():
                raise CollectionAlreadyExists(
                    f"collection {name!r} already exists in database {database!r}"
                ) from e
            raise BackendError(
                f"create_collection failed for {database!r}/{name!r}: {e}"
            ) from e
        except Exception as e:
            # Drop any cache entry an earlier read may have populated.
            self._invalidate_collection(database, name)
            raise BackendError(
                f"create_collection failed for {database!r}/{name!r}: {e}"
            ) from e
        # Successful create: cache will be re-populated lazily by the
        # next has_collection / describe_collection read.
        self._invalidate_collection(database, name)

    def drop_collection(self, database: str, name: str) -> None:
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, name):
            raise CollectionNotFound(
                f"collection {name!r} does not exist in database {database!r}"
            )
        try:
            self._client.drop_collection(name)
        except Exception as e:
            self._invalidate_collection(database, name)
            raise BackendError(
                f"drop_collection failed for {database!r}/{name!r}: {e}"
            ) from e
        self._invalidate_collection(database, name)

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
        # Milvus 2.4's create_index replaces an existing index on the
        # same field — there's no separate "rebuild" path. We still
        # best-effort drop the old one first so the operator's intent
        # ("rebuild") reads correctly in the request log and any
        # misconfigured index doesn't bleed into the new one.
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        if field_name not in {f["name"] for f in schema["fields"]}:
            raise StoreError(
                f"field {field_name!r} is not in the collection schema"
            )
        if field_name != schema.get("vector_field"):
            raise StoreError(
                f"create_index only supports the vector field "
                f"{schema.get('vector_field')!r}; got {field_name!r}"
            )
        if metric_type not in _METRIC:
            raise StoreError(
                f"unsupported metric_type {metric_type!r}; expected one of {sorted(_METRIC)}"
            )

        # Drop any pre-existing index on the same field. ``drop_index``
        # on a field with no index raises on some Milvus versions —
        # swallow those so a fresh create path stays clean.
        try:
            self._client.drop_index(collection, field_name=field_name)
        except Exception:
            pass

        try:
            ip = self._client.prepare_index_params()
            ip.add_index(
                field_name=field_name,
                index_type=index_type,
                metric_type=_METRIC[metric_type],
                params=dict(params or {}),
            )
            self._client.create_index(
                collection_name=collection, index_params=ip,
            )
        except (StoreError,):
            raise
        except Exception as e:
            self._invalidate_collection(database, collection)
            raise BackendError(
                f"create_index failed for {database!r}/{collection!r}"
                f"/{field_name!r}: {e}"
            ) from e
        # Index changes don't alter the schema, but a stale describe
        # cache would mislead a follow-up call. Invalidate to be safe.
        self._invalidate_collection(database, collection)

    def drop_index(
        self,
        database: str,
        collection: str,
        *,
        field_name: str,
    ) -> None:
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        if field_name not in {f["name"] for f in schema["fields"]}:
            raise StoreError(
                f"field {field_name!r} is not in the collection schema"
            )
        if field_name != schema.get("vector_field"):
            raise StoreError(
                f"drop_index only supports the vector field "
                f"{schema.get('vector_field')!r}; got {field_name!r}"
            )
        try:
            self._client.drop_index(collection, field_name=field_name)
        except Exception as e:
            # pymilvus raises when no index exists on some versions.
            # Surface that as an idempotent success — a no-op drop
            # is the most user-friendly behaviour for a dashboard
            # "delete index" button.
            msg = str(e).lower()
            if "not found" in msg or "not exist" in msg:
                return
            self._invalidate_collection(database, collection)
            raise BackendError(
                f"drop_index failed for {database!r}/{collection!r}"
                f"/{field_name!r}: {e}"
            ) from e
        self._invalidate_collection(database, collection)

    def describe_collection(
        self, database: str, name: str,
    ) -> dict[str, Any]:
        """Return a normalised schema dict:

        ``{
            "primary_field": "id",
            "vector_field": "vector",
            "dim": 1024,
            "metric": "cosine",
            "count": 0,
            "fields": [{"name", "dtype", "is_primary", "dim"}],
            "indexes": [{"field_name", "metric_type", "index_type", "params"}],
        }``
        """
        key = (database, name)
        now = time.monotonic()
        with self._lock:
            cached = self._schema_cache.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, name):
            raise CollectionNotFound(
                f"collection {name!r} does not exist in database {database!r}"
            )

        try:
            raw = self._client.describe_collection(name)
        except MilvusException as e:
            if e.code == _ERR_COLLECTION_NOT_FOUND:
                raise CollectionNotFound(
                    f"collection {name!r} does not exist in database {database!r}"
                ) from e
            raise BackendError(
                f"describe_collection failed for {database!r}/{name!r}: {e}"
            ) from e
        except Exception as e:
            raise BackendError(
                f"describe_collection failed for {database!r}/{name!r}: {e}"
            ) from e

        # ``describe_collection`` returns "type" as a DataType enum or int.
        primary_field = ""
        vector_field_name = ""
        vector_dim: int | None = None
        vector_field_names: list[str] = []
        fields_out: list[dict[str, Any]] = []
        for f in raw.get("fields", []):
            is_primary = bool(f.get("is_primary"))
            dtype_enum = f.get("type")
            dtype_name = self._dtype_name(dtype_enum)
            entry: dict[str, Any] = {
                "name": f["name"],
                "dtype": dtype_name,
                "is_primary": is_primary,
            }
            # ``nullable`` and ``default_value`` live at the field level in
            # pymilvus's describe_collection response (not inside ``params``).
            # Default them so they're always present in the payload — the
            # API response model will keep the defaults through Pydantic.
            entry["nullable"] = bool(f.get("nullable", False))
            if "default_value" in f and f["default_value"] is not None:
                entry["default_value"] = f["default_value"]
            params = f.get("params") or {}
            if dtype_name == "float_vector":
                vector_field_name = f["name"]
                vector_field_names.append(f["name"])
                vector_dim = int(params.get("dim") or 0)
                entry["dim"] = vector_dim
                fields_out.append(entry)
            else:
                if "max_length" in params:
                    entry["max_length"] = int(params["max_length"])
                fields_out.append(entry)
            if is_primary:
                primary_field = f["name"]

        # Enumerate every index on every vector field. ``list_indexes`` returns
        # the set of indexed field names; for each we call ``describe_index``
        # to read the index type / metric / params. Failures on a single field
        # are isolated (we just drop that field's index from the list) so one
        # misconfigured index cannot poison the whole collection summary.
        indexes_out: list[dict[str, Any]] = []
        metric = "cosine"
        indexed_vector_fields: list[str] = []
        try:
            indexed_fields = self._client.list_indexes(name)
            if isinstance(indexed_fields, list):
                indexed_vector_fields = [
                    f for f in indexed_fields if f in set(vector_field_names)
                ]
        except Exception:
            indexed_vector_fields = []
        for field_name in indexed_vector_fields:
            try:
                ixinfo = self._client.describe_index(name, field_name)
            except Exception:
                continue
            if not isinstance(ixinfo, dict):
                continue
            entry: dict[str, Any] = {
                "field_name": field_name,
                "metric_type": str(ixinfo.get("metric_type", "")).upper() or "?",
                "index_type": str(ixinfo.get("index_type", "")) or "?",
                "params": dict(ixinfo.get("params") or {}),
            }
            if field_name == vector_field_name:
                # Pin the collection-level ``metric`` to the vector field's
                # index metric so callers see a consistent value.
                metric = _INV_METRIC.get(
                    entry["metric_type"].upper(), entry["metric_type"].lower(),
                )
            # ``describe_index`` returns one entry per index segment in some
            # pymilvus versions; merge them under the same field_name so the
            # shape stays one-row-per-vector-field for the API consumer.
            existing = next(
                (e for e in indexes_out if e["field_name"] == field_name),
                None,
            )
            if existing is None:
                indexes_out.append(entry)
            else:
                # Keep the first segment's type/metric as the canonical row,
                # but union the params so operators can see all knobs.
                merged_params = dict(existing.get("params") or {})
                merged_params.update(entry["params"])
                existing["params"] = merged_params

        # Row count
        count = 0
        try:
            stats = self._client.get_collection_stats(name)
            if isinstance(stats, dict):
                count = int(stats.get("row_count", 0))
        except Exception:
            pass

        result = {
            "primary_field": primary_field,
            "vector_field": vector_field_name,
            "dim": int(vector_dim or 0),
            "metric": metric,
            "count": count,
            "fields": fields_out,
            "indexes": indexes_out,
        }
        with self._lock:
            self._schema_cache[key] = (
                time.monotonic() + _SCHEMA_CACHE_TTL_S, result,
            )
        return result

    def upsert(
        self,
        database: str,
        collection: str,
        primary_field: str,
        vector_field: str,
        ids: list[str],
        vectors: list[list[float]],
        fields: list[dict[str, Any]] | None,
    ) -> None:
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        schema_names = {f["name"] for f in schema["fields"]}
        if primary_field not in schema_names:
            raise StoreError(
                f"primary_field {primary_field!r} is not in the collection schema"
            )
        if vector_field not in schema_names:
            raise StoreError(
                f"vector_field {vector_field!r} is not in the collection schema"
            )
        expected_dim = schema["dim"]
        for i, v in enumerate(vectors):
            if len(v) != expected_dim:
                raise DimensionMismatch(
                    f"vector[{i}] dim {len(v)} != collection dim {expected_dim}",
                    expected=expected_dim, got=len(v),
                )

        rows: list[dict[str, Any]] = []
        # Cap every VARCHAR value at its declared ``max_length`` (in *bytes*,
        # not characters — Milvus counts bytes for VARCHAR). Truncating here
        # keeps callers from hitting Milvus error 1100 mid-batch and rolling
        # back an otherwise good upsert. Unknown / non-varchar fields are
        # passed through unchanged.
        varchar_caps = _varchar_byte_caps(schema["fields"])
        for i, (vid, vec) in enumerate(zip(ids, vectors)):
            row: dict[str, Any] = {primary_field: vid, vector_field: [float(x) for x in vec]}
            if fields is not None and i < len(fields):
                for k, v in (fields[i] or {}).items():
                    if k == primary_field:
                        continue
                    if k not in schema_names:
                        raise StoreError(
                            f"unknown scalar field {k!r}; declared: {sorted(schema_names)}"
                        )
                    if isinstance(v, str) and k in varchar_caps:
                        cap = varchar_caps[k]
                        if len(v.encode("utf-8")) > cap:
                            original_len = len(v.encode("utf-8"))
                            row[k] = _truncate_utf8_bytes(v, cap)
                            log.warning(
                                "varchar field %r on row %s exceeded %d bytes "
                                "(was %d); truncated to fit schema",
                                k, vid, cap, original_len,
                            )
                        else:
                            row[k] = v
                    else:
                        row[k] = v
            rows.append(row)

        try:
            self._client.upsert(collection, data=rows)
        except (StoreError, DimensionMismatch):
            raise
        except Exception as e:
            raise BackendError(
                f"upsert failed for {database!r}/{collection!r}: {e}"
            ) from e

    def delete(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str] | None = None,
        *,
        filter_expr: str | None = None,
    ) -> int:
        has_ids = ids is not None
        has_filter = filter_expr is not None and filter_expr.strip() != ""
        if has_ids == has_filter:
            raise StoreError("provide exactly one of ids or filter_expr")
        if has_ids and len(ids) == 0:  # type: ignore[arg-type]
            raise StoreError("ids must be non-empty when provided")

        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        if primary_field not in {f["name"] for f in schema["fields"]}:
            raise StoreError(
                f"primary_field {primary_field!r} is not in the collection schema"
            )

        # pymilvus 2.4's ``delete`` accepts either ``ids`` or ``filter``
        # (mutually exclusive on its side too). Build the kwargs
        # accordingly and let the SDK reject ambiguous calls.
        try:
            self._ensure_loaded(collection)
            if has_ids:
                res = self._client.delete(collection, ids=list(ids))
            else:
                res = self._client.delete(collection, filter=filter_expr)
        except (CollectionNotFound, StoreError):
            raise
        except Exception as e:
            raise BackendError(
                f"delete failed for {database!r}/{collection!r}: {e}"
            ) from e
        # The cached describe payload carries a ``count`` field — stale
        # the moment a delete lands. Drop it so the next read doesn't
        # serve the pre-delete row count for up to the cache TTL.
        self._invalidate_collection(database, collection)
        if isinstance(res, dict):
            return int(res.get("delete_count", 0))
        return 0

    def count(
        self,
        database: str,
        collection: str,
        *,
        filter_expr: str | None = None,
    ) -> int:
        """Live, tombstone-aware row count via ``count(*)``.

        ``get_collection_stats`` (the row_count inside describe_collection)
        keeps counting deleted entities until the next compaction, and the
        describe payload is cached — both make a post-delete browse view
        look unrefreshed. A real query under Strong consistency returns
        the authoritative count immediately, optionally restricted to
        rows matching ``filter_expr`` (same expression the browse call
        uses, so the pager's denominator matches the rows on screen).
        """
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        try:
            self._ensure_loaded(collection)
            rows = self._client.query(
                collection,
                filter=filter_expr or "",
                output_fields=["count(*)"],
                # Strong: the count must reflect the delete that just
                # returned, even when the request lands on a different
                # worker/process than the one that issued it.
                consistency_level="Strong",
            )
        except (CollectionNotFound, StoreError):
            raise
        except Exception as e:
            raise BackendError(
                f"count failed for {database!r}/{collection!r}: {e}"
            ) from e

        if not rows:
            return 0
        first = rows[0] if isinstance(rows[0], dict) else {}
        # Milvus returns a one-row payload ``[{"count(*)": N}]``; accept
        # the literal key but tolerate any count-shaped spelling too.
        val = first.get("count(*)")
        if val is None:
            val = next((v for k, v in first.items() if "count" in str(k)), 0)
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def get(
        self,
        database: str,
        collection: str,
        primary_field: str,
        ids: list[str],
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        if primary_field not in {f["name"] for f in schema["fields"]}:
            raise StoreError(
                f"primary_field {primary_field!r} is not in the collection schema"
            )
        if output_fields:
            unknown = [
                f for f in output_fields
                if f not in {ff["name"] for ff in schema["fields"]}
            ]
            if unknown:
                raise StoreError(
                    f"unknown output_fields {unknown}; declared: "
                    f"{[ff['name'] for ff in schema['fields']]}"
                )

        output = list({primary_field, *(output_fields or [])})
        # pymilvus's ``get`` returns string reprs on 2.4 — fall back to
        # ``query`` with an explicit filter for structured dicts.
        expr = " or ".join(
            f'{primary_field} == "{_escape(vid)}"' for vid in ids
        )
        try:
            self._ensure_loaded(collection)
            rows = self._client.query(
                collection,
                filter=expr,
                output_fields=output,
                ids=None,
            )
        except (CollectionNotFound, StoreError):
            raise
        except Exception as e:
            raise BackendError(
                f"get failed for {database!r}/{collection!r}: {e}"
            ) from e

        # Rows come back as dicts; group by primary value.
        primary_values = {r.get(primary_field) for r in rows}
        out: list[dict[str, Any]] = []
        for vid in ids:
            if vid not in primary_values:
                continue
            row = next(r for r in rows if r.get(primary_field) == vid)
            item: dict[str, Any] = {"id": vid, "fields": {}}
            if output_fields:
                item["fields"] = {k: row.get(k) for k in output_fields}
            out.append(item)
        return out

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
    ) -> list[dict[str, Any]]:
        """Paginated, filterable list view over a collection.

        Maps directly onto ``MilvusClient.query``: ``limit`` caps the
        returned row count, ``offset`` skips that many matching rows
        (pymilvus 2.4 supports both), and ``output_fields`` selects the
        scalar columns to materialise. The vector column is never
        materialised — even if the caller puts it in ``output_fields``,
        we drop it before issuing the RPC, so a wide-vector collection
        cannot be accidentally pulled through this path.
        """
        if limit < 0:
            raise StoreError(f"limit must be >= 0, got {limit}")
        if offset < 0:
            raise StoreError(f"offset must be >= 0, got {offset}")
        if limit == 0:
            return []

        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        schema_names = {f["name"] for f in schema["fields"]}
        if primary_field not in schema_names:
            raise StoreError(
                f"primary_field {primary_field!r} is not in the collection schema"
            )

        # Resolve "all scalar fields" when the caller didn't pick any.
        # The vector field is always excluded — a 1024-dim float vector
        # per row would balloon responses on a wide embedder and is not
        # what a dashboard browse view is for.
        if output_fields is None:
            output = [f["name"] for f in schema["fields"] if f["name"] != schema.get("vector_field")]
        else:
            unknown = [
                f for f in output_fields
                if f not in schema_names
            ]
            if unknown:
                raise StoreError(
                    f"unknown output_fields {unknown}; declared: "
                    f"{sorted(schema_names)}"
                )
            output = [f for f in output_fields if f != schema.get("vector_field")]
        # Always include the primary key in the projection so we can
        # wrap each row into the standard ``{"id", "fields"}`` shape.
        if primary_field not in output:
            output = [primary_field, *output]

        try:
            self._ensure_loaded(collection)
            kwargs: dict[str, Any] = {
                "filter": filter_expr or "",
                "output_fields": output,
                "limit": int(limit),
                "offset": int(offset),
                # Read-your-writes across processes: a delete issued by
                # another worker must be reflected on the next page load.
                "consistency_level": "Strong",
            }
            rows = self._client.query(collection, **kwargs)
        except (CollectionNotFound, StoreError):
            raise
        except Exception as e:
            raise BackendError(
                f"browse failed for {database!r}/{collection!r}: {e}"
            ) from e

        out: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            pid = row.get(primary_field)
            fields = {k: v for k, v in row.items() if k != primary_field}
            out.append({"id": pid, "fields": fields})
        return out

    def search(
        self,
        database: str,
        collection: str,
        vector_field: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        if vector_field not in {f["name"] for f in schema["fields"]}:
            raise StoreError(
                f"vector_field {vector_field!r} is not in the collection schema"
            )
        expected_dim = schema["dim"]
        if len(query_vector) != expected_dim:
            raise DimensionMismatch(
                f"query dim {len(query_vector)} != collection dim {expected_dim}",
                expected=expected_dim, got=len(query_vector),
            )
        if output_fields:
            unknown = [
                f for f in output_fields
                if f not in {ff["name"] for ff in schema["fields"]}
            ]
            if unknown:
                raise StoreError(
                    f"unknown output_fields {unknown}; declared: "
                    f"{[ff['name'] for ff in schema['fields']]}"
                )

        try:
            self._ensure_loaded(collection)
            results = self._client.search(
                collection,
                data=[[float(x) for x in query_vector]],
                anns_field=vector_field,
                limit=top_k,
                filter=filter_expr or "",
                output_fields=list({vector_field, *(output_fields or [])}),
            )
        except (DimensionMismatch, StoreError, CollectionNotFound):
            raise
        except Exception as e:
            raise BackendError(
                f"search failed for {database!r}/{collection!r}: {e}"
            ) from e

        hits: list[dict[str, Any]] = []
        for batch in results or []:
            for hit in batch:
                entity = hit.get("entity") if isinstance(hit, dict) else None
                fields: dict[str, Any] = {}
                if isinstance(entity, dict):
                    raw = dict(entity)
                    raw.pop(vector_field, None)
                    if output_fields:
                        fields = {k: raw.get(k) for k in output_fields}
                    else:
                        fields = raw
                hits.append({
                    "id": str(hit.get("id")),
                    "score": float(hit.get("distance", 0.0)),
                    "fields": fields,
                })
        return hits

    def search_text(
        self,
        database: str,
        collection: str,
        sparse_field: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 full-text leg; mirrors :meth:`search` result shape."""
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        declared = {f["name"] for f in schema["fields"]}
        if sparse_field not in declared:
            raise StoreError(
                f"sparse_field {sparse_field!r} is not in the collection schema"
            )
        if output_fields:
            unknown = [f for f in output_fields if f not in declared]
            if unknown:
                raise StoreError(
                    f"unknown output_fields {unknown}; declared: {sorted(declared)}"
                )

        try:
            self._ensure_loaded(collection)
            results = self._client.search(
                collection,
                data=[query_text],
                anns_field=sparse_field,
                limit=top_k,
                filter=filter_expr or "",
                output_fields=list({sparse_field, *(output_fields or [])}),
                search_params={"metric_type": "BM25"},
            )
        except (StoreError, CollectionNotFound):
            raise
        except Exception as e:
            raise BackendError(
                f"search_text failed for {database!r}/{collection!r}: {e}"
            ) from e

        hits: list[dict[str, Any]] = []
        for batch in results or []:
            for hit in batch:
                entity = hit.get("entity") if isinstance(hit, dict) else None
                fields: dict[str, Any] = {}
                if isinstance(entity, dict):
                    raw = dict(entity)
                    raw.pop(sparse_field, None)
                    if output_fields:
                        fields = {k: raw.get(k) for k in output_fields}
                    else:
                        fields = raw
                hits.append({
                    "id": str(hit.get("id")),
                    "score": float(hit.get("distance", 0.0)),
                    "fields": fields,
                })
        return hits

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self, collection: str) -> None:
        try:
            state = self._client.get_load_state(collection)
            if isinstance(state, dict):
                loaded = state.get("state")
                if loaded is not None and "LoadState.NotLoad" in str(loaded):
                    self._client.load_collection(collection)
            else:
                # If we can't tell, just load — no-op if already loaded.
                self._client.load_collection(collection)
        except Exception:
            # Don't fail the operation just because we couldn't load;
            # search/get on Milvus also surfaces "collection not loaded"
            # with a clearer error than ours.
            pass

    @staticmethod
    def _dtype_name(value: Any) -> str:
        """``DataType.VARCHAR`` → ``"varchar"``; ``21`` → ``"varchar"``."""
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, int):
            mapping = {
                1: "bool",
                2: "int8", 3: "int16", 4: "int32", 5: "int64",
                10: "float", 11: "double",
                21: "varchar",
                23: "json",
                100: "binary_vector", 101: "float_vector",
                104: "sparse_float_vector",
            }
            return mapping.get(value, str(value))
        name = getattr(value, "name", None)
        if name:
            return name.lower()
        return str(value).lower()

    @property
    def bound_db(self) -> str:
        return self._bound_db


def _escape(s: str) -> str:
    """Escape a value for safe embedding in a Milvus filter expression."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _varchar_byte_caps(fields: list[dict[str, Any]]) -> dict[str, int]:
    """Map ``field_name → max_length_bytes`` for every declared VARCHAR.

    Built from the same dict shape ``describe_collection`` produces: only
    varchar fields with a positive ``max_length`` are included. Used by
    :meth:`MilvusAdapter.upsert` to truncate user values that would
    otherwise blow past Milvus's byte cap and trigger error 1100.
    """
    caps: dict[str, int] = {}
    for f in fields:
        if f.get("dtype") != "varchar":
            continue
        ml = f.get("max_length")
        if isinstance(ml, int) and ml > 0:
            caps[f["name"]] = ml
    return caps


def _truncate_utf8_bytes(s: str, max_bytes: int) -> str:
    """Truncate ``s`` to at most ``max_bytes`` UTF-8 bytes without splitting
    a multi-byte sequence.

    Surrogate-style truncation (``s.encode()[:max_bytes]``) can leave an
    incomplete multi-byte char at the boundary, which pymilvus then rejects.
    Decoding with ``errors="ignore"`` drops the dangling tail bytes so the
    returned string is always valid UTF-8 and strictly under the cap.
    """
    if max_bytes <= 0:
        return ""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
