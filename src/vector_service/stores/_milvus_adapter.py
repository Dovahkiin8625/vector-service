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
"""
from __future__ import annotations

import threading
from typing import Any

from pymilvus import (
    DataType,
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
}

_METRIC = {"cosine": "COSINE", "ip": "IP", "l2": "L2"}
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
        self._ensure_connected()
        self._using_db(database)
        try:
            return bool(self._client.has_collection(name))
        except Exception:
            return False

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
        if self.has_collection(database, name):
            raise CollectionAlreadyExists(
                f"collection {name!r} already exists in database {database!r}"
            )
        if vector_metric not in _METRIC:
            raise StoreError(
                f"unsupported metric {vector_metric!r}; expected one of {sorted(_METRIC)}"
            )
        for ip in indexes:
            if ip.get("field_name") != vector_field_name:
                raise StoreError(
                    f"index target {ip.get('field_name')!r} is not the vector field "
                    f"{vector_field_name!r}"
                )
            if ip.get("metric_type") not in _METRIC:
                raise StoreError(
                    f"unsupported index metric {ip.get('metric_type')!r}"
                )

        try:
            schema = MilvusClient.create_schema(
                auto_id=False, enable_dynamic_field=False,
            )
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
                if f.get("nullable"):
                    kwargs["nullable"] = True
                if f.get("default_value") is not None:
                    kwargs["default_value"] = f["default_value"]
                schema.add_field(f["name"], dtype, **kwargs)
            schema.add_field(vector_field_name, DataType.FLOAT_VECTOR, dim=vector_dim)
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
        except Exception as e:
            raise BackendError(
                f"create_collection failed for {database!r}/{name!r}: {e}"
            ) from e

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
            raise BackendError(
                f"drop_collection failed for {database!r}/{name!r}: {e}"
            ) from e

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
            params = f.get("params") or {}
            if dtype_name == "float_vector":
                vector_field_name = f["name"]
                vector_dim = int(params.get("dim") or 0)
                entry["dim"] = vector_dim
                fields_out.append(entry)
            else:
                if "max_length" in params:
                    entry["max_length"] = int(params["max_length"])
                fields_out.append(entry)
            if is_primary:
                primary_field = f["name"]

        # Index metric on the vector field
        metric = "cosine"
        try:
            ixinfo = self._client.describe_index(name, vector_field_name)
            mt = ixinfo.get("metric_type") if isinstance(ixinfo, dict) else None
            if isinstance(mt, str):
                metric = _INV_METRIC.get(mt.upper(), mt.lower())
        except Exception:
            pass

        # Row count
        count = 0
        try:
            stats = self._client.get_collection_stats(name)
            if isinstance(stats, dict):
                count = int(stats.get("row_count", 0))
        except Exception:
            pass

        return {
            "primary_field": primary_field,
            "vector_field": vector_field_name,
            "dim": int(vector_dim or 0),
            "metric": metric,
            "count": count,
            "fields": fields_out,
        }

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
        ids: list[str],
    ) -> int:
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
        try:
            self._ensure_loaded(collection)
            res = self._client.delete(collection, ids=list(ids))
            # Force visibility for callers that immediately read back
            # (Milvus writes are eventually consistent).
            try:
                self._client.refresh_load(collection)
            except Exception:
                pass
        except (CollectionNotFound, StoreError):
            raise
        except Exception as e:
            raise BackendError(
                f"delete failed for {database!r}/{collection!r}: {e}"
            ) from e
        if isinstance(res, dict):
            return int(res.get("delete_count", 0))
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
