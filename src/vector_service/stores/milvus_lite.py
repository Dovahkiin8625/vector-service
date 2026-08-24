"""Milvus Lite implementation of VectorStore.

pymilvus is imported lazily inside methods so the module can be loaded
without pymilvus installed (e.g. for environments that only need other
backends).
"""
from __future__ import annotations

import json
import os
from typing import Any

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.core.filter_translator import FilterTranslationError, translate_filter
from vector_service.stores.base import Hit, VectorStore

_VALID_METRICS = {"cosine": "COSINE", "ip": "IP", "l2": "L2"}


class MilvusLiteStore(VectorStore):
    backend_name = "milvus_lite"

    def __init__(self, uri: str | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self._uri = uri or s.milvus_uri
        # 确保父目录存在
        parent = os.path.dirname(self._uri)
        if parent:
            os.makedirs(parent, exist_ok=True)

        from pymilvus import connections  # lazy: 需要 pymilvus
        try:
            connections.connect("default", uri=self._uri)
        except Exception as e:
            raise BackendError(f"failed to connect to milvus lite at {self._uri}: {e}") from e

    @property
    def backend(self) -> Any:
        """原生连接：返回当前 default connection 的引用对象（pymilvus 不暴露对象，但可用 utility）。"""
        return _MilvusLiteBackendProxy(self._uri)

    def close(self) -> None:
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except Exception:
            pass

    def create_collection(self, name, dim, *, metric="cosine", **backend_opts):
        if name in self.list_collections():
            raise CollectionAlreadyExists(f"collection {name!r} exists")
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
            coll = Collection(name=name, schema=schema, using="default")

            metric_type = _VALID_METRICS.get(metric)
            if metric_type is None:
                raise StoreError(f"unsupported metric {metric!r}")
            index_params = {
                "metric_type": metric_type,
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            coll.create_index(field_name="vector", index_params=index_params)
        except (CollectionAlreadyExists, StoreError):
            raise
        except Exception as e:
            raise BackendError(f"create_collection failed: {e}") from e

    def drop_collection(self, name):
        if name not in self.list_collections():
            raise CollectionNotFound(name)
        try:
            from pymilvus import utility
            utility.drop_collection(name, using="default")
        except Exception as e:
            raise BackendError(f"drop_collection failed: {e}") from e

    def list_collections(self) -> list[str]:
        try:
            from pymilvus import utility
            return list(utility.list_collections(using="default"))
        except Exception as e:
            raise BackendError(f"list_collections failed: {e}") from e

    def collection_info(self, name):
        self._require(name)
        try:
            from pymilvus import Collection
            coll = Collection(name, using="default")
            coll.flush()
            count = coll.num_entities
            # 从 schema 读 dim
            dim = next(
                f.params["dim"]
                for f in coll.schema.fields
                if f.name == "vector"
            )
            metric = self._metric_for(coll)
            return {"name": name, "dim": dim, "metric": metric, "count": count}
        except CollectionNotFound:
            raise
        except Exception as e:
            raise BackendError(f"collection_info failed: {e}") from e

    def upsert(self, collection, ids, vectors, metadatas=None):
        coll = self._require(collection)
        if len(ids) != len(vectors):
            raise StoreError("ids and vectors length mismatch")
        # dim 校验
        expected_dim = self.collection_info(collection)["dim"]
        for i, v in enumerate(vectors):
            if len(v) != expected_dim:
                raise DimensionMismatch(
                    f"vector[{i}] dim {len(v)} != collection dim {expected_dim}",
                    expected=expected_dim, got=len(v),
                )

        metas = metadatas or [{} for _ in ids]
        rows = [
            {"id": vid, "vector": list(map(float, vec)), "metadata_json": json.dumps(meta, ensure_ascii=False)}
            for vid, vec, meta in zip(ids, vectors, metas)
        ]
        try:
            coll.upsert(rows)
            coll.flush()
        except Exception as e:
            raise BackendError(f"upsert failed: {e}") from e

    def delete(self, collection, ids):
        coll = self._require(collection)
        try:
            expr = " or ".join(f'id == "{_escape(vid)}"' for vid in ids)
            coll.delete(expr)
            coll.flush()
        except Exception as e:
            raise BackendError(f"delete failed: {e}") from e

    def get(self, collection, ids):
        coll = self._require(collection)
        if not ids:
            return []
        try:
            expr = " or ".join(f'id == "{_escape(vid)}"' for vid in ids)
            rows = coll.query(expr=expr, output_fields=["id", "metadata_json"])
            by_id = {r["id"]: r for r in rows}
            return [
                {
                    "id": vid,
                    "vector": by_id[vid].get("vector"),  # query 不返回 vector，需 search
                    "metadata": json.loads(by_id[vid]["metadata_json"]) if vid in by_id else {},
                }
                for vid in ids if vid in by_id
            ]
        except Exception as e:
            raise BackendError(f"get failed: {e}") from e

    def search(self, collection, query_vector, top_k=10, filter=None):
        coll = self._require(collection)
        # dim 校验
        expected_dim = self.collection_info(collection)["dim"]
        if len(query_vector) != expected_dim:
            raise DimensionMismatch(
                f"query dim {len(query_vector)} != collection dim {expected_dim}",
                expected=expected_dim, got=len(query_vector),
            )
        try:
            expr = translate_filter(filter)
        except FilterTranslationError as e:
            raise StoreError(f"filter not supported: {e}") from e

        try:
            params = {"ef": 64}
            results = coll.search(
                data=[list(map(float, query_vector))],
                anns_field="vector",
                param=params,
                limit=top_k,
                expr=expr or None,
                output_fields=["metadata_json"],
            )
            hits: list[Hit] = []
            if not results:
                return hits
            for hit in results[0]:
                meta_raw = hit.entity.get("metadata_json") if hasattr(hit, "entity") else "{}"
                meta = json.loads(meta_raw) if meta_raw else {}
                hits.append(Hit(id=hit.id, score=float(hit.score), metadata=meta))
            return hits
        except (DimensionMismatch, StoreError):
            raise
        except Exception as e:
            raise BackendError(f"search failed: {e}") from e

    def _require(self, name):
        from pymilvus import Collection
        if name not in self.list_collections():
            raise CollectionNotFound(name)
        return Collection(name, using="default")

    @staticmethod
    def _metric_for(coll) -> str:
        try:
            idx = coll.indexes
            if idx:
                mt = idx[0].params.get("metric_type", "COSINE")
                inv = {v: k for k, v in _VALID_METRICS.items()}
                return inv.get(mt, mt.lower())
        except Exception:
            pass
        return "cosine"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


class _MilvusLiteBackendProxy:
    """Escape hatch: exposes a few utility methods, hides raw connection."""

    def __init__(self, uri: str):
        self.uri = uri

    def list_collections(self):
        from pymilvus import utility
        return list(utility.list_collections(using="default"))

    def describe_collection(self, name: str):
        from pymilvus import Collection
        return Collection(name, using="default").schema