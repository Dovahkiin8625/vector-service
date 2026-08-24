"""In-memory FakeStore implementing VectorStore ABC for tests."""
from __future__ import annotations

import math
from typing import Any

from vector_service.core.errors import (
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.stores.base import Hit, VectorStore


class FakeStore(VectorStore):
    """Pure-Python in-memory vector store. Brute-force cosine search."""

    def __init__(self):
        self.backend_name = "fake"
        self._collections: dict[str, _FakeCollection] = {}

    @property
    def backend(self) -> Any:
        return self  # escape hatch: 暴露自己

    def create_collection(self, name, dim, *, metric="cosine", **_opts):
        if name in self._collections:
            raise CollectionAlreadyExists(f"collection {name!r} exists")
        self._collections[name] = _FakeCollection(name, dim, metric)

    def drop_collection(self, name):
        if name not in self._collections:
            raise CollectionNotFound(name)
        del self._collections[name]

    def list_collections(self):
        return list(self._collections.keys())

    def collection_info(self, name):
        c = self._require(name)
        return {"name": c.name, "dim": c.dim, "metric": c.metric, "count": len(c.records)}

    def upsert(self, collection, ids, vectors, metadatas=None):
        c = self._require(collection)
        if len(ids) != len(vectors):
            raise StoreError("ids and vectors must have same length")
        metas = metadatas or [{} for _ in ids]
        for vid, vec, meta in zip(ids, vectors, metas):
            if len(vec) != c.dim:
                raise DimensionMismatch(
                    f"vector dim {len(vec)} != collection dim {c.dim}",
                    expected=c.dim, got=len(vec),
                )
            c.records[vid] = (list(vec), dict(meta))

    def delete(self, collection, ids):
        c = self._require(collection)
        for vid in ids:
            c.records.pop(vid, None)

    def get(self, collection, ids):
        c = self._require(collection)
        return [
            {"id": vid, "vector": c.records[vid][0], "metadata": c.records[vid][1]}
            for vid in ids if vid in c.records
        ]

    def search(self, collection, query_vector, top_k=10, filter=None):
        c = self._require(collection)
        if len(query_vector) != c.dim:
            raise DimensionMismatch(
                f"query dim {len(query_vector)} != collection dim {c.dim}",
                expected=c.dim, got=len(query_vector),
            )

        scored: list[Hit] = []
        for vid, (vec, meta) in c.records.items():
            if filter is not None and not _matches_filter(meta, filter):
                continue
            score = _cosine(query_vector, vec)
            scored.append(Hit(id=vid, score=score, metadata=dict(meta)))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def _require(self, name) -> "_FakeCollection":
        if name not in self._collections:
            raise CollectionNotFound(name)
        return self._collections[name]


class _FakeCollection:
    def __init__(self, name, dim, metric):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.records: dict[str, tuple[list[float], dict]] = {}


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _matches_filter(meta: dict, flt: dict) -> bool:
    return all(meta.get(k) == v for k, v in flt.items())