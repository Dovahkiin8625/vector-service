"""Recall channels.

A channel turns one :class:`RecallSpec` into a ranked
:class:`ChannelRun`. The pipeline builds one channel instance per
request and fans a channel out once per query variant, so channels
themselves stay stateless beyond their (database, collection) scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from vector_service.retrieval.base import ChannelHit, ChannelRun, RecallSpec


class Channel(ABC):
    """One retrieval leg type."""

    name: str

    @abstractmethod
    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        """Run one recall leg and return the ranked channel run."""

    def _wrap(self, query: str, hits) -> ChannelRun:
        return ChannelRun(
            channel=self.name,
            query=query,
            hits=[
                ChannelHit(
                    chunk_id=str(h.id),
                    score=float(h.score),
                    rank=i + 1,
                    fields=dict(h.fields),
                )
                for i, h in enumerate(hits)
            ],
        )


class DenseChannel(Channel):
    """ANN leg over the dense float vector field."""

    name = "dense"

    def __init__(self, store, database: str, collection: str,
                 embedder, vector_field: str = "vector"):
        self._store = store
        self._database = database
        self._collection = collection
        self._embedder = embedder
        self._vector_field = vector_field

    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        vector = spec.vector
        if vector is None:
            vector = self._embedder.embed_query(spec.query)
        hits = self._store.search(
            self._database,
            self._collection,
            self._vector_field,
            vector,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return self._wrap(spec.query, hits)


class BM25Channel(Channel):
    """Full-text leg over the BM25 sparse field."""

    name = "bm25"

    def __init__(self, store, database: str, collection: str,
                 sparse_field: str = "sparse"):
        self._store = store
        self._database = database
        self._collection = collection
        self._sparse_field = sparse_field

    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        hits = self._store.search_text(
            self._database,
            self._collection,
            self._sparse_field,
            spec.query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return self._wrap(spec.query, hits)
