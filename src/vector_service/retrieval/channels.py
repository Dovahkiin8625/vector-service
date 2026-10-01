"""Recall channels.

A channel turns one :class:`RecallSpec` into a ranked
:class:`ChannelRun`. The pipeline builds one channel instance per
request and fans a channel out once per query variant, so channels
themselves stay stateless beyond their (database, collection) scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from vector_service.retrieval.base import (
    ChannelHit,
    ChannelRun,
    MetaPredicate,
    RecallSpec,
)


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


class SummaryChannel(DenseChannel):
    """ANN leg over the chunk-summary vector field."""

    name = "summary"

    def __init__(self, store, database: str, collection: str,
                 embedder, summary_field: str = "summary_vector"):
        super().__init__(store, database, collection, embedder, summary_field)


def _match_numeric(value: int, op: str, target: str) -> bool:
    other = int(target)
    if op == "==":
        return value == other
    if op == "!=":
        return value != other
    if op == ">":
        return value > other
    if op == "<":
        return value < other
    if op == ">=":
        return value >= other
    return value <= other


# Bounds on the provenance lists embedded in one graph-hit field.
_GRAPH_ENTITY_FIELDS = 8
_GRAPH_COMMUNITY_FIELDS = 4


class GraphChannel(Channel):
    """GraphRAG leg: entity/community ANN → provenance leaf chunks.

    Entity and community vectors are searched in their independent
    collections; matched graph rows are mapped back to the leaf chunks
    that mention them, so the fused answer stays in the chunk vocabulary
    while carrying graph provenance (``graph_entities`` /
    ``graph_communities`` fields).
    """

    name = "graph"

    def __init__(
        self,
        *,
        store,
        repo,
        database: str,
        entity_collection: str,
        community_collection: str,
        embedder,
    ):
        self._store = store
        self._repo = repo
        self._database = database
        self._entity_coll = entity_collection
        self._community_coll = community_collection
        self._embedder = embedder

    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        *,
        doc_ids: set[str] | None = None,
        index_predicates: list[MetaPredicate] | None = None,
    ) -> ChannelRun:
        vector = self._embedder.embed_query(spec.query)
        entity_hits = self._store.search(
            self._database, self._entity_coll, "vector", vector,
            top_k=top_k, output_fields=["id"],
        )
        community_hits = self._store.search(
            self._database, self._community_coll, "vector", vector,
            top_k=top_k, output_fields=["id"],
        )
        entity_score = {str(h.id): float(h.score) for h in entity_hits}
        community_score = {str(h.id): float(h.score) for h in community_hits}

        entities = self._repo.hydrate_entities(list(entity_score))
        communities = self._repo.hydrate_communities(list(community_score))

        # Communities resolve to their member entities; all graph rows
        # are then mapped to provenance chunks in one mentions read.
        community_entities: dict[str, list[str]] = {}
        extra_entity_ids: set[str] = set()
        for cid, info in communities.items():
            members = list(info["entities"])
            community_entities[cid] = members
            extra_entity_ids.update(members)
        mention_rows = self._repo.entity_mention_rows(
            list(set(entity_score) | extra_entity_ids)
        )

        index_predicates = index_predicates or []

        def _allowed(row: dict) -> bool:
            if doc_ids is not None and row["doc_id"] not in doc_ids:
                return False
            return all(
                _match_numeric(row["chunk_index"], p.op, p.value)
                for p in index_predicates
            )

        # Accumulate score + provenance per leaf chunk.
        chunk_score: dict[str, float] = {}
        entity_for_chunk: dict[str, dict[str, dict]] = {}
        for row in mention_rows:
            eid = row["entity_id"]
            score = entity_score.get(eid)
            if score is None or not _allowed(row):
                continue
            cid_chunk = row["chunk_id"]
            if score > chunk_score.get(cid_chunk, float("-inf")):
                chunk_score[cid_chunk] = score
            bucket = entity_for_chunk.setdefault(cid_chunk, {})
            if eid not in bucket:
                info = entities.get(eid, {})
                bucket[eid] = {
                    "name": info.get("name", eid),
                    "entity_type": info.get("entity_type", ""),
                    "description": info.get("description", ""),
                    "score": score,
                }

        # Community contributions: same chunks, own provenance list.
        chunk_communities: dict[str, list[dict]] = {}
        for cid, info in communities.items():
            score = community_score[cid]
            members = set(community_entities[cid])
            for row in mention_rows:
                if row["entity_id"] not in members or not _allowed(row):
                    continue
                cid_chunk = row["chunk_id"]
                if score > chunk_score.get(cid_chunk, float("-inf")):
                    chunk_score[cid_chunk] = score
                bucket = chunk_communities.setdefault(cid_chunk, [])
                payload = {
                    "community_index": info["community_index"],
                    "summary": info["summary"],
                    "score": score,
                }
                if payload not in bucket:
                    bucket.append(payload)

        ranked = sorted(chunk_score, key=lambda c: (-chunk_score[c], c))
        hits: list[ChannelHit] = []
        for rank, chunk_id in enumerate(ranked, start=1):
            ent_list = sorted(
                entity_for_chunk.get(chunk_id, {}).values(),
                key=lambda e: -float(e["score"]),
            )[:_GRAPH_ENTITY_FIELDS]
            com_list = sorted(
                chunk_communities.get(chunk_id, []),
                key=lambda c: -float(c["score"]),
            )[:_GRAPH_COMMUNITY_FIELDS]
            hits.append(ChannelHit(
                chunk_id=chunk_id,
                score=chunk_score[chunk_id],
                rank=rank,
                fields={
                    "graph_entities": ent_list,
                    "graph_communities": com_list,
                },
            ))
        return ChannelRun(channel=self.name, query=spec.query, hits=hits)
