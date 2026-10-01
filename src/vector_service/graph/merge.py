"""Merge per-chunk extractions into one resolved graph.

Entity identity is the canonical name (whitespace-collapsed) within one
logical collection — no LLM entity-resolution round: repeated mentions
accumulate descriptions, provenance and edge weight, which matches the
GraphRAG text-unit merge while staying deterministic and offline.

All ids are deterministic hashes of (scope, identity), so a rebuild
produces the same ids and derived rows can be compared/overwritten.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from vector_service.graph.extraction import ChunkExtraction

# Caps on merged text so one noisy entity cannot bloat the corpus.
MAX_DESCRIPTION_CHARS = 2000
MAX_DESCRIPTIONS = 20


@dataclass
class BuiltEntity:
    """Persisted entity row."""

    entity_id: str
    name: str
    entity_type: str
    description: str
    created_ts: float
    updated_ts: float


@dataclass
class BuiltEdge:
    """Persisted edge row."""

    edge_id: str
    source_id: str
    target_id: str
    description: str
    weight: int
    created_ts: float
    updated_ts: float


@dataclass
class BuiltClaim:
    """Persisted claim row."""

    claim_id: str
    subject_id: str
    object_id: str | None
    chunk_id: str
    claim_type: str
    status: str
    statement: str
    created_ts: float


@dataclass
class BuiltGraph:
    """Everything replace_graph persists except communities."""

    entities: list[BuiltEntity] = field(default_factory=list)
    edges: list[BuiltEdge] = field(default_factory=list)
    claims: list[BuiltClaim] = field(default_factory=list)
    entity_mentions: list[tuple[str, str]] = field(default_factory=list)
    edge_mentions: list[tuple[str, str]] = field(default_factory=list)


class _EntityAcc:
    __slots__ = ("descriptions", "entity_type", "mentions", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.entity_type = ""
        self.descriptions: list[str] = []
        self.mentions: set[str] = set()


class _EdgeAcc:
    __slots__ = ("descriptions", "mentions", "source", "target", "weight")

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        self.descriptions: list[str] = []
        self.weight = 0
        self.mentions: set[str] = set()


def canonical_name(name: str) -> str:
    """Collapse whitespace; extraction already trimmed/stripped."""
    return " ".join(str(name).split())


class GraphBuilder:
    """Accumulates extractions and builds the resolved graph."""

    def __init__(
        self, database: str, collection: str, *, now: float
    ) -> None:
        self._scope = f"{database}/{collection}/"
        self._now = now
        self._entities: dict[str, _EntityAcc] = {}
        self._edges: dict[tuple[str, str], _EdgeAcc] = {}
        self._claims: list[BuiltClaim] = []
        self._claim_ids: set[str] = set()

    def entity_id_for(self, name: str) -> str:
        """Deterministic entity id for a canonical name in this scope."""
        digest = hashlib.sha1(
            (self._scope + name).encode("utf-8")
        ).hexdigest()
        return f"ge_{digest[:16]}"

    def _edge_id(self, source_id: str, target_id: str) -> str:
        digest = hashlib.sha1(
            f"{source_id}>{target_id}".encode()
        ).hexdigest()
        return f"ed_{digest[:16]}"

    def add_chunk(self, chunk_id: str, extraction: ChunkExtraction) -> None:
        """Merge one chunk's extraction; entities resolve before edges."""
        for raw in extraction.entities:
            name = canonical_name(raw.name)
            acc = self._entities.get(name)
            if acc is None:
                acc = _EntityAcc(name)
                self._entities[name] = acc
            if raw.entity_type and not acc.entity_type:
                acc.entity_type = raw.entity_type
            if raw.description and raw.description not in acc.descriptions:
                acc.descriptions.append(raw.description)
            acc.mentions.add(chunk_id)

        for raw in extraction.edges:
            source = canonical_name(raw.source)
            target = canonical_name(raw.target)
            if source == target or source not in self._entities:
                # Dangling endpoint (target unseen in this/any chunk) —
                # a graph row must reference a real entity.
                continue
            if target not in self._entities:
                continue
            key = (source, target)
            acc = self._edges.get(key)
            if acc is None:
                acc = _EdgeAcc(source, target)
                self._edges[key] = acc
            acc.weight += 1
            if raw.description and raw.description not in acc.descriptions:
                acc.descriptions.append(raw.description)
            acc.mentions.add(chunk_id)

        for raw in extraction.claims:
            subject = canonical_name(raw.subject)
            if subject not in self._entities:
                continue
            obj = canonical_name(raw.object) if raw.object else ""
            object_id: str | None = None
            if obj:
                if obj not in self._entities or obj == subject:
                    object_id = None
                else:
                    object_id = self.entity_id_for(obj)
            subject_id = self.entity_id_for(subject)
            # Identity is the claim content (not the source chunk): the
            # same statement re-extracted in another chunk is one claim;
            # first-seen chunk wins provenance.
            claim_id = "cl_" + hashlib.sha1(
                f"{subject_id}/{object_id or ''}/{raw.claim_type}/"
                f"{raw.status}/{raw.statement}".encode()
            ).hexdigest()[:16]
            if claim_id in self._claim_ids:
                continue
            self._claim_ids.add(claim_id)
            self._claims.append(
                BuiltClaim(
                    claim_id=claim_id,
                    subject_id=subject_id,
                    object_id=object_id,
                    chunk_id=chunk_id,
                    claim_type=raw.claim_type,
                    status=raw.status,
                    statement=raw.statement,
                    created_ts=self._now,
                )
            )

    def build(self) -> BuiltGraph:
        """Resolve accumulators into sorted, deduplicated persist records."""
        id_for_name = {
            name: self.entity_id_for(name) for name in self._entities
        }
        entities: list[BuiltEntity] = []
        for name in sorted(self._entities):
            acc = self._entities[name]
            entities.append(
                BuiltEntity(
                    entity_id=id_for_name[name],
                    name=name,
                    entity_type=acc.entity_type,
                    description=_join_descriptions(acc.descriptions),
                    created_ts=self._now,
                    updated_ts=self._now,
                )
            )

        edges: list[BuiltEdge] = []
        edge_mentions: list[tuple[str, str]] = []
        for source, target in sorted(self._edges):
            acc = self._edges[(source, target)]
            source_id, target_id = id_for_name[source], id_for_name[target]
            edges.append(
                BuiltEdge(
                    edge_id=self._edge_id(source_id, target_id),
                    source_id=source_id,
                    target_id=target_id,
                    description=_join_descriptions(acc.descriptions),
                    weight=acc.weight,
                    created_ts=self._now,
                    updated_ts=self._now,
                )
            )
            for chunk_id in acc.mentions:
                edge_mentions.append((edges[-1].edge_id, chunk_id))

        entity_mentions: list[tuple[str, str]] = []
        for name in sorted(self._entities):
            entity_id = id_for_name[name]
            for chunk_id in self._entities[name].mentions:
                entity_mentions.append((entity_id, chunk_id))

        return BuiltGraph(
            entities=entities,
            edges=edges,
            claims=list(self._claims),
            entity_mentions=sorted(entity_mentions),
            edge_mentions=sorted(edge_mentions),
        )


def _join_descriptions(descriptions: list[str]) -> str:
    """Join distinct descriptions, bounded in count and total length."""
    kept: list[str] = []
    used = 0
    for text in descriptions[:MAX_DESCRIPTIONS]:
        if used + len(text) > MAX_DESCRIPTION_CHARS:
            break
        kept.append(text)
        used += len(text) + 1
    return "\n".join(kept)
