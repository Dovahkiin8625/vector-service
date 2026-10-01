"""Merge per-chunk extractions into one deterministic graph."""
from __future__ import annotations

from vector_service.graph.extraction import (
    ChunkExtraction,
    RawClaim,
    RawEdge,
    RawEntity,
)
from vector_service.graph.merge import GraphBuilder, canonical_name


def _extraction(*, entities=(), edges=(), claims=()):
    return ChunkExtraction(
        entities=[RawEntity(*e) for e in entities],
        edges=[RawEdge(*e) for e in edges],
        claims=[RawClaim(*c) for c in claims],
    )


def _builder():
    return GraphBuilder("default", "ingest", now=1000.0)


def test_same_entity_merges_across_chunks():
    b = _builder()
    b.add_chunk("d1_0", _extraction(entities=[
        ("Alice", "person", "an analyst"),
    ]))
    b.add_chunk("d1_1", _extraction(entities=[
        ("Alice", "person", "team lead"),
        ("Acme", "org", "a vendor"),
    ]))
    graph = b.build()
    assert [e.name for e in graph.entities] == ["Acme", "Alice"]
    alice = next(e for e in graph.entities if e.name == "Alice")
    # Both distinct descriptions kept; both provenance mentions.
    assert "an analyst" in alice.description
    assert "team lead" in alice.description
    mentions = {m for m in graph.entity_mentions
                if m[0] == alice.entity_id}
    assert mentions == {(alice.entity_id, "d1_0"),
                        (alice.entity_id, "d1_1")}


def test_first_entity_type_wins():
    b = _builder()
    b.add_chunk("d1_0", _extraction(entities=[("Pat", "person", "")]))
    b.add_chunk("d1_1", _extraction(entities=[("Pat", "role", "later")]))
    pat = next(e for e in b.build().entities if e.name == "Pat")
    assert pat.entity_type == "person"


def test_entity_ids_are_deterministic():
    b1, b2 = _builder(), _builder()
    b1.add_chunk("d1_0", _extraction(entities=[("Alice", "person", "x")]))
    b2.add_chunk("d9_9", _extraction(entities=[("Alice", "person", "y")]))
    g1, g2 = b1.build(), b2.build()
    assert g1.entities[0].entity_id == g2.entities[0].entity_id
    assert g1.entities[0].entity_id == b1.entity_id_for("Alice")
    assert g1.entities[0].entity_id.startswith("ge_")


def test_dangling_edge_is_dropped():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", "x")],
        edges=[("Alice", "Ghost", "knows")],
    ))
    graph = b.build()
    assert graph.edges == []
    assert graph.edge_mentions == []


def test_self_edge_is_dropped():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", "x")],
        edges=[("Alice", "Alice", "self")],
    ))
    assert b.build().edges == []


def test_edge_accumulates_weight_and_descriptions():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", ""), ("Bob", "person", "")],
        edges=[("Alice", "Bob", "meets")],
    ))
    b.add_chunk("d1_1", _extraction(
        entities=[("Alice", "person", ""), ("Bob", "person", "")],
        edges=[("Alice", "Bob", "meets again")],
    ))
    graph = b.build()
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.weight == 2
    assert "meets" in edge.description
    assert "meets again" in edge.description
    assert (edge.edge_id, "d1_0") in graph.edge_mentions
    assert (edge.edge_id, "d1_1") in graph.edge_mentions


def test_claim_with_unknown_subject_dropped():
    b = _builder()
    b.add_chunk("d1_0", _extraction(claims=[
        ("Ghost", None, "fact", "true", "unsupported statement"),
    ]))
    assert b.build().claims == []


def test_claim_object_unknown_resolves_to_none_but_kept():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", "")],
        claims=[("Alice", "Phantom", "mention", "suspect",
                 "Alice refers to Phantom")],
    ))
    claims = b.build().claims
    assert len(claims) == 1
    assert claims[0].object_id is None
    assert claims[0].chunk_id == "d1_0"
    assert claims[0].status == "suspect"


def test_claim_object_resolves_to_entity_id():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", ""), ("Acme", "org", "")],
        claims=[("Alice", "Acme", "employment", "true", "works there")],
    ))
    graph = b.build()
    claims = graph.claims
    acme = next(e for e in graph.entities if e.name == "Acme")
    assert claims[0].object_id == acme.entity_id


def test_duplicate_claim_dedup_within_builder():
    b = _builder()
    b.add_chunk("d1_0", _extraction(
        entities=[("Alice", "person", "")],
        claims=[("Alice", None, "fact", "true", "same statement")],
    ))
    b.add_chunk("d1_1", _extraction(
        entities=[("Alice", "person", "")],
        claims=[("Alice", None, "fact", "true", "same statement")],
    ))
    assert len(b.build().claims) == 1


def test_canonical_name_collapses_whitespace():
    assert canonical_name("  Alice   B.  Cooke \n") == "Alice B. Cooke"
