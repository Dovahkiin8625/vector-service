"""Weighted label-propagation community detection."""
from __future__ import annotations

from types import SimpleNamespace

from vector_service.graph.communities import (
    community_id_for,
    detect_communities,
)


def _node(entity_id):
    return SimpleNamespace(entity_id=entity_id)


def _edge(source, target, weight=1):
    return SimpleNamespace(source_id=source, target_id=target, weight=weight)


def _partition(edges, ids):
    return detect_communities(
        [_node(i) for i in ids], edges
    )


def test_connected_components_become_communities():
    ids = ["a", "b", "c", "d", "e", "f"]
    edges = [
        _edge("a", "b"), _edge("b", "c"), _edge("a", "c"),
        _edge("d", "e"),
    ]
    communities = _partition(edges, ids)
    assert communities == [["a", "b", "c"], ["d", "e"], ["f"]]


def test_detection_is_deterministic():
    ids = ["a", "b", "c", "d"]
    edges = [_edge("a", "b"), _edge("c", "d")]
    first = _partition(edges, ids)
    second = _partition(edges, ids)
    assert first == second == [["a", "b"], ["c", "d"]]


def test_weighted_edges_still_one_community():
    ids = ["a", "b", "c"]
    edges = [
        _edge("a", "b", weight=5),
        _edge("a", "c", weight=1),
        _edge("b", "c", weight=1),
    ]
    assert _partition(edges, ids) == [["a", "b", "c"]]


def test_singleton_graph():
    assert _partition([], ["lonely"]) == [["lonely"]]


def test_empty_graph():
    assert detect_communities([], []) == []


def test_community_id_is_deterministic_and_scoped():
    members = ["a", "b"]
    cid = community_id_for("default", "ingest", members)
    assert cid.startswith("cm_")
    assert community_id_for("default", "ingest", list(members)) == cid
    assert community_id_for("default", "other", members) != cid
    assert community_id_for("default", "ingest", ["a", "c"]) != cid
