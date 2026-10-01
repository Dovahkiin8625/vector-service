"""GraphChannel: entity/community ANN mapped back to provenance chunks.

Real :class:`CorpusRepository` (seeded document + graph rows) and an
in-memory store fake standing in for the two thin graph collections.
"""
from __future__ import annotations

import pytest

from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.graph.communities import community_id_for, detect_communities
from vector_service.graph.extraction import (
    ChunkExtraction,
    RawEdge,
    RawEntity,
)
from vector_service.graph.merge import GraphBuilder
from vector_service.graph.names import graph_collection_names
from vector_service.retrieval.base import MetaPredicate, RecallSpec
from vector_service.retrieval.channels import GraphChannel
from vector_service.stores.base import Hit

# ---- fakes -------------------------------------------------------------


class FakeStore:
    """Graph ANN hits keyed by physical collection (id-only projection)."""

    def __init__(self, *, plan):
        # plan: {collection: [(id, score), ...]}
        self._plan = plan
        self.calls = []

    def search(self, database, collection, field, vector, *,
               top_k=10, filter_expr=None, output_fields=None):
        self.calls.append({
            "collection": collection, "top_k": top_k,
            "output_fields": output_fields,
        })
        return [
            Hit(id=entity_id, score=score, fields={"id": entity_id})
            for entity_id, score in self._plan.get(collection, [])
        ]


# ---- fixture -----------------------------------------------------------


@pytest.fixture
def seeded(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    doc = DocumentRecord(
        doc_id="d1", database="default", collection="ingest",
        filename="年报.pdf", mime="application/pdf",
        content_hash="a" * 64, title="年度报告", author="张三",
        page_count=10,
    )
    chunks = [
        ChunkRecord(chunk_id="d1_0", doc_id="d1", database="default",
                    collection="ingest", chunk_index=0, text="第一段",
                    page_number=1, char_start=0, char_end=50, token_count=5),
        ChunkRecord(chunk_id="d1_1", doc_id="d1", database="default",
                    collection="ingest", chunk_index=1, text="第二段",
                    page_number=2, char_start=50, char_end=100, token_count=5),
    ]
    repo.store_document(doc, chunks)

    builder = GraphBuilder("default", "ingest", now=1000.0)
    builder.add_chunk("d1_0", ChunkExtraction(
        entities=[
            RawEntity("Alice", "person", "an analyst"),
            RawEntity("Bob", "person", "a colleague"),
        ],
        edges=[RawEdge("Alice", "Bob", "works with")],
    ))
    builder.add_chunk("d1_1", ChunkExtraction(
        entities=[
            RawEntity("Alice", "person", "team lead"),
            RawEntity("Bob", "person", "a colleague"),
        ],
        edges=[RawEdge("Alice", "Bob", "collaborates")],
    ))
    built = builder.build()

    member_sets = detect_communities(built.entities, built.edges)
    members = next(m for m in member_sets if len(m) >= 2)
    cid = community_id_for("default", "ingest", members)
    from types import SimpleNamespace

    community_row = SimpleNamespace(
        community_id=cid, community_index=0,
        summary="Alice 与 Bob 的合作社区", created_ts=1000.0,
    )
    repo.replace_graph(
        "default", "ingest",
        entities=built.entities, entity_mentions=built.entity_mentions,
        edges=built.edges, edge_mentions=built.edge_mentions,
        claims=[], communities=[community_row],
        community_members=[(cid, eid) for eid in members],
    )

    id_for = {e.name: e.entity_id for e in built.entities}
    entity_coll, community_coll = graph_collection_names("ingest")
    return {
        "repo": repo,
        "ids": id_for,
        "community_id": cid,
        "entity_coll": entity_coll,
        "community_coll": community_coll,
    }


class _Embedder:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]


def _channel(seeded, plan):
    store = FakeStore(plan=plan)
    channel = GraphChannel(
        store=store, repo=seeded["repo"], database="default",
        entity_collection=seeded["entity_coll"],
        community_collection=seeded["community_coll"],
        embedder=_Embedder(),
    )
    return store, channel


# ---- tests -------------------------------------------------------------


def test_graph_hits_map_back_to_leaf_chunks(seeded):
    ids = seeded["ids"]
    plan = {
        seeded["entity_coll"]: [(ids["Alice"], 0.9), (ids["Bob"], 0.75)],
        seeded["community_coll"]: [(seeded["community_id"], 0.7)],
    }
    store, channel = _channel(seeded, plan)
    run = channel.recall(RecallSpec(query="Alice 关系"), top_k=10)

    assert run.channel == "graph"
    assert run.query == "Alice 关系"
    # Output stays in the chunk vocabulary; both provenance chunks rank.
    assert [h.chunk_id for h in run.hits] == ["d1_0", "d1_1"]
    top = run.hits[0]
    names = [e["name"] for e in top.fields["graph_entities"]]
    assert names == ["Alice", "Bob"]
    alice = top.fields["graph_entities"][0]
    assert alice["entity_type"] == "person"
    assert alice["description"]  # hydrated from the corpus
    assert alice["score"] == 0.9
    communities = top.fields["graph_communities"]
    assert len(communities) == 1
    assert communities[0]["community_index"] == 0
    assert communities[0]["summary"]
    assert communities[0]["score"] == 0.7
    # Both graph legs used the id-only projection.
    assert all(c["output_fields"] == ["id"] for c in store.calls)


def test_doc_ids_filter_excludes_everything(seeded):
    ids = seeded["ids"]
    plan = {seeded["entity_coll"]: [(ids["Alice"], 0.9)]}
    _, channel = _channel(seeded, plan)
    run = channel.recall(
        RecallSpec(query="q"), top_k=10, doc_ids={"d9"}
    )
    assert run.hits == []


def test_chunk_index_predicate_filters_mentions(seeded):
    ids = seeded["ids"]
    plan = {
        seeded["entity_coll"]: [(ids["Alice"], 0.9), (ids["Bob"], 0.75)],
        seeded["community_coll"]: [(seeded["community_id"], 0.7)],
    }
    _, channel = _channel(seeded, plan)
    run = channel.recall(
        RecallSpec(query="q"), top_k=10,
        index_predicates=[MetaPredicate("chunk_index", "==", "1")],
    )
    assert [h.chunk_id for h in run.hits] == ["d1_1"]
    # All provenance on d1_1: entities + community survive the predicate.
    top = run.hits[0]
    assert top.fields["graph_entities"] != []
    assert top.fields["graph_communities"] != []


def test_community_only_hit_without_direct_entity_still_ranks(seeded):
    # ANN only matched the community; its members resolve through
    # community_members even though neither entity was a direct ANN hit.
    plan = {seeded["community_coll"]: [(seeded["community_id"], 0.7)]}
    _, channel = _channel(seeded, plan)
    run = channel.recall(RecallSpec(query="q"), top_k=10)
    assert [h.chunk_id for h in run.hits] == ["d1_0", "d1_1"]
    for hit in run.hits:
        assert hit.fields["graph_communities"][0]["score"] == 0.7
        assert hit.fields["graph_entities"] == []


def test_no_graph_hits_returns_empty_run(seeded):
    _, channel = _channel(seeded, {})
    run = channel.recall(RecallSpec(query="q"), top_k=10)
    assert run.hits == []
