"""Recall channels: dense ANN leg and BM25 sparse-vector leg."""
from __future__ import annotations

from vector_service.retrieval.base import RecallSpec
from vector_service.retrieval.channels import BM25Channel, DenseChannel
from vector_service.stores.base import Hit


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


class FakeBM25:
    def __init__(self, query_sparse=None):
        self.calls = []
        self._query_sparse = {1: 1.0} if query_sparse is None else query_sparse

    def encode_query(self, repo, database, logical_collection,
                     physical_collection, query):
        self.calls.append({
            "database": database,
            "logical_collection": logical_collection,
            "physical_collection": physical_collection,
            "query": query,
        })
        return dict(self._query_sparse)


class RaisingBM25:
    def encode_query(self, repo, database, logical_collection,
                     physical_collection, query):
        raise RuntimeError("cannot build BM25 stats: no chunks")


class FakeStore:
    def __init__(self):
        self.search_calls = []
        self.sparse_calls = []

    def search(self, database, collection, vector_field, query_vector,
               top_k=10, filter_expr=None, output_fields=None):
        self.search_calls.append({
            "database": database, "collection": collection,
            "vector_field": vector_field, "vector": query_vector,
            "top_k": top_k, "filter_expr": filter_expr,
        })
        return [Hit(id="c1", score=0.9, fields={"text": "a"}),
                Hit(id="c2", score=0.8, fields={"text": "b"})]

    def search_sparse(self, database, collection, sparse_field, query_sparse,
                      top_k=10, filter_expr=None, output_fields=None):
        self.sparse_calls.append({
            "database": database, "collection": collection,
            "sparse_field": sparse_field, "query_sparse": query_sparse,
            "top_k": top_k, "filter_expr": filter_expr,
        })
        return [Hit(id="c3", score=5.0, fields={"text": "c"})]


def test_dense_embeds_query_and_assigns_ranks():
    store, emb = FakeStore(), FakeEmbedder()
    ch = DenseChannel(store, "default", "ingest", emb)
    run = ch.recall(RecallSpec("季度营收"), top_k=20,
                    filter_expr='doc_id == "d1"')
    assert emb.queries == ["季度营收"]
    assert store.search_calls[0]["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert store.search_calls[0]["top_k"] == 20
    assert store.search_calls[0]["filter_expr"] == 'doc_id == "d1"'
    assert run.channel == "dense"
    assert run.query == "季度营收"
    assert [h.chunk_id for h in run.hits] == ["c1", "c2"]
    assert [h.rank for h in run.hits] == [1, 2]


def test_dense_skips_embedder_when_vector_provided():
    store, emb = FakeStore(), FakeEmbedder()
    ch = DenseChannel(store, "default", "ingest", emb)
    ch.recall(RecallSpec("季度营收", vector=[0.5, 0.5, 0.5, 0.5]), top_k=10)
    assert emb.queries == []
    assert store.search_calls[0]["vector"] == [0.5, 0.5, 0.5, 0.5]


def test_bm25_channel_encodes_query_and_searches_sparse():
    store, enc = FakeStore(), FakeBM25(query_sparse={7: 2.5, 3: 1.0})
    ch = BM25Channel(store, "default", "ingest", enc,
                     logical_collection="年报库")
    run = ch.recall(RecallSpec("季度营收"), top_k=15)
    # The query is encoded client-side against the logical collection's
    # stats (cached under the physical name), never sent as raw text.
    assert enc.calls[0]["query"] == "季度营收"
    assert enc.calls[0]["logical_collection"] == "年报库"
    assert enc.calls[0]["physical_collection"] == "ingest"
    assert store.sparse_calls[0]["query_sparse"] == {7: 2.5, 3: 1.0}
    assert store.sparse_calls[0]["sparse_field"] == "sparse"
    assert store.sparse_calls[0]["top_k"] == 15
    assert run.channel == "bm25"
    assert run.hits[0].chunk_id == "c3"
    assert run.hits[0].rank == 1


def test_bm25_channel_defaults_logical_to_physical():
    store, enc = FakeStore(), FakeBM25()
    ch = BM25Channel(store, "default", "ingest", enc)
    ch.recall(RecallSpec("季度营收"), top_k=10)
    assert enc.calls[0]["logical_collection"] == "ingest"
    assert enc.calls[0]["physical_collection"] == "ingest"


def test_bm25_channel_without_stats_returns_empty():
    store = FakeStore()
    ch = BM25Channel(store, "default", "ingest", RaisingBM25())
    run = ch.recall(RecallSpec("季度营收"), top_k=10)
    assert run.channel == "bm25"
    assert run.hits == []
    assert store.sparse_calls == []
