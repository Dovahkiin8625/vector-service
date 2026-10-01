"""Recall channels: dense ANN leg and BM25 full-text leg."""
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


class FakeStore:
    def __init__(self):
        self.search_calls = []
        self.text_calls = []

    def search(self, database, collection, vector_field, query_vector,
               top_k=10, filter_expr=None, output_fields=None):
        self.search_calls.append({
            "database": database, "collection": collection,
            "vector_field": vector_field, "vector": query_vector,
            "top_k": top_k, "filter_expr": filter_expr,
        })
        return [Hit(id="c1", score=0.9, fields={"text": "a"}),
                Hit(id="c2", score=0.8, fields={"text": "b"})]

    def search_text(self, database, collection, sparse_field, query_text,
                    top_k=10, filter_expr=None, output_fields=None):
        self.text_calls.append({
            "database": database, "collection": collection,
            "sparse_field": sparse_field, "query": query_text,
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


def test_bm25_channel_sends_raw_text():
    store = FakeStore()
    ch = BM25Channel(store, "default", "ingest")
    run = ch.recall(RecallSpec("季度营收"), top_k=15)
    assert store.text_calls[0]["query"] == "季度营收"
    assert store.text_calls[0]["sparse_field"] == "sparse"
    assert store.text_calls[0]["top_k"] == 15
    assert run.channel == "bm25"
    assert run.hits[0].chunk_id == "c3"
    assert run.hits[0].rank == 1
