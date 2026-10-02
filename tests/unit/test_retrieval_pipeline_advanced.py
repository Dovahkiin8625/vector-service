"""RetrievalPipeline §4 orchestration: routing, hydration, compression.

Combines real :class:`CorpusRepository` (where predicates/hydration are
exercised) with in-memory store/model fakes. Pins:

- intent routing drives the channel set, availability-gated;
- graph leg wiring through a minimal repo fake;
- fused chunks hydrate content + citation fields from the corpus;
- token-budget compression keeps the top chunk regardless;
- metadata/filename predicates resolve via SQLite;
- LLM-router failure degrades to the heuristic decision.
"""
from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace

from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.retrieval.base import RetrievedChunk
from vector_service.retrieval.pipeline import (
    RetrievalPipeline,
    _compress_to_budget,
)
from vector_service.schemas.retrieval import RetrievalRequest, to_result
from vector_service.stores.base import CollectionInfo, Hit


def async_test(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


# ---- fakes -------------------------------------------------------------


class FakeStore:
    def __init__(self, *, info_fields, collection_names=(),
                 dense_hits=(), text_hits=()):
        self._info_fields = list(info_fields)
        self._collection_names = list(collection_names)
        self._dense = list(dense_hits)
        self._text = list(text_hits)
        self.search_calls = []
        self.sparse_calls = []

    def list_collections(self, database):
        return list(self._collection_names)

    def collection_info(self, db, coll):
        return CollectionInfo(
            database=db, name=coll, dim=4, metric="cosine", count=0,
            fields=list(self._info_fields),
        )

    def search(self, db, coll, field, vector, *, top_k=10,
               filter_expr=None, output_fields=None):
        self.search_calls.append({
            "collection": coll, "filter_expr": filter_expr,
            "output_fields": output_fields,
        })
        return [
            Hit(id=hit_id, score=score, fields=dict(fields))
            for hit_id, score, fields in self._dense
        ]

    def search_sparse(self, db, coll, field, query_sparse, *, top_k=10,
                      filter_expr=None, output_fields=None):
        self.sparse_calls.append({
            "filter_expr": filter_expr,
            "query_sparse": query_sparse,
        })
        return [
            Hit(id=hit_id, score=score, fields=dict(fields))
            for hit_id, score, fields in self._text
        ]


class FakeBM25:
    def __init__(self):
        self.queries = []

    def encode_query(self, repo, database, logical_collection,
                     physical_collection, query):
        self.queries.append(query)
        return {3: 1.5}


class FakeEmbedder:
    model_name = "fake"
    dim = 4

    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class RecordingReranker:
    _impl = object()

    def __init__(self):
        self.documents = None

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit

        self.documents = list(documents)
        return [
            ScoredHit(index=i, score=0.9 - 0.1 * i)
            for i in range(min(len(documents), top_n or len(documents)))
        ]


class FakeSettings:
    pass


_INFO_FIELDS = [
    {"name": "id"}, {"name": "text"}, {"name": "sparse"},
]
_INFO_FIELDS_SUMMARY = _INFO_FIELDS + [{"name": "summary_vector"}]


def _pipe(store, *, repo=None, reranker=None, bm25=None, **kwargs):
    return RetrievalPipeline(
        settings=FakeSettings(),
        store=store,
        embedder=FakeEmbedder(),
        reranker=reranker,
        repo=repo,
        bm25=FakeBM25() if bm25 is None else bm25,
        **kwargs,
    )


def _req(query="季度营收", **kwargs):
    return RetrievalRequest(query=query, **kwargs)


# ---- routing: channel selection ----------------------------------------


@async_test
async def test_semantic_routing_selects_dense_and_summary():
    store = FakeStore(
        info_fields=_INFO_FIELDS_SUMMARY,
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )
    pipe = _pipe(store)
    events = []
    result = await pipe.retrieve(
        _req("季度营收情况如何？", routing={"enabled": True},
             rerank={"enabled": False}),
        emit=events.append,
    )
    assert result.route is not None
    assert result.route.router == "heuristic"
    assert result.route.intents == ["semantic"]
    channels = {run.channel for run in result.channel_runs}
    assert channels == {"dense", "summary"}
    assert next(e["stage"] for e in events) == "route"


@async_test
async def test_graph_intent_without_graph_index_skips_graph_leg():
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )
    pipe = _pipe(store)
    result = await pipe.retrieve(
        _req("张三和李四是什么关系？", routing={"enabled": True},
             rerank={"enabled": False}),
    )
    assert result.route.graph is True
    channels = {run.channel for run in result.channel_runs}
    assert "graph" not in channels
    assert channels == {"dense"}


@async_test
async def test_routing_drops_bm25_leg_without_sparse_field():
    # Routing degrades best-effort: a collection without the sparse
    # field loses the bm25 leg instead of failing the request.
    store = FakeStore(
        info_fields=[{"name": "id"}, {"name": "text"}],
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )
    pipe = _pipe(store)
    result = await pipe.retrieve(
        _req("季度营收", routing={"enabled": True},
             rerank={"enabled": False}),
    )
    assert result.route.bm25 is True
    channels = {run.channel for run in result.channel_runs}
    assert channels == {"dense"}


class FakeGraphRepo:
    """Just enough repo surface for one graph leg."""

    def get_binding(self, db, coll):
        # Unbound: physical == logical.
        return None

    def graph_stats(self, db, coll):
        return {"entities_count": 1}

    def document_ids_for_predicates(self, db, coll, predicates):
        # No document-level constraints in this fake.
        return None

    def hydrate(self, chunk_ids):
        # Content/citation injection is not exercised here.
        return {}

    def hydrate_entities(self, ids):
        return {ids[0]: {
            "name": "Alice", "entity_type": "person", "description": "x",
        }}

    def hydrate_communities(self, ids):
        return {}

    def entity_mention_rows(self, ids):
        return [{
            "entity_id": ids[0], "chunk_id": "d1_0",
            "doc_id": "d1", "chunk_index": 0,
        }]


@async_test
async def test_graph_leg_runs_when_graph_is_available():
    from vector_service.graph.names import graph_collection_names

    entity_coll, community_coll = graph_collection_names("ingest")

    class GraphAwareStore(FakeStore):
        def search(self, db, coll, field, vector, *, top_k=10,
                   filter_expr=None, output_fields=None):
            if coll == entity_coll:
                return [Hit(id="ge_1", score=0.8, fields={"id": "ge_1"})]
            if coll == community_coll:
                return []
            return super().search(
                db, coll, field, vector, top_k=top_k,
                filter_expr=filter_expr, output_fields=output_fields,
            )

    store = GraphAwareStore(
        info_fields=_INFO_FIELDS,
        collection_names=[entity_coll, community_coll],
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )
    pipe = _pipe(store, repo=FakeGraphRepo())
    result = await pipe.retrieve(
        _req("张三和李四是什么关系？", routing={"enabled": True},
             rerank={"enabled": False}),
    )
    channels = {run.channel for run in result.channel_runs}
    assert "graph" in channels
    chunk = result.chunks[0]
    assert "graph" in chunk.matched_channels
    assert chunk.fields["graph_entities"][0]["name"] == "Alice"


# ---- hydration: citation back-link --------------------------------------


def _seeded_corpus(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    doc = DocumentRecord(
        doc_id="d1", database="default", collection="ingest",
        filename="年报.pdf", mime="application/pdf",
        content_hash="a" * 64, title="年度报告", author="张三", page_count=10,
    )
    chunks = [
        ChunkRecord(chunk_id="d1_0", doc_id="d1", database="default",
                    collection="ingest", chunk_index=0, text="第一段内容",
                    section_header="引言", page_number=1,
                    char_start=0, char_end=50, token_count=5),
        ChunkRecord(chunk_id="d1_1", doc_id="d1", database="default",
                    collection="ingest", chunk_index=1, text="第二段内容",
                    section_header="正文", page_number=2,
                    char_start=50, char_end=100, token_count=5),
    ]
    repo.store_document(doc, chunks)
    return repo


@async_test
async def test_hydrate_injects_content_and_citation_fields(tmp_path):
    repo = _seeded_corpus(tmp_path)
    # Hits arrive contentless (thin index holds no text/citation fields).
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {})],
        text_hits=[("d1_1", 5.0, {})],
    )
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(_req(rerank={"enabled": False}))
    by_id = {c.chunk_id: c.fields for c in result.chunks}
    first = by_id["d1_0"]
    assert first["text"] == "第一段内容"
    assert first["filename"] == "年报.pdf"
    assert first["title"] == "年度报告"
    assert first["author"] == "张三"
    assert first["page_number"] == 1
    assert first["char_start"] == 0
    assert first["char_end"] == 50


@async_test
async def test_rerank_runs_on_hydrated_text(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {}), ("d1_1", 0.8, {})],
    )
    reranker = RecordingReranker()
    pipe = _pipe(store, repo=repo, reranker=reranker)
    result = await pipe.retrieve(_req(top_k=2))
    assert reranker.documents == ["第一段内容", "第二段内容"]
    assert [c.chunk_id for c in result.chunks] == ["d1_0", "d1_1"]


# ---- compression -------------------------------------------------------


@async_test
async def test_compress_stage_truncates_to_token_budget(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {}), ("d1_1", 0.8, {})],
    )
    pipe = _pipe(store, repo=repo)
    events = []
    result = await pipe.retrieve(
        _req(rerank={"enabled": False}, context={"max_tokens": 8}),
        emit=events.append,
    )
    assert [e["stage"] for e in events][-1] == "compress"
    assert [c.chunk_id for c in result.chunks] == ["d1_0"]
    trace = next(tr for tr in result.traces if tr.stage == "compress")
    assert trace.detail == {"budget": 8, "kept": 1, "dropped": 1}


def _chunk(chunk_id, token_count):
    return RetrievedChunk(
        chunk_id=chunk_id, fields={"token_count": token_count},
        fusion_score=1.0, matched_channels=[],
    )


def test_compress_to_budget_unit():
    chunks = [_chunk("a", 100), _chunk("b", 5), _chunk("c", 5)]
    # Top chunk always kept even over budget; following chunks fit.
    assert [c.chunk_id for c in _compress_to_budget(chunks, 10)] == [
        "a", "b"
    ]


def test_compress_missing_token_count_counts_as_zero():
    chunks = [
        RetrievedChunk(chunk_id="a", fields={}, fusion_score=1.0,
                       matched_channels=[]),
        _chunk("b", 5),
    ]
    # Missing count counts as zero: a is kept; b is then cut by the budget.
    assert [c.chunk_id for c in _compress_to_budget(chunks, 3)] == ["a"]


# ---- predicates / filename via SQLite ----------------------------------


@async_test
async def test_metadata_predicate_no_match_returns_empty(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(info_fields=_INFO_FIELDS, dense_hits=[])
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(
        _req("年报 author:不存在的作者QQQ", routing={"enabled": True},
             rerank={"enabled": False}),
    )
    assert result.chunks == []
    # Planned legs survive as empty trace runs.
    assert all(run.hits == [] for run in result.channel_runs)


@async_test
async def test_filename_filter_resolves_to_doc_id_expr(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {})],
        text_hits=[("d1_1", 5.0, {})],
    )
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(
        _req(rerank={"enabled": False},
             filter={"filename": "年报.pdf"}),
    )
    assert len(result.chunks) == 2
    exprs = [c["filter_expr"] for c in store.search_calls]
    assert 'doc_id in ["d1"]' in exprs


@async_test
async def test_filename_filter_no_match_returns_empty(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(info_fields=_INFO_FIELDS)
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(
        _req(rerank={"enabled": False},
             filter={"filename": "no-such-file.zzz"}),
    )
    assert result.chunks == []


# ---- weighted fusion ---------------------------------------------------


@async_test
async def test_weighted_fusion_end_to_end(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {}), ("d1_1", 0.8, {})],
        text_hits=[("d1_1", 5.0, {})],
    )
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(_req(
        rerank={"enabled": False},
        fusion={"method": "weighted", "weights": {
            "dense": 0.8, "bm25": 0.4, "summary": 0.0, "graph": 0.0,
        }},
        top_k=5,
    ))
    assert len(result.chunks) == 2
    assert all(c.fusion_score > 0 for c in result.chunks)
    trace = next(tr for tr in result.traces if tr.stage == "fuse")
    assert trace.detail["method"] == "weighted"


# ---- LLM router degradation --------------------------------------------


@async_test
async def test_llm_classifier_failure_falls_back_to_heuristic():
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )

    def raising_chat():
        def fn(messages):
            raise RuntimeError("llm exploded")

        return fn

    pipe = _pipe(store, chat_getter=lambda s: SimpleNamespace(
        as_chat_fn=raising_chat), chat_check=lambda s: True)
    result = await pipe.retrieve(
        _req("季度营收情况如何？",
             routing={"enabled": True, "use_llm": True},
             rerank={"enabled": False}),
    )
    assert result.route.router == "heuristic"


@async_test
async def test_llm_classifier_decision_is_used():
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {"text": "x"})],
    )

    def chat():
        def fn(messages):
            return ('{"intents": ["semantic"], "filters": [],'
                    ' "query": "cleaned query"}')

        return fn

    pipe = _pipe(store, chat_getter=lambda s: SimpleNamespace(as_chat_fn=chat),
                 chat_check=lambda s: True)
    result = await pipe.retrieve(
        _req("raw surface",
             routing={"enabled": True, "use_llm": True},
             rerank={"enabled": False}),
    )
    assert result.route.router == "llm"
    # Recall ran on the LLM-cleaned query, not the raw surface.
    dense_runs = [r for r in result.channel_runs if r.channel == "dense"]
    assert dense_runs[0].query == "cleaned query"


# ---- response schema mapping -------------------------------------------


@async_test
async def test_to_result_maps_route(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {})],
    )
    pipe = _pipe(store, repo=repo)
    result = await pipe.retrieve(
        _req("季度营收情况如何？", routing={"enabled": True},
             rerank={"enabled": False}),
    )
    response = to_result(result)
    assert response.route is not None
    assert response.route.router == "heuristic"
    assert response.route.predicates == []
    payload = response.model_dump()
    assert payload["route"]["intents"] == ["semantic"]


@async_test
async def test_to_result_route_none_when_routing_disabled(tmp_path):
    repo = _seeded_corpus(tmp_path)
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {})],
    )
    result = await _pipe(store, repo=repo).retrieve(
        _req(rerank={"enabled": False})
    )
    assert to_result(result).route is None
