"""RetrievalPipeline orchestration with in-memory fakes."""
from __future__ import annotations

import pytest

import asyncio
import functools

from vector_service.retrieval.pipeline import RetrievalPipeline, build_filter_expr
from vector_service.schemas.retrieval import RetrievalRequest
from vector_service.stores.base import CollectionInfo, Hit


def async_test(coro):
    """Run an async test via asyncio.run (no pytest-asyncio in this repo)."""

    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


_SPARSE_INFO_FIELDS = [
    {"name": "id"}, {"name": "text"}, {"name": "sparse"},
]
_V1_INFO_FIELDS = [{"name": "id"}, {"name": "text"}]


class FakeStore:
    def __init__(self, v2=True):
        self.v2 = v2
        self.search_calls = []
        self.text_calls = []

    def collection_info(self, db, coll):
        return CollectionInfo(
            database=db, name=coll, dim=4, metric="cosine", count=0,
            fields=list(_SPARSE_INFO_FIELDS if self.v2 else _V1_INFO_FIELDS),
        )

    def search(self, db, coll, field, vector, top_k=10,
               filter_expr=None, output_fields=None):
        self.search_calls.append({"vector": vector, "top_k": top_k,
                                  "filter_expr": filter_expr})
        return [Hit(id="c1", score=0.9, fields={
            "text": "dense text", "doc_id": "d1", "chunk_index": 0,
            "section_header": "S1", "page_number": 1, "filename": "a.pdf",
        })]

    def search_text(self, db, coll, field, query, top_k=10,
                    filter_expr=None, output_fields=None):
        self.text_calls.append({"query": query, "top_k": top_k,
                                "filter_expr": filter_expr})
        return [Hit(id="c2", score=5.0, fields={
            "text": "lexical text", "doc_id": "d1", "chunk_index": 1,
            "section_header": "S2", "page_number": 2, "filename": "a.pdf",
        })]


class FakeEmbedder:
    model_name = "fake"
    dim = 4

    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeReranker:
    _impl = object()

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit

        return [ScoredHit(index=0, score=0.99)][: top_n or len(documents)]


class FakeSettings:
    pass


def _pipe(**kwargs):
    defaults = {
        "settings": FakeSettings(),
        "store": FakeStore(),
        "embedder": FakeEmbedder(),
        "reranker": FakeReranker(),
        "chat_check": lambda s: True,
        "chat_getter": lambda s: None,
    }
    defaults.update(kwargs)
    return RetrievalPipeline(**defaults)


def _req(**kwargs):
    kwargs.setdefault("query", "季度营收")
    return RetrievalRequest(**kwargs)


@async_test
async def test_dense_only_basic_retrieval():
    pipe = _pipe()
    result = await pipe.retrieve(_req(
        channels={"dense": True, "bm25": False},
        rerank={"enabled": False},
    ))
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.chunks[0].matched_channels == ["dense"]
    assert [tr.stage for tr in result.traces] == ["rewrite", "recall", "fuse"]
    assert len(result.channel_runs) == 1


@async_test
async def test_hybrid_fans_out_two_legs():
    pipe = _pipe()
    result = await pipe.retrieve(_req(rerank={"enabled": False}))
    channels = {run.channel for run in result.channel_runs}
    assert channels == {"dense", "bm25"}
    assert len(result.chunks) == 2
    c2 = next(c for c in result.chunks if c.chunk_id == "c2")
    assert c2.matched_channels == ["bm25"]


@async_test
async def test_rerank_reorders_and_truncates():
    pipe = _pipe()
    result = await pipe.retrieve(_req(top_k=1))
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.chunks[0].rerank_score == 0.99
    assert [tr.stage for tr in result.traces][-1] == "rerank"


@async_test
async def test_bm25_on_v1_collection_raises_with_migration_hint():
    pipe = _pipe(store=FakeStore(v2=False))
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req())
    assert exc.value.status_code == 422
    error = exc.value.detail["error"]
    assert error["code"] == "retrieval_channel_unsupported"
    assert error["migration_available"] is True


@async_test
async def test_missing_embedder_is_503():
    pipe = _pipe(embedder=None)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req())
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "embedder_unavailable"


@async_test
async def test_missing_reranker_is_503():
    pipe = _pipe(reranker=None)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req(rerank={"enabled": True}))
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "reranker_not_loaded"


@async_test
async def test_rewrite_without_llm_is_503():
    pipe = _pipe(chat_check=lambda s: False)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req(rewrite={"enabled": True}))
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "llm_unavailable"


class FakeChatClient:
    def __init__(self):
        self.calls = 0

    def as_chat_fn(self):
        def fn(messages):
            self.calls += 1
            if self.calls == 1:
                return '{"queries": ["v1", "v2"]}'
            return '{"query": "broader?"}'

        return fn


@async_test
async def test_multi_query_and_step_back_add_legs():
    chat = FakeChatClient()
    pipe = _pipe(chat_getter=lambda s: chat)
    result = await pipe.retrieve(_req(
        rewrite={"enabled": True, "methods": ["multi_query", "step_back"]},
        rerank={"enabled": False},
    ))
    # dense legs: original + 2 variants + step-back = 4; bm25 legs likewise 4
    by_channel = {"dense": 0, "bm25": 0}
    for run in result.channel_runs:
        by_channel[run.channel] += 1
    assert by_channel == {"dense": 4, "bm25": 4}
    queries = set(result.plan.lexical_queries)
    assert {"季度营收", "v1", "v2", "broader?"} <= queries


@async_test
async def test_mmr_stage_runs_when_enabled():
    pipe = _pipe()
    result = await pipe.retrieve(_req(
        mmr={"enabled": True, "lambda_mult": 0.7},
        rerank={"enabled": False},
    ))
    assert any(tr.stage == "mmr" for tr in result.traces)


@async_test
async def test_emit_receives_stage_events():
    pipe = _pipe()
    events = []
    await pipe.retrieve(
        _req(channels={"dense": True, "bm25": False},
             rerank={"enabled": False}),
        emit=events.append,
    )
    assert [e["stage"] for e in events] == ["rewrite", "recall", "fuse"]


def test_build_filter_expr():
    assert build_filter_expr(__import__(
        "vector_service.schemas.retrieval", fromlist=["FilterSpec"]
    ).FilterSpec()) is None
    expr = build_filter_expr(__import__(
        "vector_service.schemas.retrieval", fromlist=["FilterSpec"]
    ).FilterSpec(doc_id=" d1 ", filename="年 报"))
    assert expr == 'doc_id == "d1" and filename like "%年 报%"'
