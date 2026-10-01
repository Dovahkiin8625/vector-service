"""RetrievalPipeline observability (TODO §7).

One retrieval carries both an OpenTelemetry span tree and Prometheus
metrics. Pins:

- spans cover rewrite → per-channel recall legs → hydrate → rerank in
  one trace, children linked under a ``retrieval`` root;
- root/stage attributes describe the request;
- counters observe request volume, per-channel leg outcome and raw
  hit volume, fusion in/out (post-fusion survival), per-request channel
  hits, and the rerank stage (duration/chars/candidates).
"""
from __future__ import annotations

import asyncio
import functools

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from tests.unit.test_retrieval_pipeline_advanced import (
    _INFO_FIELDS,
    FakeEmbedder,
    FakeSettings,
    FakeStore,
    RecordingReranker,
)
from vector_service.core.metrics import REGISTRY
from vector_service.retrieval.pipeline import _TRACER, RetrievalPipeline
from vector_service.schemas.retrieval import RetrievalRequest


def async_test(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


@pytest.fixture
def recorded():
    """Install an SDK provider with an in-memory exporter for one test."""
    prev = trace._TRACER_PROVIDER
    prev_done = trace._TRACER_PROVIDER_SET_ONCE._done
    prev_real_tracer = _TRACER._real_tracer
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    # The pipeline's module-level proxy tracer caches the first real
    # tracer it resolves; drop that cache so it rebinds to this provider.
    _TRACER._real_tracer = None
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        provider.shutdown()
        trace._TRACER_PROVIDER = prev
        trace._TRACER_PROVIDER_SET_ONCE._done = prev_done
        _TRACER._real_tracer = prev_real_tracer


def _pipeline(*, reranker=None):
    store = FakeStore(
        info_fields=_INFO_FIELDS,
        dense_hits=[("d1_0", 0.9, {"text": "季度营收内容"})],
    )
    pipe = RetrievalPipeline(
        settings=FakeSettings(),
        store=store,
        embedder=FakeEmbedder(),
        reranker=reranker,
    )
    return store, pipe


def _span_map(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def _value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ---- spans -------------------------------------------------------------


@async_test
async def test_spans_cover_stages_in_one_trace(recorded):
    _, pipe = _pipeline()

    await pipe.retrieve(
        RetrievalRequest(query="季度营收", rerank={"enabled": False})
    )

    spans = recorded.get_finished_spans()
    by_name = {span.name: span for span in spans}
    expected = {
        "retrieval", "rewrite", "recall", "recall.dense", "recall.bm25",
        "fuse", "hydrate",
    }
    assert expected <= set(by_name)

    root = by_name["retrieval"]
    assert root.parent is None
    trace_id = root.context.trace_id
    # Every span belongs to the same trace and links back towards root.
    for span in spans:
        assert span.context.trace_id == trace_id
    assert by_name["recall.dense"].parent.span_id == by_name["recall"].context.span_id
    assert by_name["recall.bm25"].parent.span_id == by_name["recall"].context.span_id
    assert by_name["hydrate"].parent.span_id == by_name["fuse"].context.span_id
    assert by_name["rewrite"].parent.span_id == root.context.span_id


@async_test
async def test_root_span_attributes(recorded):
    _, pipe = _pipeline()

    await pipe.retrieve(
        RetrievalRequest(query="季度营收", top_k=5, rerank={"enabled": False})
    )

    attrs = _span_map(recorded)["retrieval"].attributes
    assert attrs["vs.database"] == "default"
    assert attrs["vs.collection"] == "ingest"
    assert attrs["vs.physical_collection"] == "ingest"
    assert attrs["vs.top_k"] == 5
    assert attrs["vs.query_length"] == 4

    leg = _span_map(recorded)["recall.dense"]
    assert leg.attributes["vs.query_length"] == 4


@async_test
async def test_rerank_stage_has_span(recorded):
    _, pipe = _pipeline(reranker=RecordingReranker())

    await pipe.retrieve(RetrievalRequest(query="季度营收"))

    assert "retrieval.rerank" in _span_map(recorded)


# ---- metrics -----------------------------------------------------------


@async_test
async def test_metrics_record_request_channel_and_fusion(recorded):
    _, pipe = _pipeline()

    before_req = _value("vs_retrieval_requests_total")
    before_runs = _value(
        "vs_retrieval_channel_runs_total",
        channel="dense", status="ok",
    )
    before_empty = _value(
        "vs_retrieval_channel_runs_total",
        channel="bm25", status="empty",
    )
    before_hits = _value(
        "vs_retrieval_recalled_hits_total", channel="dense"
    )
    before_fuse_in = _value(
        "vs_retrieval_fusion_input_total", method="rrf"
    )
    before_fuse_out = _value(
        "vs_retrieval_fusion_output_total", method="rrf"
    )
    before_hit_req = _value(
        "vs_retrieval_channel_hit_requests_total", channel="dense"
    )

    await pipe.retrieve(
        RetrievalRequest(query="季度营收", rerank={"enabled": False})
    )

    assert _value("vs_retrieval_requests_total") - before_req == 1
    assert (
        _value("vs_retrieval_channel_runs_total",
               channel="dense", status="ok") - before_runs == 1
    )
    assert (
        _value("vs_retrieval_channel_runs_total",
               channel="bm25", status="empty") - before_empty == 1
    )
    assert (
        _value("vs_retrieval_recalled_hits_total", channel="dense")
        - before_hits == 1
    )
    assert (
        _value("vs_retrieval_fusion_input_total", method="rrf")
        - before_fuse_in == 1
    )
    assert (
        _value("vs_retrieval_fusion_output_total", method="rrf")
        - before_fuse_out == 1
    )
    assert (
        _value("vs_retrieval_channel_hit_requests_total", channel="dense")
        - before_hit_req == 1
    )
    # Empty channel doesn't count as a per-request hit.
    assert REGISTRY.get_sample_value(
        "vs_retrieval_channel_hit_requests_total",
        {"channel": "bm25"},
    ) is None or _value(
        "vs_retrieval_channel_hit_requests_total", channel="bm25"
    ) == 0.0


@async_test
async def test_metrics_record_rerank_stage(recorded):
    _, pipe = _pipeline(reranker=RecordingReranker())

    before_count = _value(
        "vs_retrieval_rerank_candidates_count", model="unknown"
    )
    before_chars = _value(
        "vs_retrieval_rerank_input_chars_count", model="unknown"
    )
    before_dur = _value(
        "vs_retrieval_rerank_stage_duration_seconds_count", model="unknown"
    )

    await pipe.retrieve(RetrievalRequest(query="季度营收"))

    assert (
        _value("vs_retrieval_rerank_candidates_count", model="unknown")
        - before_count == 1
    )
    assert (
        _value("vs_retrieval_rerank_input_chars_count", model="unknown")
        - before_chars == 1
    )
    assert (
        _value("vs_retrieval_rerank_stage_duration_seconds_count",
               model="unknown")
        - before_dur == 1
    )
