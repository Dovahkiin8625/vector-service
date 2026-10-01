"""Dataclasses shared across the retrieval pipeline."""
from __future__ import annotations

from vector_service.retrieval.base import (
    ChannelHit,
    ChannelRun,
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)


def test_recall_spec_defaults():
    spec = RecallSpec(query="季度营收")
    assert spec.query == "季度营收"
    assert spec.vector is None
    assert spec.hypothetical is None


def test_channel_run_carries_ranked_hits():
    run = ChannelRun(
        channel="dense",
        query="q",
        hits=[
            ChannelHit(chunk_id="c1", score=0.9, rank=1, fields={"text": "a"}),
            ChannelHit(chunk_id="c2", score=0.8, rank=2, fields={"text": "b"}),
        ],
    )
    assert run.hits[0].rank == 1
    assert run.hits[1].fields["text"] == "b"


def test_plan_and_result_defaults():
    plan = RetrievalPlan(
        original_query="q", dense_specs=[RecallSpec("q")], lexical_queries=["q"]
    )
    assert plan.sub_queries == []
    chunk = RetrievedChunk(
        chunk_id="c1", fields={"text": "a"}, fusion_score=0.5,
        matched_channels=["dense", "bm25"],
    )
    assert chunk.rerank_score is None
    result = RetrievalResult(
        query="q", chunks=[chunk], plan=plan, channel_runs=[], traces=[]
    )
    assert result.traces == []


def test_stage_trace_detail_defaults():
    tr = StageTrace(stage="fuse", duration_ms=12)
    assert tr.detail == {}
