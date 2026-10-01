"""Pydantic models for POST /v1/retrieval and its response."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    to_result,
)


def _base_kwargs(**overrides):
    kwargs = {"query": "q"}
    kwargs.update(overrides)
    return kwargs


def test_request_defaults():
    req = RetrievalRequest(query="q")
    assert req.database == "default"
    assert req.collection == "ingest"
    assert req.top_k == 10
    assert req.channels.dense and req.channels.bm25
    assert req.fusion.method == "rrf"
    assert req.fusion.rrf_k == 60
    assert req.rewrite.enabled is False
    assert req.mmr.enabled is False
    assert req.rerank.enabled is True
    assert req.rerank.candidate_pool == 25


def test_request_requires_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="")


@pytest.mark.parametrize("top_k", [0, 101])
def test_top_k_bounds(top_k):
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", top_k=top_k)


def test_at_least_one_channel():
    with pytest.raises(ValidationError, match="channel"):
        RetrievalRequest(query="q", channels={"dense": False, "bm25": False})


def test_fusion_method_enum():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={"method": "magic"})


def test_rrf_k_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={"method": "rrf", "rrf_k": 0})


def test_weights_must_have_one_positive():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={
            "method": "weighted",
            "weights": {"dense": 0, "bm25": 0, "summary": 0, "graph": 0},
        })


def test_weights_partial_inputs_keep_other_channel_defaults():
    req = RetrievalRequest(query="q", fusion={
        "method": "weighted",
        "weights": {"dense": 0, "bm25": 0},
    })
    # summary/graph defaults keep the weight set valid.
    assert req.fusion.weights.summary == 0.5
    assert req.fusion.weights.graph == 0.5


def test_routing_and_context_defaults():
    req = RetrievalRequest(query="q")
    assert req.routing.enabled is False
    assert req.routing.use_llm is False
    assert req.context.max_tokens is None


def test_context_max_tokens_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", context={"max_tokens": 0})


def test_weights_non_negative():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={
            "method": "weighted",
            "weights": {"dense": -1, "bm25": 1},
        })


def test_rewrite_methods_validated():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={"enabled": True, "methods": ["nope"]})


def test_hyde_alpha_and_n_variants_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={
            "enabled": True, "hyde_alpha": 1.5,
        })
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={
            "enabled": True, "n_variants": 6,
        })


def test_mmr_lambda_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", mmr={"enabled": True, "lambda_mult": -0.1})


def test_rerank_pool_bounds_and_relation():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", top_k=10,
                         rerank={"enabled": True, "candidate_pool": 65})
    with pytest.raises(ValidationError, match="candidate_pool"):
        RetrievalRequest(query="q", top_k=30,
                         rerank={"enabled": True, "candidate_pool": 25})


def test_pool_relation_skipped_when_rerank_disabled():
    # pool < top_k is fine when the rerank stage never runs.
    req = RetrievalRequest(
        query="q", top_k=100,
        rerank={"enabled": False, "candidate_pool": 25},
    )
    assert req.rerank.enabled is False


def test_to_result_maps_dataclasses():
    from vector_service.retrieval.base import (
        ChannelHit, ChannelRun, RecallSpec, RetrievalPlan, RetrievalResult,
        RetrievedChunk, StageTrace,
    )

    result = RetrievalResult(
        query="q",
        chunks=[RetrievedChunk(
            chunk_id="c1", fields={"text": "a"}, fusion_score=0.03,
            matched_channels=["dense", "bm25"], rerank_score=0.9,
        )],
        plan=RetrievalPlan(
            original_query="q",
            dense_specs=[RecallSpec("q", hypothetical="h-doc")],
            lexical_queries=["q"],
        ),
        channel_runs=[ChannelRun(
            channel="dense", query="q",
            hits=[ChannelHit("c1", 0.9, 1, {"text": "a"})],
        )],
        traces=[StageTrace("fuse", 4, {"method": "rrf"})],
    )
    resp = to_result(result)
    assert isinstance(resp, RetrievalResponse)
    dumped = resp.model_dump()
    assert dumped["chunks"][0]["chunk_id"] == "c1"
    assert dumped["chunks"][0]["rerank_score"] == 0.9
    assert dumped["plan"]["dense_specs"][0] == {
        "query": "q", "hypothetical": "h-doc",
    }
    assert dumped["channel_runs"][0]["hits"][0]["rank"] == 1
    assert dumped["traces"][0]["detail"] == {"method": "rrf"}
