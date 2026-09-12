"""Unit tests for ``schemas.rerank``."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.schemas.rerank import (
    RerankModelsResponse,
    RerankRequest,
    RerankResponse,
    RerankResultItem,
    RerankerInfo,
)


def test_rerank_request_minimal():
    r = RerankRequest(query="q", documents=["a", "b"])
    assert r.query == "q"
    assert r.documents == ["a", "b"]
    assert r.top_n is None
    assert r.model is None


def test_rerank_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RerankRequest(query="", documents=["a"])


def test_rerank_request_rejects_empty_documents():
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=[])


def test_rerank_request_top_n_must_be_positive():
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=["a"], top_n=0)


def test_rerank_request_rejects_extras():
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=["a"], evil=True)


def test_rerank_response_roundtrip():
    r = RerankResponse(
        model="bge-reranker-v2-m3",
        results=[RerankResultItem(index=0, score=0.5)],
        request_id="req_abc",
    )
    assert r.model_dump() == {
        "model": "bge-reranker-v2-m3",
        "results": [{"index": 0, "score": 0.5}],
        "request_id": "req_abc",
    }


def test_reranker_info_and_models_response():
    m = RerankModelsResponse(data=[RerankerInfo(name="bge-reranker-v2-m3")])
    assert m.data[0].name == "bge-reranker-v2-m3"
