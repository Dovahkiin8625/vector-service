"""API tests for ``POST /v1/chunk`` strategy dispatch.

Focuses on what the strategy unit tests can't cover: dependency
resolution at the route boundary — semantic needs a loaded
embedder, llm / add_context need the configured external LLM —
plus request validation. No real embed stack: a fake embedder with
deterministic vectors stands in.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vector_service.api import chunk as chunk_mod
from vector_service.api.chunk import router as chunk_router


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "Cats" in t else [0.0, 1.0] for t in texts]


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(chunk_router)
    a.state.embedder = FakeEmbedder()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_ambient_llm(monkeypatch):
    """Pin LLM as unconfigured regardless of the ambient .env file;
    the success test overrides this itself."""
    monkeypatch.setattr(chunk_mod, "is_llm_configured", lambda _s: False)


def _body(**overrides) -> dict:
    data = {"markdown": "# Title\n\nHello world. " * 10}
    data.update(overrides)
    return data


def test_chunk_default_recursive(client):
    r = client.post("/v1/chunk", json=_body())
    assert r.status_code == 200
    payload = r.json()
    assert payload["chunks"]
    # Chunk items expose the nullable context field.
    assert payload["chunks"][0]["context"] is None


def test_chunk_semantic_without_embedder_is_503(app):
    app.state.embedder = None
    r = TestClient(app).post(
        "/v1/chunk", json=_body(strategy="semantic")
    )
    assert r.status_code == 503
    assert r.json()["detail"]["error"]["code"] == "embedder_unavailable"


def test_chunk_semantic_with_embedder(client):
    # 20 sentences per side keep both groups above the 50-token floor.
    md = "Cats meow. " * 20 + "Dogs bark. " * 20
    r = client.post(
        "/v1/chunk",
        json=_body(markdown=md, strategy="semantic",
                   options={"breakpoint_percentile": 95}),
    )
    assert r.status_code == 200
    assert len(r.json()["chunks"]) == 2


def test_chunk_llm_without_config_is_503(client):
    r = client.post("/v1/chunk", json=_body(strategy="llm"))
    assert r.status_code == 503
    assert r.json()["detail"]["error"]["code"] == "llm_unavailable"


def test_chunk_add_context_without_config_is_503(client):
    r = client.post("/v1/chunk", json=_body(add_context=True))
    assert r.status_code == 503
    assert r.json()["detail"]["error"]["code"] == "llm_unavailable"


def test_chunk_unknown_strategy_is_422(client):
    r = client.post("/v1/chunk", json=_body(strategy="mystery"))
    assert r.status_code == 422


def test_chunk_overlap_ge_size_is_400(client):
    r = client.post(
        "/v1/chunk",
        json=_body(chunk_size=100, chunk_overlap=100),
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "invalid_chunk_config"
