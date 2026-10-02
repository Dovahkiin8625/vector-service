"""HTTP surface for /v1/retrieval (JSON + NDJSON) and capabilities."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.retrieval import router as retrieval_router
from vector_service.stores.base import CollectionInfo, Hit


class FakeStore:
    def list_databases(self):
        return ["default"]

    def list_collections(self, database):
        return ["ingest"]

    def collection_info(self, db, coll):
        fields = [{"name": "id"}, {"name": "text"}, {"name": "sparse"}]
        return CollectionInfo(database=db, name=coll, dim=4, metric="cosine",
                              count=0, fields=fields)

    @staticmethod
    def _project(fields, output_fields):
        if output_fields is None:
            return {}
        return {k: v for k, v in fields.items() if k in output_fields}

    def search(self, *a, **kw):
        return [Hit(id="c1", score=0.9,
                    fields=self._project({"text": "dense"},
                                         kw.get("output_fields")))]

    def search_sparse(self, *a, **kw):
        return [Hit(id="c2", score=5.0,
                    fields=self._project({"text": "lexical"},
                                         kw.get("output_fields")))]


class FakeEmbedder:
    def embed_query(self, q):
        return [0.1, 0.2, 0.3, 0.4]


class FakeBM25:
    def encode_query(self, repo, database, logical_collection,
                     physical_collection, query):
        return {3: 1.5}


class FakeReranker:
    _impl = object()

    def rerank(self, q, docs, top_n=None):
        from vector_service.rerankers.base import ScoredHit

        return [ScoredHit(index=0, score=0.99)]


class FakeSettings:
    llm = None


def _http_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": "error", "message": str(detail)}})


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(retrieval_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.settings = FakeSettings()
    app.state.store = FakeStore()
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    app.state.bm25 = FakeBM25()
    return TestClient(app)


def _body(**overrides):
    body = {"query": "季度营收"}
    body.update(overrides)
    return body


def test_json_retrieval_returns_envelope(client):
    resp = client.post("/v1/retrieval", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "季度营收"
    assert len(data["chunks"]) >= 1
    assert {"channel_runs", "plan", "traces"} <= set(data)
    # Hits carry scalar content (text must be in the explicit projection).
    assert data["chunks"][0]["fields"].get("text")


def test_stream_emits_stage_then_result(client):
    resp = client.post("/v1/retrieval/stream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["rewrite", "recall", "fuse", "rerank"]
    assert events[-1]["type"] == "result"
    assert "chunks" in events[-1]


def test_stream_postflight_backend_error_is_503_event(client):
    from vector_service.core.errors import BackendError

    def boom(*a, **kw):
        raise BackendError("milvus unavailable")

    client.app.state.store.search = boom
    client.app.state.store.search_sparse = boom
    resp = client.post("/v1/retrieval/stream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    err = events[-1]
    assert err["type"] == "error"
    assert err["status"] == 503
    assert err["error"]["code"] == "store_unavailable"


def test_preflight_error_is_plain_json_not_ndjson(client):
    client.app.state.embedder = None
    resp = client.post("/v1/retrieval/stream", json=_body())
    assert resp.status_code == 503
    assert "json" in resp.headers["content-type"]
    assert resp.json()["error"]["code"] == "embedder_unavailable"


def test_rerank_off_top_k_100_does_not_422(client):
    # candidate_pool relation only applies when rerank is enabled.
    resp = client.post("/v1/retrieval", json=_body(
        rerank={"enabled": False}, top_k=100))
    assert resp.status_code == 200


def test_json_embedder_error_maps_via_global_handler():
    from vector_service.core.errors import EmbedderError
    from vector_service.main import create_app

    class RaisingEmbedder:
        def embed_query(self, q):
            raise EmbedderError("embed model exploded")

        def embed_documents(self, texts):
            raise EmbedderError("embed model exploded")

    app = create_app()
    app.state.settings = FakeSettings()
    app.state.store = FakeStore()
    app.state.embedder = RaisingEmbedder()
    app.state.reranker = FakeReranker()
    app.state.bm25 = FakeBM25()
    client = TestClient(app)  # no context manager -> lifespan does not run

    resp = client.post("/v1/retrieval", json=_body(rerank={"enabled": False}))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "embedder_unavailable"


def test_validation_error_422(client):
    resp = client.post("/v1/retrieval", json=_body(
        channels={"dense": False, "bm25": False}))
    assert resp.status_code == 422


def test_capabilities_reports_llm_flag(client):
    resp = client.get("/v1/retrieval/capabilities?database=default")
    assert resp.status_code == 200
    assert resp.json() == {"llm_configured": False}
