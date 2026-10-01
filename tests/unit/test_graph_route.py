"""HTTP surface for the GraphRAG build / status / delete endpoints."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.graph import router as graph_router
from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.graph.names import graph_collection_names


class FakeStore:
    def __init__(self):
        self.dropped = []

    def drop_collection(self, database, name):
        self.dropped.append((database, name))


class _JobsCfg:
    max_attempts = 3
    wake_on_submit = False


def _http_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": "error", "message": str(detail)}})


@pytest.fixture
def client(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    app = FastAPI()
    app.include_router(graph_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.corpus = repo
    app.state.store = FakeStore()
    app.state.settings = SimpleNamespace(jobs=_JobsCfg())
    return TestClient(app)


def _seed_document(client):
    repo = client.app.state.corpus
    doc = DocumentRecord(
        doc_id="d1", database="default", collection="ingest",
        filename="年报.pdf", mime="application/pdf", content_hash="a" * 64,
    )
    chunks = [ChunkRecord(
        chunk_id="d1_0", doc_id="d1", database="default",
        collection="ingest", chunk_index=0, text="leaf text",
    )]
    repo.store_document(doc, chunks)


_PATH = "/v1/databases/default/collections/ingest"


# ---- build -------------------------------------------------------------


def test_graph_build_rejects_non_corpus_collection(client):
    resp = client.post(f"{_PATH}/graph/build", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_graph_build_submits_job_for_corpus_collection(client):
    _seed_document(client)
    entity_coll, community_coll = graph_collection_names("ingest")
    resp = client.post(f"{_PATH}/graph/build", json={"batch_size": 4})
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["entity_collection"] == entity_coll
    assert data["community_collection"] == community_coll

    job = client.app.state.corpus.get_job(data["job_id"])
    assert job["status"] == "queued"
    assert job["job_type"] == "graph_build"
    params = json.loads(job["params_json"])
    assert params["entity_coll"] == entity_coll
    assert params["batch_size"] == 4


def test_graph_build_rejects_invalid_options(client):
    _seed_document(client)
    resp = client.post(f"{_PATH}/graph/build", json={
        "community_iterations": 0,
    })
    assert resp.status_code == 422


# ---- status ------------------------------------------------------------


def test_graph_status_reports_zero_counts(client):
    entity_coll, community_coll = graph_collection_names("ingest")
    resp = client.get(f"{_PATH}/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["database"] == "default"
    assert data["collection"] == "ingest"
    assert data["entity_collection"] == entity_coll
    assert data["community_collection"] == community_coll
    assert data["entities_count"] == 0
    assert data["edges_count"] == 0
    assert data["claims_count"] == 0
    assert data["communities_count"] == 0
    assert data["communities"] == []


# ---- delete ------------------------------------------------------------


def test_graph_delete_clears_rows_and_drops_collections(client):
    _seed_document(client)
    entity_coll, community_coll = graph_collection_names("ingest")

    # Seed one graph row directly to prove the delete reaches it.
    repo = client.app.state.corpus
    with repo._txn() as conn:
        conn.execute(
            "INSERT INTO entities (entity_id, database, collection, name, "
            "entity_type, description, created_ts, updated_ts) "
            "VALUES ('ge_1', 'default', 'ingest', 'Alice', 'person', '', 1, 1)"
        )

    resp = client.delete(f"{_PATH}/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["entity_collection"] == entity_coll
    assert data["community_collection"] == community_coll

    store = client.app.state.store
    assert ("default", entity_coll) in store.dropped
    assert ("default", community_coll) in store.dropped
    assert client.get(f"{_PATH}/graph").json()["entities_count"] == 0


def test_graph_delete_succeeds_when_collections_absent(client):
    # FakeStore drop succeeds; real route also tolerates CollectionNotFound.
    resp = client.delete(f"{_PATH}/graph")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
