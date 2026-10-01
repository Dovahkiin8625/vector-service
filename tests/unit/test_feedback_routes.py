"""HTTP surface for /v1/feedback: submit + list."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.feedback import router as feedback_router
from vector_service.corpus import CorpusRepository


def _http_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": "error",
                                           "message": str(detail)}})


@pytest.fixture
def client(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    app = FastAPI()
    app.include_router(feedback_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.corpus = repo
    return TestClient(app)


def _body(**overrides):
    body = {"query": "季度营收", "kind": "up"}
    body.update(overrides)
    return body


def test_submit_answer_level_feedback_returns_id(client):
    resp = client.post("/v1/feedback", json=_body())
    assert resp.status_code == 200
    feedback_id = resp.json()["feedback_id"]
    assert feedback_id.startswith("fb_")


def test_submit_chunk_level_feedback_with_pipeline_snapshot(client):
    resp = client.post("/v1/feedback", json=_body(
        kind="click", chunk_id="d1_0",
        pipeline={"channels": ["dense"], "rerank": {"enabled": True}},
    ))
    assert resp.status_code == 200

    rows, total = client.app.state.corpus.list_feedback()
    assert total == 1
    row = rows[0]
    assert row["chunk_id"] == "d1_0"
    assert row["kind"] == "click"


def test_invalid_kind_is_422(client):
    resp = client.post("/v1/feedback", json=_body(kind="star"))
    assert resp.status_code == 422


def test_kind_is_normalized_case_and_spaces(client):
    resp = client.post("/v1/feedback", json=_body(kind=" DOWN "))
    assert resp.status_code == 200
    rows, _ = client.app.state.corpus.list_feedback()
    assert rows[0]["kind"] == "down"


def test_list_filters_by_scope_and_kind(client):
    client.post("/v1/feedback", json=_body(
        kind="up", database="default", collection="ingest"))
    client.post("/v1/feedback", json=_body(
        kind="down", database="default", collection="ingest"))
    client.post("/v1/feedback", json=_body(
        kind="up", database="otherdb", collection="ingest"))

    resp = client.get(
        "/v1/feedback?database=default&collection=ingest&kind=up"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["kind"] == "up"
    assert data["limit"] == 50 and data["offset"] == 0


def test_list_unknown_kind_is_ignored_not_422(client):
    client.post("/v1/feedback", json=_body(kind="up"))
    resp = client.get("/v1/feedback?kind=nonsense")
    assert resp.status_code == 200
    # An unknown kind falls back to no kind filter.
    assert resp.json()["total"] == 1


def test_list_limit_is_clamped(client):
    for _ in range(3):
        client.post("/v1/feedback", json=_body())
    resp = client.get("/v1/feedback?limit=99999")
    assert resp.status_code == 200
    assert resp.json()["limit"] == 200
