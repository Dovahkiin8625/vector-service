"""HTTP surface for /v1/evaluation: sets, questions, versions, runs, gates."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.evaluation import router as evaluation_router
from vector_service.corpus import CorpusRepository


class _JobsCfg:
    max_attempts = 3
    wake_on_submit = False


class _Settings:
    jobs = _JobsCfg()


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
    app.include_router(evaluation_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.corpus = repo
    app.state.settings = _Settings()
    return TestClient(app)


# ---- sets ----


def test_create_list_get_delete_set(client):
    resp = client.post("/v1/evaluation/sets", json={
        "database": "default", "collection": "ingest",
        "name": "基线集", "description": "desc",
    })
    assert resp.status_code == 201
    set_id = resp.json()["set_id"]
    assert set_id.startswith("evs_")

    resp = client.get("/v1/evaluation/sets?database=default")
    assert resp.json()["total"] == 1

    resp = client.get(f"/v1/evaluation/sets/{set_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "基线集"

    assert client.delete(f"/v1/evaluation/sets/{set_id}").status_code == 200
    resp = client.get(f"/v1/evaluation/sets/{set_id}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "eval_set_not_found"


def test_list_sets_filters_by_collection(client):
    client.post("/v1/evaluation/sets", json={"name": "a",
                                              "collection": "ingest"})
    client.post("/v1/evaluation/sets", json={"name": "b",
                                              "collection": "other"})
    resp = client.get("/v1/evaluation/sets?collection=other")
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["name"] == "b"


# ---- questions ----


def test_add_and_list_questions(client):
    set_id = client.post("/v1/evaluation/sets", json={"name": "s"}).json()[
        "set_id"]
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/questions",
        json={"questions": [
            {"question": "营收？", "expected_chunk_ids": ["d1_0"]},
            {"question": "利润？", "expected_doc_ids": ["d1"]},
        ]},
    )
    assert resp.status_code == 201
    items = resp.json()["items"]
    assert [item["question"] for item in items] == ["营收？", "利润？"]
    assert items[0]["expected_chunk_ids"] == ["d1_0"]

    resp = client.get(f"/v1/evaluation/sets/{set_id}/questions")
    assert len(resp.json()["items"]) == 2


def test_add_questions_unknown_set_is_404(client):
    resp = client.post(
        "/v1/evaluation/sets/evs_none/questions",
        json={"questions": [
            {"question": "q", "expected_chunk_ids": ["x"]},
        ]},
    )
    assert resp.status_code == 404


def test_question_without_expectation_is_422(client):
    set_id = client.post("/v1/evaluation/sets", json={"name": "s"}).json()[
        "set_id"]
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/questions",
        json={"questions": [{"question": "q"}]},
    )
    assert resp.status_code == 422


# ---- versions ----


def _set_with_question(client):
    set_id = client.post("/v1/evaluation/sets", json={"name": "s"}).json()[
        "set_id"]
    client.post(
        f"/v1/evaluation/sets/{set_id}/questions",
        json={"questions": [
            {"question": "q", "expected_chunk_ids": ["d1_0"]},
        ]},
    )
    return set_id


def test_freeze_version_and_list(client):
    set_id = _set_with_question(client)
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/versions", json={"tag": "v1"}
    )
    assert resp.status_code == 201
    version_id = resp.json()["version_id"]
    assert version_id.startswith("evv_")
    assert resp.json()["question_count"] == 1

    resp = client.get(f"/v1/evaluation/sets/{set_id}/versions")
    assert [item["tag"] for item in resp.json()["items"]] == ["v1"]


def test_duplicate_version_tag_is_409(client):
    set_id = _set_with_question(client)
    client.post(
        f"/v1/evaluation/sets/{set_id}/versions", json={"tag": "v1"}
    )
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/versions", json={"tag": "v1"}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "version_tag_exists"


def test_freeze_empty_set_is_422(client):
    set_id = client.post("/v1/evaluation/sets", json={"name": "s"}).json()[
        "set_id"]
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/versions", json={"tag": "v1"}
    )
    assert resp.status_code == 422


# ---- runs ----


def test_submit_run_returns_202(client):
    set_id = _set_with_question(client)
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/runs",
        json={"template": {"rerank": {"enabled": False}}},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["run_id"].startswith("evr_")
    rows = client.get(f"/v1/evaluation/sets/{set_id}/runs").json()["items"]
    assert len(rows) == 1


def test_submit_run_unknown_set_is_404(client):
    resp = client.post("/v1/evaluation/sets/evs_none/runs", json={})
    assert resp.status_code == 404


def test_submit_run_empty_set_is_422(client):
    set_id = client.post("/v1/evaluation/sets", json={"name": "s"}).json()[
        "set_id"]
    resp = client.post(f"/v1/evaluation/sets/{set_id}/runs", json={})
    assert resp.status_code == 422


def test_submit_run_unknown_version_is_404(client):
    set_id = _set_with_question(client)
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/runs",
        json={"version_id": "evv_missing"},
    )
    assert resp.status_code == 404


def test_submit_run_include_answer_without_llm_is_503(client):
    set_id = _set_with_question(client)
    resp = client.post(
        f"/v1/evaluation/sets/{set_id}/runs",
        json={"include_answer": True,
               "template": {"rerank": {"enabled": False}}},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "llm_unavailable"


def test_get_unknown_run_is_404(client):
    resp = client.get("/v1/evaluation/runs/evr_missing")
    assert resp.status_code == 404


# ---- gates ----


def _version_id(client, set_id, tag="v1"):
    return client.post(
        f"/v1/evaluation/sets/{set_id}/versions", json={"tag": tag}
    ).json()["version_id"]


def test_create_gate_returns_201(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    resp = client.post("/v1/evaluation/gates", json={
        "database": "default", "collection": "ingest",
        "set_id": set_id, "version_id": version_id,
        "min_recall": 0.8,
    })
    assert resp.status_code == 201
    gate = resp.json()["items"][0]
    assert gate["gate_id"].startswith("gat_")
    assert gate["min_recall"] == 0.8


def test_create_gate_bad_set_or_version_is_404(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    resp = client.post("/v1/evaluation/gates", json={
        "set_id": "evs_missing", "version_id": version_id,
        "min_recall": 0.8,
    })
    assert resp.status_code == 404

    resp = client.post("/v1/evaluation/gates", json={
        "set_id": set_id, "version_id": "evv_missing",
        "min_recall": 0.8,
    })
    assert resp.status_code == 404


def test_create_duplicate_gate_is_409(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    body = {
        "database": "default", "collection": "ingest",
        "set_id": set_id, "version_id": version_id,
        "min_recall": 0.8,
    }
    client.post("/v1/evaluation/gates", json=body)
    resp = client.post("/v1/evaluation/gates", json=body)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "gate_exists"


def test_gate_without_criteria_is_422(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    resp = client.post("/v1/evaluation/gates", json={
        "set_id": set_id, "version_id": version_id,
    })
    assert resp.status_code == 422


def test_gate_drop_without_baseline_is_422(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    resp = client.post("/v1/evaluation/gates", json={
        "set_id": set_id, "version_id": version_id,
        "max_recall_drop": 0.1,
    })
    assert resp.status_code == 422


def test_list_get_delete_gate(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    gate_id = client.post("/v1/evaluation/gates", json={
        "set_id": set_id, "version_id": version_id,
        "min_recall": 0.8,
    }).json()["items"][0]["gate_id"]

    assert client.get("/v1/evaluation/gates").json()["total"] == 1
    assert client.get(f"/v1/evaluation/gates/{gate_id}").status_code == 200
    assert client.delete(f"/v1/evaluation/gates/{gate_id}").status_code == 200
    assert client.get(f"/v1/evaluation/gates/{gate_id}").status_code == 404


def test_submit_gate_check_without_canary_is_409(client):
    set_id = _set_with_question(client)
    version_id = _version_id(client, set_id)
    gate_id = client.post("/v1/evaluation/gates", json={
        "set_id": set_id, "version_id": version_id,
        "min_recall": 0.8,
    }).json()["items"][0]["gate_id"]

    resp = client.post(
        f"/v1/evaluation/gates/{gate_id}/checks"
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "gate_no_canary"


def test_list_gate_checks_unknown_gate_is_404(client):
    resp = client.get("/v1/evaluation/gates/gat_none/checks")
    assert resp.status_code == 404
