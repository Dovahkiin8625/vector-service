"""Tests for the asynchronous jobs API: submit / get / list.

No real parse/embed work happens on submit — these pin that
``POST /v1/jobs/ingest``:

- runs the full pre-flight validation (profile/MIME/size/params) and
  returns 202 with a queued job even while the embedder is unloaded;
- spools the raw upload and persists doc_id/params/attempts;
- and that GET endpoints report the persisted state.

A real :class:`CorpusRepository` runs against a temp SQLite file; the
worker is not started (queued jobs simply wait).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.jobs import router as jobs_router
from vector_service.corpus import CorpusRepository


class _ParserSettings:
    def __init__(self, max_mb=16):
        self.max_file_size_mb = max_mb


class _JobSettings:
    def __init__(self, spool_dir):
        self.spool_dir = spool_dir
        self.max_attempts = 2
        self.wake_on_submit = True


class _FakeSettings:
    def __init__(self, spool_dir, max_mb=16):
        self.parser = _ParserSettings(max_mb)
        self.jobs = _JobSettings(spool_dir)
        self.llm = None


def _http_error_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(
            status_code=exc.status_code, content={"error": detail["error"]}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "error", "message": str(detail)}},
    )


@pytest.fixture
def client(tmp_path):
    a = FastAPI()
    a.include_router(jobs_router)
    a.add_exception_handler(HTTPException, _http_error_handler)

    spool_root = tmp_path / "jobs"
    a.state.settings = _FakeSettings(spool_root)
    a.state.corpus = CorpusRepository(tmp_path / "corpus.db")
    a.state.corpus.initialize()
    # Embedder deliberately absent: submission must not require it.
    a.state.embedder = None
    a.state.spool_root = spool_root
    return TestClient(a)


SAMPLE = (
    "The quarterly report covers revenue and margins. "
    "Each section summarises a business unit.\n\n"
) * 4


def _form(**overrides) -> dict:
    data = {
        "database": "default",
        "collection": "ingest",
        "embed_model": "bge-m3",
        "chunk_size": "500",
        "chunk_overlap": "75",
        "metadata": "{}",
    }
    data.update(overrides)
    return data


def _submit(
    client, *, content=None, filename="doc.txt", content_type="text/plain", form=None
):
    data = SAMPLE.encode() if content is None else content
    files = {"file": (filename, data, content_type)}
    return client.post("/v1/jobs/ingest", data=form or _form(), files=files)


# ---- submit ------------------------------------------------------------


def test_submit_returns_202_and_spools_upload(client):
    r = _submit(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    assert isinstance(job_id, str) and job_id

    # Raw bytes landed on disk under <spool_dir>/<job_id>/upload.
    spool = client.app.state.spool_root / job_id / "upload"
    assert spool.read_bytes() == SAMPLE.encode()


def test_submit_persists_job_row_with_doc_id_and_params(client):
    r = _submit(client, form=_form(metadata='{"title": "年报"}'))
    job_id = r.json()["job_id"]
    row = client.app.state.corpus.get_job(job_id)
    assert row["status"] == "queued"
    assert row["doc_id"] and isinstance(row["doc_id"], str)
    assert row["spool_path"].endswith(f"{job_id}\\upload") or row[
        "spool_path"
    ].endswith(f"{job_id}/upload")
    assert row["attempts"] == 0
    assert row["max_attempts"] == 2

    params = json.loads(row["params_json"])
    assert params["database"] == "default"
    assert params["chunk_size"] == 500
    assert params["chunk_overlap"] == 75
    assert params["metadata"] == {"title": "年报"}
    assert params["filename"] == "doc.txt"
    assert params["mime"] == "text/plain"


def test_submit_does_not_require_embedder_loaded(client):
    # Embedder is None in this fixture yet submission succeeds — the
    # embedder-state check is deferred to worker execution.
    r = _submit(client)
    assert r.status_code == 202


def test_submit_bad_profile_is_400_and_leaves_no_spool(client):
    r = _submit(client, form=_form(profile="turbo"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_profile"
    # Validation runs before the spool directory is created.
    assert not client.app.state.spool_root.exists()


def test_submit_unsupported_mime_is_415(client):
    r = _submit(client, filename="virus.exe", content_type="application/x-msdownload")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_mime"


def test_submit_empty_file_is_400(client):
    r = _submit(client, content=b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_file"


def test_submit_oversized_file_is_413(client, tmp_path):
    # Swap in a 1 MB cap and send 2 MB.
    client.app.state.settings = _FakeSettings(client.app.state.spool_root, max_mb=1)
    r = _submit(client, content=b"x" * (2 * 1024 * 1024))
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"


def test_submit_bad_metadata_json_is_400(client):
    r = _submit(client, form=_form(metadata="{not json"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_metadata"


# ---- get / list --------------------------------------------------------


def test_get_job_reports_queued_status_shape(client):
    job_id = _submit(client).json()["job_id"]
    r = client.get(f"/v1/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["stage"] is None
    assert body["doc_id"]
    assert body["attempts"] == 0
    assert body["max_attempts"] == 2
    assert body["cancel_requested"] is False
    assert body["progress"] == {"current": None, "total": None}
    assert body["chunk_count"] == 0
    assert body["error"] is None
    assert body["finished_ts"] is None


def test_get_unknown_job_is_404(client):
    r = client.get("/v1/jobs/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_list_jobs_newest_first_and_status_filter(client):
    id1 = _submit(client).json()["job_id"]
    id2 = _submit(client).json()["job_id"]

    r = client.get("/v1/jobs")
    assert r.status_code == 200
    assert r.json()["total"] == 2
    assert [j["job_id"] for j in r.json()["items"]] == [id2, id1]

    r = client.get("/v1/jobs", params={"status": "queued"})
    assert r.json()["total"] == 2

    r = client.get("/v1/jobs", params={"status": "done"})
    assert r.json()["total"] == 0
    assert r.json()["items"] == []


# ---- cancel ------------------------------------------------------------


def test_cancel_flags_a_queued_job(client):
    job_id = _submit(client).json()["job_id"]
    r = client.post(f"/v1/jobs/{job_id}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == job_id
    assert body["cancel_requested"] is True
    # The queued row itself is untouched; the worker observes the flag.
    assert client.app.state.corpus.get_job(job_id)["status"] == "queued"


def test_cancel_unknown_job_is_404(client):
    r = client.post("/v1/jobs/nope/cancel")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_cancel_terminal_job_is_409(client):
    job_id = _submit(client).json()["job_id"]
    client.app.state.corpus.mark_job(job_id, "done")

    r = client.post(f"/v1/jobs/{job_id}/cancel")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "job_already_terminal"
    assert client.app.state.corpus.get_job(job_id)["cancel_requested"] == 0
