"""Tests for ``POST /v1/parse/stream`` — the NDJSON progress stream.

Mirrors the assertions of ``test_ingest_stream.py`` for the parse
contract:

- success: ``stage(parse)`` → per-page ``progress`` events → terminal
  ``result`` carrying the ParseResponse fields;
- an in-flight parser failure rides inside the 200 NDJSON stream as an
  ``error`` event (canonical envelope + HTTP-equivalent status);
- pre-flight failures (MIME / size / empty upload) happen BEFORE the
  stream opens and come back as ordinary JSON envelopes;
- the classic ``POST /v1/parse`` route keeps returning JSON.

No real Docling: ``get_docling_parser`` is monkeypatched to a fake that
ticks per-page progress; text formats go through the dependency-free
MarkdownParser.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api import parse as parse_mod
from vector_service.api.parse import router as parse_router
from vector_service.parsers.base import ParsedDocument


class _ParserSettings:
    max_file_size_mb = 16


class _FakeSettings:
    parser = _ParserSettings()


class FakeDoclingParser:
    """Emits three page ticks, then a 3-page markdown result."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail

    async def parse_bytes(self, data, mime, on_progress=None, **_kwargs):
        if on_progress is not None:
            # Small sleeps make the worker-thread callback path
            # (call_soon_threadsafe) realistic.
            for done in (1, 2, 3):
                on_progress(done, 3)
        if self.fail:
            raise RuntimeError("docling conversion failed for in-memory")
        return ParsedDocument(
            markdown="# title\n\nbody",
            metadata={"mime_type": mime, "page_count": 3},
        )


class _BoomDoclingParser:
    async def parse_bytes(self, data, mime, on_progress=None, **_kwargs):
        raise RuntimeError("docling exploded")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(
        parse_mod, "get_docling_parser", lambda: FakeDoclingParser()
    )
    a = FastAPI()
    a.include_router(parse_router)
    a.state.settings = _FakeSettings()

    @a.exception_handler(HTTPException)
    async def _h(_request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": detail["error"]},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "error", "message": str(detail)}},
        )

    return a


@pytest.fixture
def client(app):
    return TestClient(app)


def _events(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def _pdf_upload(client, **kwargs):
    files = {"file": ("doc.pdf", kwargs.pop("content", b"%PDF-1.4"),
                      "application/pdf")}
    return client.post("/v1/parse/stream", files=files)


def test_stream_emits_stage_progress_then_result(client):
    r = _pdf_upload(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers.get("cache-control") == "no-cache"

    events = _events(r)
    assert events[0] == {"type": "stage", "stage": "parse"}
    progress = [e for e in events if e["type"] == "progress"]
    assert [(e["page"], e["total"]) for e in progress] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert all(e["stage"] == "parse" for e in progress)
    terminal = events[-1]
    assert terminal["type"] == "result"
    assert terminal["markdown"] == "# title\n\nbody"
    assert terminal["metadata"]["page_count"] == 3
    assert [e["type"] for e in events].count("result") == 1
    assert set(e["type"] for e in events) <= {"stage", "progress", "result"}


def test_stream_parser_runtime_error_is_error_event(client, monkeypatch):
    monkeypatch.setattr(
        parse_mod, "get_docling_parser", lambda: _BoomDoclingParser()
    )
    r = _pdf_upload(client)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _events(r)
    assert events[0] == {"type": "stage", "stage": "parse"}
    terminal = events[-1]
    assert terminal["type"] == "error"
    assert terminal["status"] == 500
    assert terminal["error"]["code"] == "parser_failed"
    assert "docling exploded" in terminal["error"]["message"]


def test_preflight_unsupported_mime_is_plain_json_415(client):
    r = client.post(
        "/v1/parse/stream",
        files={"file": ("x.bin", b"\x00\x01", "application/octet-stream")},
    )
    assert r.status_code == 415
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "unsupported_mime"


def test_preflight_empty_file_is_plain_json_400(client):
    r = _pdf_upload(client, content=b"")
    assert r.status_code == 400
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "empty_file"


def test_preflight_oversize_file_is_plain_json_413(client):
    r = _pdf_upload(client, content=b"x" * (16 * 1024 * 1024 + 1))
    assert r.status_code == 413
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "file_too_large"


def test_classic_parse_route_still_returns_json(client):
    r = client.post(
        "/v1/parse",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    body = r.json()
    assert body["markdown"] == "# title\n\nbody"
    assert body["metadata"]["page_count"] == 3
    assert isinstance(body, dict) and "type" not in body  # no event envelope


def test_text_upload_emits_no_progress_events(client):
    r = client.post(
        "/v1/parse/stream",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 200
    events = _events(r)
    assert [e["type"] for e in events] == ["stage", "result"]
    assert events[-1]["markdown"] == "hello world"
