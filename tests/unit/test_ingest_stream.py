"""Tests for ``POST /v1/ingest/stream`` — the NDJSON stage-event route.

The stream route shares its pipeline core with the classic
``POST /v1/ingest`` JSON route; these tests pin the differences:

- the success stream emits ``parse -> chunk -> embed -> upsert`` stage
  events followed by a terminal ``result`` event carrying the
  IngestResponse fields;
- an in-flight failure (embedder raising) still produces HTTP 200 over
  ``application/x-ndjson`` and ends with an ``error`` event carrying the
  canonical status/code;
- pre-flight failures (bad params / MIME / embedder state) happen BEFORE
  the stream opens, so they come back as ordinary JSON error envelopes;
- the classic JSON route still returns a plain IngestResponse.

No real Docling/Milvus/BGE stack: text/plain goes through the dependency
free MarkdownParser, embedder + store are in-memory fakes.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api import ingest as ingest_mod
from vector_service.api.ingest import router as ingest_router
from vector_service.core.errors import EmbedderError


class _ParserSettings:
    max_file_size_mb = 16


class _FakeSettings:
    parser = _ParserSettings()
    inference_timeout_seconds = 30.0


class FakeEmbedder:
    model_name = "fake-model"
    dim = 4

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _CollInfo:
    def __init__(self, dim):
        self.dim = dim


class FakeStore:
    """Minimal store surface exercised by the ingest pipeline."""

    def __init__(self):
        self.dbs = ["default"]
        self.colls: dict[str, dict[str, int]] = {}
        self.upserts: list[dict] = []

    def list_databases(self):
        return list(self.dbs)

    def create_database(self, name, **_opts):
        self.dbs.append(name)

    def list_collections(self, database):
        return list(self.colls.get(database, {}))

    def create_collection(self, database, name, primary_field,
                          vector_field, scalar_fields, indexes=None):
        self.colls.setdefault(database, {})[name] = vector_field.dim

    def collection_info(self, database, name):
        return _CollInfo(self.colls[database][name])

    def upsert(self, database, collection, primary_field, vector_field,
               ids, vectors, fields=None):
        self.upserts.append({
            "database": database,
            "collection": collection,
            "primary_field": primary_field,
            "vector_field": vector_field,
            "ids": ids,
            "vectors": vectors,
            "fields": fields,
        })


def _http_error_handler(_request: Request, exc: HTTPException):
    """Mirror the production envelope shape: {"error": {...}}."""
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "error", "message": str(detail)}},
    )


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(ingest_router)
    a.add_exception_handler(HTTPException, _http_error_handler)
    a.state.settings = _FakeSettings()
    a.state.embedder = FakeEmbedder()
    a.state.store = FakeStore()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


SAMPLE = ("The quarterly report covers revenue, margins, and headcount. "
          "Each section summarises a different business unit.\n\n"
          "Operations expanded into two new regions this period. ") * 4


def _form(**overrides) -> dict:
    data = {
        "database": "default",
        "collection": "ingest",
        "embed_model": "fake-model",
        "chunk_size": "800",
        "chunk_overlap": "80",
        "metadata": "{}",
    }
    data.update(overrides)
    return data


def _post_stream(client, *, data=None, filename="doc.txt",
                 content=b"", content_type="text/plain"):
    files = {"file": (filename, content, content_type)}
    return client.post("/v1/ingest/stream", data=data or _form(), files=files)


def _events(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


# -----------------------------------------------------------------------
# Success stream
# -----------------------------------------------------------------------


def test_stream_emits_all_stages_then_result(client):
    r = _post_stream(client, content=SAMPLE.encode("utf-8"))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers.get("cache-control") == "no-cache"

    events = _events(r)
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["parse", "chunk", "embed", "upsert"]

    terminal = events[-1]
    assert terminal["type"] == "result"
    assert terminal["chunk_count"] >= 1
    assert terminal["tokens_used"] >= 1
    assert terminal["page_count"] is None
    assert isinstance(terminal["doc_id"], str) and terminal["doc_id"]

    # Exactly one terminal event, no stray event types.
    assert [e["type"] for e in events].count("result") == 1
    assert all(e["type"] in ("stage", "result") for e in events)


def test_stream_success_actually_upserts_chunks(client):
    store = client.app.state.store
    r = _post_stream(client, data=_form(metadata='{"title": "Q3 report"}'),
                     content=SAMPLE.encode("utf-8"))
    events = _events(r)
    doc_id = events[-1]["doc_id"]
    chunk_count = events[-1]["chunk_count"]

    assert len(store.upserts) == 1
    call = store.upserts[0]
    assert call["database"] == "default"
    assert call["collection"] == "ingest"
    assert call["primary_field"] == "id"
    assert call["vector_field"] == "vector"
    assert len(call["ids"]) == chunk_count
    assert all(i.startswith(doc_id + "_") for i in call["ids"])
    assert all(len(v) == 4 for v in call["vectors"])

    first = call["fields"][0]
    assert first["doc_id"] == doc_id
    assert first["chunk_index"] == 0
    assert first["text"]
    assert first["token_count"] >= 1
    assert first["title"] == "Q3 report"
    # Parser passthrough + filename backfill land in scalar fields.
    assert first["filename"] == "doc.txt"


def test_stream_auto_creates_database_and_collection(client):
    store = client.app.state.store
    r = _post_stream(client, data=_form(database="kb"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 200
    assert "kb" in store.dbs
    assert "ingest" in store.colls["kb"]
    assert store.colls["kb"]["ingest"] == 4  # FakeEmbedder.dim


def test_stream_forwards_parse_page_progress(client, monkeypatch):
    """Docling page ticks ride the ingest stream as progress events.

    They stay strictly inside the parse stage (after the parse stage
    event, before chunk) so the dashboard can render ``n/total 页``
    during the slowest stage.
    """
    from vector_service.parsers.base import ParsedDocument

    class FakeDoclingParser:
        async def parse_bytes(self, data, mime, on_progress=None, **_kwargs):
            if on_progress is not None:
                for done in (1, 2, 3):
                    on_progress(done, 3)
            return ParsedDocument(
                markdown="# title\n\n" + SAMPLE,
                metadata={"mime_type": mime, "page_count": 3},
            )

    # The pipeline must go through the process-wide getter (the
    # monkeypatch is invisible to a directly-constructed DoclingParser).
    monkeypatch.setattr(
        ingest_mod, "get_docling_parser", lambda: FakeDoclingParser()
    )

    r = _post_stream(
        client, filename="doc.pdf",
        content=b"%PDF-1.4\nfake", content_type="application/pdf",
    )
    assert r.status_code == 200
    events = _events(r)

    progress = [e for e in events if e["type"] == "progress"]
    assert [(e["stage"], e["page"], e["total"]) for e in progress] == [
        ("parse", 1, 3),
        ("parse", 2, 3),
        ("parse", 3, 3),
    ]

    # Ordering: stage(parse) → all ticks → stage(chunk) → ... → result.
    idx_parse_stage = next(
        i for i, e in enumerate(events)
        if e["type"] == "stage" and e["stage"] == "parse"
    )
    idx_chunk_stage = next(
        i for i, e in enumerate(events)
        if e["type"] == "stage" and e["stage"] == "chunk"
    )
    assert idx_parse_stage < events.index(progress[0])
    assert events.index(progress[-1]) < idx_chunk_stage
    assert events[-1]["type"] == "result"
    assert events[-1]["page_count"] == 3


# -----------------------------------------------------------------------
# In-flight failure -> error event inside a 200 NDJSON stream
# -----------------------------------------------------------------------


def test_stream_embedder_failure_is_error_event(client):
    def _boom(_texts):
        raise EmbedderError("CUDA out of memory")

    client.app.state.embedder.embed_documents = _boom

    r = _post_stream(client, content=SAMPLE.encode("utf-8"))
    # The stream had already opened with 200; the failure rides as an event.
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")

    events = _events(r)
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["parse", "chunk", "embed"]  # never reached upsert

    terminal = events[-1]
    assert terminal["type"] == "error"
    assert terminal["status"] == 503
    assert terminal["error"]["code"] == "embedder_unavailable"
    assert "CUDA out of memory" in terminal["error"]["message"]

    # Nothing was written, so no rollback/upsert trace.
    assert client.app.state.store.upserts == []


# -----------------------------------------------------------------------
# Pre-flight failures -> ordinary JSON envelopes, no stream
# -----------------------------------------------------------------------


def test_preflight_no_embedder_is_plain_json_503(app):
    app.state.embedder = None
    client = TestClient(app)
    r = _post_stream(client, data=_form(embed_model="fake-model"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 503
    assert "application/json" in r.headers["content-type"]
    assert "x-ndjson" not in r.headers["content-type"]
    body = r.json()
    assert body["error"]["code"] == "embedder_unavailable"
    # Nothing in the body looks like an event envelope.
    assert "type" not in body


def test_preflight_bad_chunk_size_is_plain_json_400(client):
    r = _post_stream(client, data=_form(chunk_size="99999"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 400
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "invalid_chunk_size"


def test_preflight_bad_strategy_is_plain_json_400(client):
    r = _post_stream(client, data=_form(strategy="bogus"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 400
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "invalid_strategy"


def test_preflight_bad_chunk_options_is_plain_json_400(client):
    r = _post_stream(client, data=_form(chunk_options="not-json"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_chunk_options"


def test_preflight_llm_strategy_without_config_is_503(client):
    r = _post_stream(client, data=_form(strategy="llm"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "llm_unavailable"


def test_preflight_add_context_without_config_is_503(client):
    r = _post_stream(client, data=_form(add_context="true"),
                     content=SAMPLE.encode("utf-8"))
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "llm_unavailable"


def test_named_strategy_runs_end_to_end(client):
    r = _post_stream(client, data=_form(strategy="paragraph"),
                     content=SAMPLE.encode("utf-8"))
    events = _events(r)
    assert r.status_code == 200
    assert events[-1]["type"] == "result"
    assert events[-1]["chunk_count"] >= 1


def test_preflight_unsupported_mime_is_plain_json_415(client):
    r = _post_stream(
        client, data=_form(), filename="payload.bin",
        content=b"\x00\x01binary", content_type="application/octet-stream",
    )
    assert r.status_code == 415
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "unsupported_mime"


# -----------------------------------------------------------------------
# Empty-document early result + classic JSON route regression
# -----------------------------------------------------------------------


def test_stream_empty_document_ends_after_chunk_stage(client, monkeypatch):
    # Simulate a document that parses but yields zero chunks. The
    # pipeline builds chunkers through the module-level factory name.
    class _EmptyChunker:
        def chunk(self, markdown):  # noqa: ARG002
            return []

    monkeypatch.setattr(
        ingest_mod, "build_chunker",
        lambda *_a, **_k: _EmptyChunker(),
    )

    r = _post_stream(client, content=b"   \n\n  \n")
    assert r.status_code == 200
    events = _events(r)
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["parse", "chunk"]  # no embed / upsert work
    terminal = events[-1]
    assert terminal["type"] == "result"
    assert terminal["chunk_count"] == 0
    assert terminal["tokens_used"] == 0
    assert terminal["doc_id"]
    assert client.app.state.store.upserts == []


def test_classic_ingest_route_still_returns_plain_json(client):
    r = client.post(
        "/v1/ingest",
        data=_form(),
        files={"file": ("doc.txt", SAMPLE.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    body = r.json()
    assert body["chunk_count"] >= 1
    assert body["doc_id"]
    # The JSON route is not the event contract.
    assert "type" not in body
