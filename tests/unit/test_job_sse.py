"""Tests for SSE job-progress streaming (``GET /v1/jobs/{id}/events``).

Two layers:

- :class:`JobEventBus` unit behavior — subscribe/unsubscribe, fan-out,
  non-blocking drop for a full queue;
- end-to-end frames against the real router + :class:`IngestWorker`:
  initial snapshot, every stage transition, terminal close, cancel-route
  nudge, and the 404 envelope.

The end-to-end app runs through :class:`TestClient` (server thread +
portal). A small ``mark_job`` delay gives the SSE generator a scheduling
window per stage so the stage sequence is pinned deterministically,
without any production timing hooks.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.jobs import (
    cancel_job,
    stream_job_events,
)
from vector_service.api.jobs import (
    router as jobs_router,
)
from vector_service.corpus import CorpusRepository
from vector_service.corpus.blobs import BlobStore
from vector_service.corpus.bm25 import SparseBM25
from vector_service.jobs import IngestWorker, JobEventBus

SAMPLE = (
    "The quarterly report covers revenue and margins. "
    "Each sentence adds a little more retrievable text.\n\n"
) * 6


# ---- fakes / settings --------------------------------------------------


class _FakeAdapter:
    class _Client:
        def __init__(self, deleted):
            self.deleted = deleted

        def delete(self, collection, filter):
            self.deleted.append((collection, filter))

    def __init__(self, deleted):
        self._client = self._Client(deleted)


class _FakeStore:
    def __init__(self):
        self.deleted: list[tuple] = []
        self.upserts = 0
        self._adapter = _FakeAdapter(self.deleted)
        # Optional gate parked INSIDE upsert, after the upserting stage
        # nudge and before finish_job persists done.
        self.upsert_parked: threading.Event | None = None
        self.upsert_released: threading.Event | None = None

    def list_databases(self):
        return ["default"]

    def create_database(self, name):
        pass

    def list_collections(self, database):
        return []

    def collection_info(self, database, collection):
        return None

    def create_collection(self, **kwargs):
        pass

    def upsert(self, *args, **kwargs):
        self.upserts += 1
        if self.upsert_parked is not None and self.upsert_released is not None:
            self.upsert_parked.set()
            assert self.upsert_released.wait(10), "upsert never released"


class _FakeEmbedder:
    model_name = "bge-m3"
    dim = 4

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _JobsSettings:
    def __init__(self, spool_dir):
        self.spool_dir = str(spool_dir)
        self.max_attempts = 2
        # Never let the idle worker claim on its own; tests wake it.
        self.wake_on_submit = False
        self.poll_interval_seconds = 30.0
        self.retry_backoff_base_seconds = 0.0
        self.retry_backoff_max_seconds = 0.0
        self.heartbeat_seconds = 300.0
        self.sse_resync_seconds = 0.5


class _Settings:
    def __init__(self, spool_dir):
        self.parser = SimpleNamespace(max_file_size_mb=16)
        self.jobs = _JobsSettings(spool_dir)
        self.inference_timeout_seconds = 30.0
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


def _form() -> dict:
    return {
        "database": "default",
        "collection": "ingest",
        "embed_model": "bge-m3",
        "chunk_size": "500",
        "chunk_overlap": "75",
        "metadata": "{}",
    }


@pytest.fixture
def app(tmp_path):
    a = FastAPI()
    a.include_router(jobs_router)
    a.add_exception_handler(HTTPException, _http_error_handler)

    spool_root = tmp_path / "jobs"
    a.state.settings = _Settings(spool_root)
    a.state.corpus = CorpusRepository(tmp_path / "corpus.db")
    a.state.corpus.initialize()
    a.state.job_bus = JobEventBus()
    a.state.embedder = _FakeEmbedder()
    a.state.spool_root = spool_root
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


def _submit(client) -> str:
    files = {"file": ("doc.txt", SAMPLE.encode(), "text/plain")}
    r = client.post("/v1/jobs/ingest", data=_form(), files=files)
    assert r.status_code == 202
    return r.json()["job_id"]


# ---- SSE frame parsing -------------------------------------------------


def parse_frames(text: str) -> list[tuple[str, dict]]:
    """Decode complete SSE frames, skipping comments/keepalives."""
    frames: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        lines = block.splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        event = "message"
        data_lines: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        frames.append((event, json.loads("".join(data_lines))))
    return frames


# ---- bus unit tests ----------------------------------------------------


def test_bus_delivers_nudges_and_unsubscribes():
    async def body():
        bus = JobEventBus()
        queue = bus.subscribe("j1")
        assert bus.subscriber_count("j1") == 1
        assert bus.publish("j1", "stage") == 1
        nudge = await queue.get()
        assert nudge == {"kind": "stage"}

        bus.unsubscribe("j1", queue)
        assert bus.subscriber_count("j1") == 0
        # Unknown/unsubscribed jobs accept publishes as no-ops.
        assert bus.publish("j1", "stage") == 0
        assert bus.publish("other", "stage") == 0

    asyncio.run(body())


def test_bus_full_queue_drops_without_blocking():
    async def body():
        bus = JobEventBus(maxsize=1)
        queue = bus.subscribe("j1")
        assert bus.publish("j1", "a") == 1
        # Second nudge is dropped rather than blocking the publisher.
        assert bus.publish("j1", "b") == 0
        assert (await queue.get())["kind"] == "a"
        # Freed space takes nudges again.
        assert bus.publish("j1", "c") == 1
        assert (await queue.get())["kind"] == "c"

    asyncio.run(body())


def test_bus_fans_out_and_isolates_slow_subscriber():
    async def body():
        bus = JobEventBus(maxsize=1)
        slow = bus.subscribe("j1")
        fast = bus.subscribe("j1")
        assert bus.publish("j1", "x") == 2  # both queues take the nudge
        assert (await fast.get())["kind"] == "x"  # free the fast queue
        # Now delivered only to the non-full subscriber; the publisher
        # still reports the one reachable queue.
        assert bus.publish("j1", "y") == 1
        assert (await fast.get())["kind"] == "y"
        assert (await slow.get())["kind"] == "x"

    asyncio.run(body())


# ---- endpoint: error / terminal shapes ---------------------------------


def test_events_unknown_job_is_404(client):
    r = client.get("/v1/jobs/nope/events")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_events_terminal_job_sends_one_frame_and_closes(client, app):
    job_id = _submit(client)
    app.state.corpus.mark_job(job_id, "done")

    r = client.get(f"/v1/jobs/{job_id}/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = parse_frames(r.text)
    assert len(frames) == 1
    event, data = frames[0]
    assert event == "job"
    assert data["job_id"] == job_id
    assert data["status"] == "done"


# ---- endpoint: live worker stream --------------------------------------
# Starlette 1.6's TestClient runs the ASGI app to completion before
# returning, so a never-idle SSE response can't be consumed live through
# it. Drive the route function directly on a running loop instead: real
# corpus/bus/worker, route-level StreamingResponse.body_iterator.


_PARAMS = {
    "database": "default",
    "collection": "ingest",
    "chunk_size": 500,
    "chunk_overlap": 75,
    "embed_model": "bge-m3",
    "profile": "auto",
    "strategy": "recursive",
    "chunk_options": {},
    "add_context": False,
    "metadata": {},
    "filename": "doc.txt",
    "mime": "text/plain",
}


def _live_state(tmp_path):
    """Build the same app-state the lifespan would, minus HTTP/lifespan."""
    spool_root = tmp_path / "jobs"
    state = SimpleNamespace(
        settings=_Settings(spool_root),
        corpus=CorpusRepository(tmp_path / "corpus.db"),
        bm25=SparseBM25(tmp_path / "bm25"),
        store=_FakeStore(),
        embedder=_FakeEmbedder(),
        blob_store=BlobStore(tmp_path / "originals"),
        job_bus=JobEventBus(),
    )
    state.corpus.initialize()
    return state, spool_root


def _enqueue(state, spool_root, job_id="j1"):
    """Plant a queued row with an intact spool, like a fresh submission."""
    spool_dir = spool_root / job_id
    spool_dir.mkdir(parents=True)
    spool = spool_dir / "upload"
    spool.write_bytes(SAMPLE.encode())
    state.corpus.create_job(
        job_id,
        doc_id=f"doc-{job_id}",
        database="default",
        collection="ingest",
        filename="doc.txt",
        mime="text/plain",
        params_json=json.dumps(_PARAMS),
        spool_path=str(spool),
        max_attempts=2,
    )
    return job_id


def _request_for(state):
    fake_app = SimpleNamespace(state=state)
    scope = {"type": "http", "headers": [], "app": fake_app, "state": {}}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def test_events_stream_stages_to_done(tmp_path):
    state, spool_root = _live_state(tmp_path)
    corpus = state.corpus
    _enqueue(state, spool_root)

    # Deterministic per-stage rendezvous: the pipeline's mark_job runs
    # in an executor thread, persists the stage, then parks until the
    # driver releases it. The driver releases one stage, consumes that
    # frame, waits until the next mark is parked — so nudges can never
    # coalesce, regardless of thread scheduling.
    stages = ("parsing", "chunking", "embedding", "upserting")
    parked = {stage: threading.Event() for stage in stages}
    released = {stage: threading.Event() for stage in stages}
    orig_mark = corpus.mark_job

    def _gated_mark(job, status, **kwargs):
        # Park BEFORE persisting: SSE frames reload the row, so the
        # next stage must not be committed until this stage's frame is
        # consumed.
        if status in parked:
            parked[status].set()
            assert released[status].wait(10), f"stage never released: {status}"
        orig_mark(job, status, **kwargs)

    corpus.mark_job = _gated_mark
    state.store.upsert_parked = threading.Event()
    state.store.upsert_released = threading.Event()

    worker = IngestWorker(SimpleNamespace(state=state))

    async def body():
        request = _request_for(state)
        response = await stream_job_events(request, "j1")
        assert response.media_type == "text/event-stream"
        iterator = response.body_iterator

        # Initial queued frame before the worker can claim anything;
        # this also completes bus.subscribe.
        chunks = [await iterator.__anext__()]
        worker.start()
        try:
            for stage in stages:
                await asyncio.to_thread(parked[stage].wait, 10)
                released[stage].set()
                chunks.append(await iterator.__anext__())
            # Pipeline is parked inside store.upsert; release it so
            # finish_job can persist done and the done frame follows.
            await asyncio.to_thread(state.store.upsert_parked.wait, 10)
            state.store.upsert_released.set()
            chunks.append(await iterator.__anext__())
            # Terminal frame delivered; generator returns on its own.
        finally:
            await worker.stop()
        return chunks

    frames: list[tuple[str, dict]] = []
    for chunk in asyncio.run(body()):
        frames.extend(parse_frames(chunk))

    statuses = [data["status"] for _event, data in frames]
    assert statuses == [
        "queued",
        "parsing",
        "chunking",
        "embedding",
        "upserting",
        "done",
    ]
    assert all(event == "job" for event, _data in frames)
    final = frames[-1][1]
    assert final["doc_id"] == "doc-j1"
    assert final["chunk_count"] >= 1
    assert final["finished_ts"] is not None
    # System of record + derived index both got the document.
    assert corpus.get_document("doc-j1") is not None
    assert state.store.upserts == 1
    # Spool promoted into the content-addressed originals store.
    assert not spool_root.joinpath("j1").exists()
    digest = corpus.get_document("doc-j1")["content_hash"]
    assert state.blob_store.has(digest)


def test_events_cancel_route_pushes_flag_immediately(tmp_path):
    state, spool_root = _live_state(tmp_path)
    _enqueue(state, spool_root)

    async def body():
        request = _request_for(state)
        response = await stream_job_events(request, "j1")
        iterator = response.body_iterator

        initial = parse_frames(await iterator.__anext__())
        assert initial[0][1]["status"] == "queued"

        # Same action the HTTP route performs, on this loop: the
        # cancel flag lands AND a nudge reaches the live stream.
        cancelled_view = await cancel_job(request, "j1")
        assert cancelled_view.cancel_requested is True

        frame = parse_frames(await iterator.__anext__())
        await iterator.aclose()
        return frame[0]

    event, data = asyncio.run(body())
    assert event == "job"
    assert data["job_id"] == "j1"
    assert data["cancel_requested"] is True
    assert data["status"] == "queued"
