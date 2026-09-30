"""Tests for the in-process :class:`IngestWorker`.

Real :class:`CorpusRepository` and :class:`SparseBM25` against temp
dirs; in-memory fakes stand in for the embedder and vector store. The
suite pins:

- the success path: queued row → four stage marks → done, corpus +
  thin-index rows written, spool removed;
- retry semantics: 5xx failures requeue (with the attempt count) up
  to ``max_attempts`` then fail; 4xx failures fail immediately;
- cancellation at a stage boundary: rollback removes the half-written
  corpus/thin-index state and the job ends cancelled;
- an unreadable spool/params row fails as ``job_corrupted``.

No pytest-asyncio here — tests run async bodies via ``asyncio.run``,
same as the rest of the repo.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from vector_service.core.errors import EmbedderError
from vector_service.corpus import CorpusRepository
from vector_service.corpus.blobs import BlobStore
from vector_service.corpus.bm25 import SparseBM25
from vector_service.jobs import IngestWorker

SAMPLE = (
    "The quarterly report covers revenue and margins for the unit. "
    "Each sentence adds a little more retrievable text.\n\n"
) * 6


# ---- fakes ------------------------------------------------------------


class _FakeEmbedder:
    """Dense embedder; optionally fail the first ``fail_times`` calls."""

    model_name = "bge-m3"
    dim = 4

    def __init__(self, *, fail_times=0):
        self._fail_times = fail_times
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise EmbedderError("simulated CUDA OOM")
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _BlockingEmbedder:
    """Embedder that blocks the first call until ``release`` is set.

    ``started`` lets the test know the worker is parked inside embed.
    """

    model_name = "bge-m3"
    dim = 4

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if not self.started.is_set():
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("blocking embedder was never released")
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeAdapter:
    class _Client:
        def __init__(self, deleted):
            self.deleted = deleted

        def delete(self, collection, filter):
            self.deleted.append((collection, filter))

    def __init__(self, deleted):
        self._client = self._Client(deleted)


class _FakeStore:
    """In-memory store supporting collection state + delete spy."""

    def __init__(self, *, existing_dim=None):
        # None → collection absent (gets created); int → collection
        # exists and collection_info reports that dim.
        self.existing_dim = existing_dim
        self.deleted: list[tuple] = []
        self.upserts = 0
        self.upsert_args: list[tuple] = []
        self._adapter = _FakeAdapter(self.deleted)

    def list_databases(self):
        return ["default"]

    def create_database(self, name):
        pass

    def list_collections(self, database):
        return ["ingest"] if self.existing_dim is not None else []

    def collection_info(self, database, collection):
        return SimpleNamespace(dim=self.existing_dim)

    def create_collection(self, **kwargs):
        pass

    def upsert(self, *args, **kwargs):
        self.upserts += 1
        self.upsert_args.append(args)


# ---- harness ----------------------------------------------------------


class _JobsCfg:
    poll_interval_seconds = 0.05
    retry_backoff_base_seconds = 0.0
    retry_backoff_max_seconds = 0.0
    wake_on_submit = True


class _Settings:
    inference_timeout_seconds = 30.0
    jobs = _JobsCfg()


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


class _Harness:
    def __init__(self, tmp_path, *, store=None, embedder=None):
        self.spool_root = tmp_path / "jobs"
        self.corpus = CorpusRepository(tmp_path / "corpus.db")
        self.corpus.initialize()
        self.bm25 = SparseBM25(tmp_path / "bm25")
        self.store = store or _FakeStore()
        self.embedder = embedder or _FakeEmbedder()
        self.blob_store = BlobStore(tmp_path / "originals")
        state = SimpleNamespace(
            settings=_Settings(),
            corpus=self.corpus,
            bm25=self.bm25,
            store=self.store,
            embedder=self.embedder,
            blob_store=self.blob_store,
        )
        self.app = SimpleNamespace(state=state)
        self.worker = IngestWorker(self.app)

    def enqueue(self, job_id="j1", *, content=SAMPLE, max_attempts=3):
        spool_dir = self.spool_root / job_id
        spool_dir.mkdir(parents=True)
        spool = spool_dir / "upload"
        data = content.encode() if isinstance(content, str) else content
        spool.write_bytes(data)
        self.corpus.create_job(
            job_id,
            doc_id=f"doc-{job_id}",
            database="default",
            collection="ingest",
            filename="doc.txt",
            mime="text/plain",
            params_json=json.dumps(_PARAMS),
            spool_path=str(spool),
            max_attempts=max_attempts,
        )
        return job_id

    def spool_dir_for(self, job_id):
        return self.spool_root / job_id


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


async def _wait_terminal(corpus, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = corpus.get_job(job_id)
        if row["status"] in ("done", "failed", "cancelled"):
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


async def _wait_spool_gone(harness, job_id, *, timeout=5.0):
    # The terminal row is written one step before the worker discards
    # the spool — wait for that last synchronous step too.
    deadline = time.time() + timeout
    while harness.spool_dir_for(job_id).exists() and time.time() < deadline:
        await asyncio.sleep(0.01)


def _run(coro):
    return asyncio.run(coro)


# ---- success ----------------------------------------------------------


def test_worker_runs_job_to_done(harness):
    async def body():
        job_id = harness.enqueue()

        # Spy on stage marks (worker reaches the repo through the
        # instance, so patching the method is enough).
        stages = []
        orig_mark = harness.corpus.mark_job

        def _spy(spied_id, status, **kwargs):
            stages.append(status)
            orig_mark(spied_id, status, **kwargs)

        harness.corpus.mark_job = _spy

        harness.worker.start()
        try:
            row = await _wait_terminal(harness.corpus, job_id)
            await _wait_spool_gone(harness, job_id)
        finally:
            await harness.worker.stop()
        return row, stages

    row, stages = _run(body())

    assert row["status"] == "done"
    assert stages == ["parsing", "chunking", "embedding", "upserting"]
    # System of record + derived index both got the document.
    assert harness.corpus.get_document("doc-j1") is not None
    assert harness.store.upserts == 1
    assert row["chunk_count"] >= 1
    assert row["doc_id"] == "doc-j1"
    # Spool gone — original bytes promoted into the content-addressed
    # originals store under the document's hash.
    assert not harness.spool_dir_for("j1").exists()
    digest = harness.corpus.get_document("doc-j1")["content_hash"]
    assert harness.blob_store.has(digest)


# ---- retry: 5xx -------------------------------------------------------


def test_worker_retries_5xx_until_max_attempts(tmp_path):
    h = _Harness(tmp_path, embedder=_FakeEmbedder(fail_times=99))

    async def body():
        job_id = h.enqueue(max_attempts=2)
        h.worker.start()
        try:
            row = await _wait_terminal(h.corpus, job_id)
            await _wait_spool_gone(h, job_id)
        finally:
            await h.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    assert row["error_code"] == "embedder_unavailable"
    assert h.embedder.calls == 2  # one inference per attempt
    assert not h.spool_dir_for("j1").exists()


def test_worker_retries_when_embedder_absent(tmp_path):
    h = _Harness(tmp_path)
    h.app.state.embedder = None

    async def body():
        job_id = h.enqueue(max_attempts=2)
        h.worker.start()
        try:
            row = await _wait_terminal(h.corpus, job_id)
        finally:
            await h.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    assert row["error_code"] == "embedder_unavailable"


def test_worker_succeeds_after_transient_failure(tmp_path):
    # Fail the first (dense) embed, then succeed on the retry.
    h = _Harness(tmp_path, embedder=_FakeEmbedder(fail_times=1))

    async def body():
        job_id = h.enqueue()
        h.worker.start()
        try:
            row = await _wait_terminal(h.corpus, job_id)
        finally:
            await h.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "done"
    assert row["attempts"] == 1
    assert h.embedder.calls == 2  # failed dense, retried dense


# ---- no retry: 4xx ----------------------------------------------------


def test_worker_does_not_retry_4xx(tmp_path):
    # Collection exists with a different dim → 422 dimension_mismatch.
    h = _Harness(tmp_path, store=_FakeStore(existing_dim=999))

    async def body():
        job_id = h.enqueue(max_attempts=3)
        h.worker.start()
        try:
            row = await _wait_terminal(h.corpus, job_id)
            await _wait_spool_gone(h, job_id)
        finally:
            await h.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "failed"
    assert row["attempts"] == 1
    assert row["error_code"] == "dimension_mismatch"
    # Rollback ran: the document committed before the dim check is gone.
    assert h.corpus.get_document("doc-j1") is None
    assert not h.spool_dir_for("j1").exists()


# ---- cancel -----------------------------------------------------------


def test_worker_cancels_at_stage_boundary(tmp_path):
    embedder = _BlockingEmbedder()
    h = _Harness(tmp_path, embedder=embedder)

    async def body():
        job_id = h.enqueue()
        h.worker.start()
        try:
            # Wait until the worker is parked inside the dense embed.
            while not embedder.started.wait(0.01):
                await asyncio.sleep(0.01)
            h.corpus.request_cancel(job_id)
            embedder.release.set()
            row = await _wait_terminal(h.corpus, job_id)
            await _wait_spool_gone(h, job_id)
        finally:
            embedder.release.set()
            await h.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "cancelled"
    assert row["cancel_requested"] == 1
    assert row["finished_ts"] is not None
    # Rollback reached every layer.
    assert h.corpus.get_document("doc-j1") is None
    assert h.store.deleted != []
    assert not h.spool_dir_for("j1").exists()


# ---- corrupted row ----------------------------------------------------


def test_worker_fails_corrupted_job(harness):
    async def body():
        job_id = harness.enqueue()
        # Drop the spool file after submission.
        spool = harness.spool_root / job_id / "upload"
        spool.unlink()
        harness.worker.start()
        try:
            row = await _wait_terminal(harness.corpus, job_id)
        finally:
            await harness.worker.stop()
        return row

    row = _run(body())
    assert row["status"] == "failed"
    assert row["error_code"] == "job_corrupted"
    assert row["attempts"] == 0  # never got a real attempt
