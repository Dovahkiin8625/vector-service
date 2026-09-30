"""Tests for startup recovery (:func:`recover_interrupted`).

Real :class:`CorpusRepository` / :class:`SparseBM25` on temp dirs,
in-memory fakes for embedder and store. The suite pins:

- stage-status job with an intact spool → half-written state cleaned,
  back to ``queued`` with attempts+1, then the worker runs it done;
- attempts already at ``max_attempts - 1`` → terminal ``failed`` with
  ``interrupted`` instead of another requeue;
- missing spool (NULL path or deleted file) → terminal ``failed`` with
  ``interrupted``;
- cancel flag carried over from the dead process → ``cancelled``;
- queued / done rows are not touched.

Async bodies run via ``asyncio.run`` like the rest of the repo.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.corpus.bm25 import SparseBM25
from vector_service.jobs import IngestWorker, recover_interrupted

SAMPLE = (
    "The quarterly report covers revenue and margins for the unit. "
    "Each sentence adds a little more retrievable text.\n\n"
) * 6


# ---- fakes (mirror test_job_worker) -----------------------------------


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


class _FakeEmbedder:
    model_name = "bge-m3"
    dim = 4

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


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


# ---- harness ----------------------------------------------------------


class _Harness:
    def __init__(self, tmp_path):
        self.spool_root = tmp_path / "jobs"
        self.corpus = CorpusRepository(tmp_path / "corpus.db")
        self.corpus.initialize()
        self.bm25 = SparseBM25(tmp_path / "bm25")
        self.store = _FakeStore()
        self.embedder = _FakeEmbedder()
        state = SimpleNamespace(
            settings=_Settings(),
            corpus=self.corpus,
            bm25=self.bm25,
            store=self.store,
            embedder=self.embedder,
        )
        self.app = SimpleNamespace(state=state)
        self.worker = IngestWorker(self.app)

    def spool_dir_for(self, job_id):
        return self.spool_root / job_id


@pytest.fixture
def harness(tmp_path):
    return _Harness(tmp_path)


def _insert_interrupted(
    h: _Harness,
    job_id: str = "j1",
    *,
    stage: str = "embedding",
    attempts: int = 0,
    max_attempts: int = 3,
    spool: str | None = "present",
    cancel: bool = False,
    half_doc: bool = False,
):
    """Plant a job the way a dead process would leave it.

    ``spool`` is ``"present"`` / ``"unlinked"`` / ``"none"``.
    """
    spool_path: Path | None = None
    if spool != "none":
        spool_dir = h.spool_root / job_id
        spool_dir.mkdir(parents=True)
        spool_path = spool_dir / "upload"
        spool_path.write_bytes(SAMPLE.encode())

    h.corpus.create_job(
        job_id,
        doc_id=f"doc-{job_id}",
        database="default",
        collection="ingest",
        filename="doc.txt",
        mime="text/plain",
        params_json=json.dumps(_PARAMS),
        spool_path=str(spool_path) if spool_path is not None else None,
        max_attempts=max_attempts,
    )
    h.corpus.mark_job(job_id, stage)
    with h.corpus._txn() as conn:
        conn.execute(
            """
            UPDATE ingest_jobs
            SET attempts = ?, cancel_requested = ?
            WHERE job_id = ?
            """,
            (attempts, int(cancel), job_id),
        )

    if half_doc:
        h.corpus.store_document(
            DocumentRecord(
                doc_id=f"doc-{job_id}",
                database="default",
                collection="ingest",
                filename="doc.txt",
                mime="text/plain",
                content_hash="a" * 64,
            ),
            [
                ChunkRecord(
                    chunk_id=f"doc-{job_id}_0",
                    doc_id=f"doc-{job_id}",
                    database="default",
                    collection="ingest",
                    chunk_index=0,
                    text="half written chunk",
                )
            ],
        )
    return h.corpus.get_job(job_id)


async def _wait_terminal(corpus, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = corpus.get_job(job_id)
        if row["status"] in ("done", "failed", "cancelled"):
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


# ---- requeue path -----------------------------------------------------


def test_recovery_requeues_and_worker_finishes(harness):
    _insert_interrupted(harness, stage="embedding", attempts=0)

    async def body():
        recovered = await recover_interrupted(harness.app)
        row = harness.corpus.get_job("j1")
        assert recovered == 1
        assert row["status"] == "queued"
        assert row["attempts"] == 1
        assert row["not_before_ts"] is None

        harness.worker.start()
        try:
            done_row = await _wait_terminal(harness.corpus, "j1")
        finally:
            await harness.worker.stop()
        return done_row

    done_row = asyncio.run(body())
    assert done_row["status"] == "done"
    assert harness.corpus.get_document("doc-j1") is not None
    assert harness.store.upserts == 1


def test_recovery_cleans_half_written_document(harness):
    _insert_interrupted(harness, stage="upserting", half_doc=True)

    async def body():
        await recover_interrupted(harness.app)

    asyncio.run(body())

    # Corpus rows gone, thin-index delete was issued for this doc.
    assert harness.corpus.get_document("doc-j1") is None
    assert harness.store.deleted
    collection, expr = harness.store.deleted[0]
    assert collection == "ingest"
    assert 'doc-j1' in expr
    row = harness.corpus.get_job("j1")
    assert row["status"] == "queued"
    # Spool is retained for the retry.
    assert harness.spool_dir_for("j1").joinpath("upload").is_file()


def test_recovery_fails_when_attempts_exhausted(harness):
    _insert_interrupted(harness, attempts=2, max_attempts=3, half_doc=True)

    async def body():
        await recover_interrupted(harness.app)

    asyncio.run(body())

    row = harness.corpus.get_job("j1")
    assert row["status"] == "failed"
    assert row["error_code"] == "interrupted"
    assert row["attempts"] == 3
    assert harness.corpus.get_document("doc-j1") is None
    assert not harness.spool_dir_for("j1").exists()


# ---- unrecoverable paths ----------------------------------------------


def test_recovery_fails_when_spool_missing(harness):
    # Both flavors: NULL spool path, and spool path pointing at a
    # deleted file.
    _insert_interrupted(harness, "j1", spool="none", half_doc=True)
    _insert_interrupted(harness, "j2", spool="unlinked", half_doc=True)
    harness.spool_root.joinpath("j2", "upload").unlink()

    async def body():
        recovered = await recover_interrupted(harness.app)
        assert recovered == 2

    asyncio.run(body())

    for job_id in ("j1", "j2"):
        row = harness.corpus.get_job(job_id)
        assert row["status"] == "failed"
        assert row["error_code"] == "interrupted"
        assert harness.corpus.get_document(f"doc-{job_id}") is None


def test_recovery_honors_pending_cancel(harness):
    _insert_interrupted(harness, cancel=True, half_doc=True)

    async def body():
        await recover_interrupted(harness.app)

    asyncio.run(body())

    row = harness.corpus.get_job("j1")
    assert row["status"] == "cancelled"
    assert row["cancel_requested"] == 1
    assert harness.corpus.get_document("doc-j1") is None
    assert not harness.spool_dir_for("j1").exists()


# ---- non-interrupted rows are untouched -------------------------------


def test_recovery_leaves_other_jobs_untouched(harness):
    _insert_interrupted(harness, "running")
    queued_id = "q1"
    queued_spool = harness.spool_root / queued_id / "upload"
    queued_spool.parent.mkdir(parents=True)
    queued_spool.write_bytes(SAMPLE.encode())
    harness.corpus.create_job(
        queued_id,
        doc_id="doc-q1",
        database="default",
        collection="ingest",
        filename="doc.txt",
        mime="text/plain",
        params_json=json.dumps(_PARAMS),
        spool_path=str(queued_spool),
    )

    async def body():
        # Only the stage row is recovered; the queued row is left
        # exactly as submitted and can still be claimed.
        recovered = await recover_interrupted(harness.app)
        assert recovered == 1

        harness.worker.start()
        try:
            rows = await asyncio.gather(
                _wait_terminal(harness.corpus, "running"),
                _wait_terminal(harness.corpus, queued_id),
            )
        finally:
            await harness.worker.stop()
        return rows

    rows = asyncio.run(body())
    assert {row["status"] for row in rows} == {"done"}
