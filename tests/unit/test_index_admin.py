"""Tests for index rebuild, promotion and consistency administration.

Pins the §3 contract backed by ``vector_service.api.rebuild``:

- reindex submits a ``rebuild`` job; the worker rederives every leaf
  vector into a fresh physical collection parked as canary while the
  old index keeps serving (rows + BM25 stats independent);
- promote atomically rewrites binding + registry and drops the old
  physical index/stats; a parked regression gate must authorize the
  exact canary;
- consistency compares the corpus leaf set with physical rows
  (missing/orphan); repair deletes orphans and rederives missing rows;
- deleting leaves discards BM25 stats for every physical ref so the
  next query refits over surviving leaves.

Real :class:`CorpusRepository`, :class:`SparseBM25` and
:class:`IngestWorker`; an in-memory store with actual row state stands
in for Milvus. Async bodies run via ``asyncio.run`` — no pytest-asyncio.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.management import router as management_router
from vector_service.api.rebuild import router as rebuild_router
from vector_service.core.errors import CollectionAlreadyExists, CollectionNotFound
from vector_service.corpus import CorpusRepository
from vector_service.corpus.blobs import BlobStore
from vector_service.corpus.bm25 import SparseBM25
from vector_service.jobs import IngestWorker

#: Three sections → three leaves (same shape as test_job_worker).
MULTI_SECTION = (
    "# H0\n\n"
    + " ".join(f"word{i}" for i in range(30))
    + "\n\n## H1\n\n"
    + " ".join(f"token{i}" for i in range(30))
    + "\n\n## H2\n\n"
    + " ".join(f"item{i}" for i in range(30))
)


# ---- fakes ------------------------------------------------------------


class _FakeEmbedder:
    model_name = "bge-m3"
    dim = 4

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _BlockingOnceEmbedder:
    """Block the very first embed call until released."""

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


class _MemoryStore:
    """Row-bearing in-memory store: upsert/delete/browse/count are real."""

    backend_name = "memory"

    def __init__(self):
        self.databases = ["default"]
        self.colls: dict[tuple[str, str], dict] = {}

    def list_databases(self):
        return list(self.databases)

    def create_database(self, name, **_opts):
        if name not in self.databases:
            self.databases.append(name)

    def list_collections(self, db):
        return [coll for (d, coll) in self.colls if d == db]

    def collection_info(self, db, coll):
        entry = self.colls.get((db, coll))
        return SimpleNamespace(dim=entry["spec"]["vector_field"].dim) if entry else None

    def create_collection(self, **kwargs):
        key = (kwargs["database"], kwargs["name"])
        if key in self.colls:
            raise CollectionAlreadyExists(kwargs["name"])
        self.colls[key] = {"spec": kwargs, "rows": {}}

    def drop_collection(self, database, name):
        key = (database, name)
        if key not in self.colls:
            raise CollectionNotFound(name)
        del self.colls[key]

    def upsert(
        self,
        database,
        collection,
        _primary_field,
        _vector_field,
        ids,
        vectors,
        fields,
        *,
        extra_vectors=None,
        sparse_vectors=None,
    ):
        rows = self.colls[(database, collection)]["rows"]
        for i, cid in enumerate(ids):
            rows[cid] = {
                "fields": dict(fields[i]),
                "vector": vectors[i],
                "extra": {k: v[i] for k, v in (extra_vectors or {}).items()},
                "sparse": {k: v[i] for k, v in (sparse_vectors or {}).items()},
            }

    def delete(self, database, collection, _primary_field, ids, filter_expr=None):
        key = (database, collection)
        if key not in self.colls:
            raise CollectionNotFound(collection)
        rows = self.colls[key]["rows"]
        n = 0
        for cid in ids:
            if cid in rows:
                del rows[cid]
                n += 1
        return n

    def browse(
        self, database, collection, _primary_field, *, limit, offset, output_fields
    ):
        key = (database, collection)
        if key not in self.colls:
            raise CollectionNotFound(collection)
        rows = self.colls[key]["rows"]
        page = sorted(rows)[offset : offset + limit]
        return [{"id": cid, **rows[cid]["fields"]} for cid in page]

    def count_rows(self, database, collection):
        key = (database, collection)
        if key not in self.colls:
            raise CollectionNotFound(collection)
        return len(self.colls[key]["rows"])


# ---- harness ----------------------------------------------------------


class _JobsCfg:
    poll_interval_seconds = 0.05
    retry_backoff_base_seconds = 0.0
    retry_backoff_max_seconds = 0.0
    wake_on_submit = True
    max_attempts = 3


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
    "add_summary": False,
    "metadata": {},
    "filename": "doc.txt",
    "mime": "text/plain",
}


class _AdminHarness:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.corpus = CorpusRepository(tmp_path / "corpus.db")
        self.corpus.initialize()
        self.bm25 = SparseBM25(tmp_path / "bm25")
        self.store = _MemoryStore()
        self.blob_store = BlobStore(tmp_path / "originals")
        self.settings = _Settings()
        self.embedder = _FakeEmbedder()

        self.app = FastAPI()
        self.app.include_router(rebuild_router)
        self.app.include_router(management_router)

        # Mirror main.create_app's canonical error shape so every test
        # reads r.json()["error"] uniformly.
        @self.app.exception_handler(HTTPException)
        async def _http_handler(_request, exc):
            detail = exc.detail
            if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
                return JSONResponse(status_code=exc.status_code, content=detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": "error", "message": str(detail)}},
            )
        self.app.state.settings = self.settings
        self.app.state.corpus = self.corpus
        self.app.state.bm25 = self.bm25
        self.app.state.store = self.store
        self.app.state.blob_store = self.blob_store
        self.app.state.embedder = self.embedder

        self.worker = IngestWorker(self.app)
        self.app.state.job_worker = self.worker
        self.client = TestClient(self.app)

    def seed_ingest(self, content=MULTI_SECTION):
        """Run one real ingest job through the worker; returns doc_id."""
        job_id = "seed"
        doc_id = f"doc-{job_id}"
        spool_dir = self.tmp_path / "jobs" / job_id
        spool_dir.mkdir(parents=True)
        spool = spool_dir / "upload"
        spool.write_bytes(content.encode())
        self.corpus.create_job(
            job_id,
            doc_id=doc_id,
            database="default",
            collection="ingest",
            filename="doc.txt",
            mime="text/plain",
            params_json=json.dumps(_PARAMS),
            spool_path=str(spool),
        )
        return asyncio.run(self._drain(job_id))

    async def _drain(self, job_id):
        self.worker.start()
        try:
            return await _wait_terminal(self.corpus, job_id)
        finally:
            await self.worker.stop()

    def run_job(self, job_id):
        return asyncio.run(self._drain(job_id))

    def submit_reindex(self, **body_overrides):
        body = {"canary_percent": 10, "batch_size": 64}
        body.update(body_overrides)
        return self.client.post(
            "/v1/databases/default/collections/ingest/reindex", json=body
        )

    def leaf_ids(self):
        with self.corpus._txn() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE level='chunk' ORDER BY chunk_index"
            ).fetchall()
        return [row["chunk_id"] for row in rows]

    def bm25_path(self, ref):
        return self.bm25._state_file("default", ref)


async def _wait_terminal(corpus, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = corpus.get_job(job_id)
        if row["status"] in ("done", "failed", "cancelled"):
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


@pytest.fixture
def h(tmp_path):
    return _AdminHarness(tmp_path)


# ---- submit ------------------------------------------------------------


def test_reindex_rejects_non_corpus_collection(h):
    r = h.client.post(
        "/v1/databases/default/collections/ghost/reindex",
        json={"canary_percent": 0, "batch_size": 64},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_reindex_submit_queues_rebuild_job(h):
    h.seed_ingest()
    r = h.submit_reindex()
    assert r.status_code == 202
    payload = r.json()
    assert payload["status"] == "queued"
    target_ref = payload["index_ref"]
    row = h.corpus.get_job(payload["job_id"])
    assert row["job_type"] == "rebuild"
    params = json.loads(row["params_json"])
    assert params["target_ref"] == target_ref
    # Nothing built yet — target collection absent.
    assert target_ref not in h.store.list_collections("default")


# ---- rebuild pipeline --------------------------------------------------


def test_rebuild_parks_canary_and_keeps_old_index(h):
    h.seed_ingest()
    total = len(h.leaf_ids())

    r = h.submit_reindex(canary_percent=25)
    target_ref = r.json()["index_ref"]
    job_id = r.json()["job_id"]

    row = h.run_job(job_id)
    assert row["status"] == "done"
    assert row["chunk_count"] == total

    # New physical index fully populated...
    assert h.store.count_rows("default", target_ref) == total
    # ...with dense + summary + sparse vectors per row.
    stored = h.store.colls[("default", target_ref)]["rows"]
    for data in stored.values():
        assert data["vector"] and data["extra"]["summary_vector"]
        assert data["sparse"]["sparse"]

    # Old index still present and serving (active unchanged).
    assert h.store.count_rows("default", "ingest") == total
    binding = h.corpus.get_binding("default", "ingest")
    assert binding["active_ref"] == "ingest"
    assert binding["canary_ref"] == target_ref
    assert binding["canary_percent"] == 25
    assert binding["model"] == "bge-m3"

    # Each physical index has its own independent BM25 stats set.
    assert h.bm25_path(target_ref).is_file()
    assert h.bm25_path("ingest").is_file()


# ---- promote ------------------------------------------------------------


def _park_canary(h):
    submit = h.submit_reindex().json()
    assert h.run_job(submit["job_id"])["status"] == "done"
    return submit["index_ref"]


def test_promote_switches_binding_and_drops_old(h):
    h.seed_ingest()
    total = len(h.leaf_ids())
    target_ref = _park_canary(h)

    r = h.client.post(
        "/v1/databases/default/collections/ingest/reindex/promote"
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["active_ref"] == target_ref
    assert payload["retired_ref"] == "ingest"
    assert payload["model"] == "bge-m3"

    binding = h.corpus.get_binding("default", "ingest")
    assert binding["active_ref"] == target_ref
    assert binding["canary_ref"] is None
    assert binding["canary_percent"] == 0

    # Old physical index + BM25 state retired; new one kept.
    assert "ingest" not in h.store.list_collections("default")
    assert h.store.count_rows("default", target_ref) == total
    assert not h.bm25_path("ingest").exists()
    assert h.bm25_path(target_ref).is_file()

    # Registry rows (dense/sparse/summary) now point at the new physical
    # index; summary keeps its ":summary_vector" ref suffix.
    with h.corpus._txn() as conn:
        refs = {
            row["index_ref"]
            for row in conn.execute("SELECT index_ref FROM chunk_indexes").fetchall()
        }
    assert refs == {target_ref, f"{target_ref}:summary_vector"}


def test_promote_without_canary_is_409(h):
    h.seed_ingest()
    r = h.client.post(
        "/v1/databases/default/collections/ingest/reindex/promote"
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reindex_no_canary"


def test_promote_blocked_by_gate_until_check_passes_for_canary(h):
    h.seed_ingest()
    target_ref = _park_canary(h)

    # Gate FKs require a real eval set + frozen version.
    h.corpus.create_eval_set(
        "set-1", database="default", collection="ingest", name="regression set"
    )
    h.corpus.create_eval_version(
        "ver-1", set_id="set-1", tag="v1", question_count=0, snapshot_json="{}"
    )
    h.corpus.create_regression_gate(
        "gate-1",
        database="default",
        collection="ingest",
        set_id="set-1",
        version_id="ver-1",
    )
    # Latest check failed → promotion blocked.
    h.corpus.create_gate_check(
        "check-1",
        gate_id="gate-1",
        run_id="run-1",
        candidate_ref=target_ref,
        status="failed",
    )
    r = h.client.post(
        "/v1/databases/default/collections/ingest/reindex/promote"
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "gate_blocked"

    # A pass for a DIFFERENT ref does not authorize this canary.
    h.corpus.create_gate_check(
        "check-2",
        gate_id="gate-1",
        run_id="run-2",
        candidate_ref="some-other-ref",
        status="passed",
    )
    r = h.client.post(
        "/v1/databases/default/collections/ingest/reindex/promote"
    )
    assert r.status_code == 409

    # A pass matching this exact canary → promotion proceeds.
    h.corpus.create_gate_check(
        "check-3",
        gate_id="gate-1",
        run_id="run-3",
        candidate_ref=target_ref,
        status="passed",
    )
    r = h.client.post(
        "/v1/databases/default/collections/ingest/reindex/promote"
    )
    assert r.status_code == 200
    assert r.json()["active_ref"] == target_ref


# ---- consistency --------------------------------------------------------


def test_consistency_reports_ok(h):
    h.seed_ingest()
    total = len(h.leaf_ids())
    r = h.client.post(
        "/v1/databases/default/collections/ingest/consistency", json={}
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["repaired"] is False
    assert payload["corpus_leaves"] == total
    item = payload["indexes"][0]
    assert item["ref"] == "ingest"
    assert item["milvus_rows"] == total
    assert item["missing_in_milvus"] == []
    assert item["orphans_in_milvus"] == []


def test_consistency_finds_missing_and_orphan(h):
    h.seed_ingest()
    leaf_ids = h.leaf_ids()

    rows = h.store.colls[("default", "ingest")]["rows"]
    del rows[leaf_ids[0]]  # a corpus leaf missing from the physical index
    rows["junk-orphan"] = {  # physical row with no corpus counterpart
        "fields": {}, "vector": [], "extra": {}, "sparse": {},
    }

    r = h.client.post(
        "/v1/databases/default/collections/ingest/consistency", json={}
    )
    payload = r.json()
    assert payload["ok"] is False
    item = payload["indexes"][0]
    assert item["missing_in_milvus"] == [leaf_ids[0]]
    assert item["orphans_in_milvus"] == ["junk-orphan"]


def test_consistency_repair_rederives_and_cleans(h):
    h.seed_ingest()
    leaf_ids = h.leaf_ids()

    rows = h.store.colls[("default", "ingest")]["rows"]
    del rows[leaf_ids[0]]
    rows["junk-orphan"] = {
        "fields": {}, "vector": [], "extra": {}, "sparse": {},
    }

    r = h.client.post(
        "/v1/databases/default/collections/ingest/consistency",
        json={"repair": True},
    )
    payload = r.json()
    assert payload["ok"] is True
    assert payload["repaired"] is True

    # Post-repair physical state exactly mirrors the corpus.
    stored = h.store.colls[("default", "ingest")]["rows"]
    assert set(stored) == set(leaf_ids)
    rederived = stored[leaf_ids[0]]
    assert rederived["vector"]
    assert rederived["extra"]["summary_vector"]
    assert rederived["sparse"]["sparse"]


def test_consistency_repair_without_embedder_is_503(h):
    h.seed_ingest()
    h.app.state.embedder = None
    r = h.client.post(
        "/v1/databases/default/collections/ingest/consistency",
        json={"repair": True},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "embedder_unavailable"


# ---- cancellation -------------------------------------------------------


def test_rebuild_cancel_cleans_target(h):
    h.seed_ingest()
    # Force one leaf per batch; a blocking embedder parks the worker
    # inside the first batch so cancellation is observed at the gate.
    blocking = _BlockingOnceEmbedder()
    h.app.state.embedder = blocking
    submit = h.submit_reindex(batch_size=1).json()
    target_ref = submit["index_ref"]
    job_id = submit["job_id"]

    async def body():
        h.worker.start()
        try:
            while not blocking.started.wait(0.01):
                await asyncio.sleep(0.01)
            h.corpus.request_cancel(job_id)
            blocking.release.set()
            return await _wait_terminal(h.corpus, job_id)
        finally:
            await h.worker.stop()

    row = asyncio.run(body())
    assert row["status"] == "cancelled"
    # Half-written target cleaned: physical collection and BM25 state.
    assert target_ref not in h.store.list_collections("default")
    assert not h.bm25_path(target_ref).exists()


# ---- BM25 invalidation on delete ----------------------------------------


def test_delete_leaves_discards_bm25_for_every_physical_ref(h):
    h.seed_ingest()
    target_ref = _park_canary(h)

    # Both refs carry independent stats while the canary is parked.
    assert h.bm25_path("ingest").is_file()
    assert h.bm25_path(target_ref).is_file()

    victim = h.leaf_ids()[0]
    r = h.client.post(
        "/v1/databases/default/collections/ingest/vectors/delete",
        json={"primary_field": "id", "ids": [victim]},
    )
    assert r.status_code == 200

    # Stats for both refs discarded — next query refits over survivors.
    assert not h.bm25_path("ingest").exists()
    assert not h.bm25_path(target_ref).exists()
    remaining = [cid for cid in h.leaf_ids() if cid != victim]
    assert h.store.count_rows("default", "ingest") == len(remaining)
    assert h.store.count_rows("default", target_ref) == len(remaining)
