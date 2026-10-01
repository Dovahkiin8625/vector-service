"""Periodic storage maintenance (TODO §6): backup, VACUUM gating, blob sweep.

Real :class:`CorpusRepository` and :class:`BlobStore` over temp dirs;
settings are the real BackupSettings/MaintenanceSettings with paths
redirected into the tmp directory. Pins:

- prune_backups keeps the newest N snapshots, ignores other files;
- a cycle writes one consistent backup (openable DB) before VACUUM;
- VACUUM runs at/above the free-ratio threshold, skips below it;
- blob sweep deletes unreferenced originals, keeps referenced ones;
- the started worker honors wake() and cleanly stops;
- a disabled worker performs no work.
"""
from __future__ import annotations

import asyncio
import functools
import sqlite3
import time
from types import SimpleNamespace

from vector_service.core.config import BackupSettings, MaintenanceSettings
from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.corpus.blobs import BlobStore
from vector_service.jobs.maintenance import MaintenanceWorker, prune_backups


def async_test(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


# ---- builders ----------------------------------------------------------


def _corpus_with_doc(tmp_path, *, hash_="a" * 64):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    doc = DocumentRecord(
        doc_id="d1", database="default", collection="ingest",
        filename="a.txt", mime="text/plain", content_hash=hash_,
    )
    chunks = [
        ChunkRecord(chunk_id="d1_0", doc_id="d1", database="default",
                    collection="ingest", chunk_index=0, text="内容"),
    ]
    repo.store_document(doc, chunks)
    return repo


def _settings(tmp_path, *, backup_enabled=True, retain=2,
              vacuum_min_free_ratio=0.2, blob_sweep_enabled=False):
    return SimpleNamespace(
        backup=BackupSettings(
            enabled=backup_enabled, dir=tmp_path / "backups", retain=retain,
        ),
        maintenance=MaintenanceSettings(
            enabled=True, interval_seconds=3600.0,
            vacuum_min_free_ratio=vacuum_min_free_ratio,
            blob_sweep_enabled=blob_sweep_enabled,
        ),
    )


def _worker(repo, settings, *, blob_store=None):
    state = SimpleNamespace(
        settings=settings, corpus=repo, blob_store=blob_store,
    )
    return MaintenanceWorker(SimpleNamespace(state=state))


# ---- prune_backups -----------------------------------------------------


def test_prune_backups_keeps_newest_and_ignores_other_files(tmp_path):
    stamps = [
        "20260101T000000Z", "20260102T000000Z",
        "20260103T000000Z", "20260104T000000Z",
    ]
    for stamp in stamps:
        (tmp_path / f"corpus-{stamp}.db").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"keep me")

    removed = prune_backups(tmp_path, retain=2)

    assert removed == 2
    remaining = sorted(p.name for p in tmp_path.glob("corpus-*.db"))
    assert remaining == [
        "corpus-20260103T000000Z.db", "corpus-20260104T000000Z.db",
    ]
    assert (tmp_path / "notes.txt").is_file()


# ---- backup cycle ------------------------------------------------------


@async_test
async def test_cycle_writes_backup_then_prunes(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(tmp_path, retain=2)
    worker = _worker(repo, settings)

    await worker._cycle()

    backups = list((tmp_path / "backups").glob("corpus-*.db"))
    assert len(backups) == 1
    # The snapshot is a complete, independently openable database.
    with sqlite3.connect(str(backups[0])) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
    assert rows == 1


@async_test
async def test_cycle_skips_backup_when_disabled(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(tmp_path, backup_enabled=False)
    worker = _worker(repo, settings)

    await worker._cycle()

    assert list((tmp_path / "backups").glob("corpus-*.db")) == []


# ---- VACUUM gating -----------------------------------------------------


@async_test
async def test_cycle_vacuum_runs_at_zero_threshold(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(
        tmp_path, backup_enabled=False, vacuum_min_free_ratio=0.0,
    )
    calls = []
    repo.vacuum = _spy(calls, repo.vacuum)
    worker = _worker(repo, settings)

    await worker._cycle()

    assert len(calls) == 1


@async_test
async def test_cycle_vacuum_skips_below_ratio(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(
        tmp_path, backup_enabled=False, vacuum_min_free_ratio=0.99,
    )
    calls = []
    repo.vacuum = _spy(calls, repo.vacuum)
    worker = _worker(repo, settings)

    await worker._cycle()

    assert len(calls) == 0


def _spy(calls, original):
    def wrapper():
        calls.append(1)
        return original()

    return wrapper


# ---- blob sweep --------------------------------------------------------


@async_test
async def test_cycle_blob_sweep_removes_orphans_keeps_referenced(tmp_path):
    referenced_hash = "b" * 64
    orphan_hash = "c" * 64
    repo = _corpus_with_doc(tmp_path, hash_=referenced_hash)
    blob_store = BlobStore(tmp_path / "originals")
    blob_store.put_bytes(b"referenced bytes", referenced_hash)
    blob_store.put_bytes(b"orphan bytes", orphan_hash)

    settings = _settings(
        tmp_path, backup_enabled=False, vacuum_min_free_ratio=1.0,
        blob_sweep_enabled=True,
    )
    worker = _worker(repo, settings, blob_store=blob_store)

    await worker._cycle()

    assert blob_store.has(referenced_hash)
    assert not blob_store.has(orphan_hash)


# ---- start / wake / stop -----------------------------------------------


@async_test
async def test_worker_wake_runs_cycle_and_stops(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(tmp_path)
    worker = _worker(repo, settings)

    worker.start()
    worker.wake()
    deadline = time.time() + 5
    while time.time() < deadline:
        if list((tmp_path / "backups").glob("corpus-*.db")):
            break
        await asyncio.sleep(0.02)
    await worker.stop()

    assert list((tmp_path / "backups").glob("corpus-*.db")) != []


@async_test
async def test_worker_disabled_does_nothing(tmp_path):
    repo = _corpus_with_doc(tmp_path)
    settings = _settings(tmp_path)
    settings.maintenance.enabled = False
    worker = _worker(repo, settings)

    await worker.run()

    assert list((tmp_path / "backups").glob("corpus-*.db")) == []
