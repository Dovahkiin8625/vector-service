"""Periodic storage-maintenance worker.

A second background loop alongside :class:`~vector_service.jobs.worker.IngestWorker`:

- **VACUUM gating**: SQLite only grows; DELETE/UPDATE leave pages on the
  freelist. Each cycle reads the free-page ratio and runs VACUUM + a
  TRUNCATE WAL checkpoint when it crosses the configured threshold.
- **Backup**: a consistent snapshot of the corpus via the sqlite online
  backup API, written before any VACUUM, with age-count retention.
  Milvus vectors are not backed up (rebuildable derived index).
- **Blob sweep**: originals in the content-addressed blob store are
  removed when no document references their hash. Delete routes do this
  eagerly; the sweep catches anything a crash left behind.

VACUUM and repository reads run in the sqlite pool; filesystem walks and
blob deletes on the default executor. A failed cycle is logged and retried
next interval — maintenance never kills the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vector_service.core.logging import get_logger
from vector_service.core.threadpools import run_in_sqlite

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


def prune_backups(directory: Path, retain: int) -> int:
    """Keep the newest ``retain`` corpus snapshots; return removed count.

    Snapshot names carry a UTC ``YYYYMMDDTHHMMSSZ`` stamp, so lexical
    order is chronological. Non-snapshot files in the directory are
    left alone.
    """
    files = sorted(directory.glob("corpus-*.db"), reverse=True)
    stale = files[retain:]
    for path in stale:
        path.unlink()
    return len(stale)


class MaintenanceWorker:
    """Runs VACUUM + blob-sweep cycles at a fixed interval."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app
        self._wake_event: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._stopped = False

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("maintenance worker already started")
        self._wake_event = asyncio.Event()
        self._stopped = False
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stopped = True
        if self._wake_event is not None:
            self._wake_event.set()
        task = self._task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        self._wake_event = None

    def wake(self) -> None:
        """Interrupt the interval wait (used by tests)."""
        if self._wake_event is not None:
            self._wake_event.set()

    async def run(self) -> None:
        settings = self._app.state.settings
        maint = settings.maintenance
        if not maint.enabled:
            log.info("maintenance_disabled")
            return
        interval = maint.interval_seconds
        # A fresh process has nothing to reclaim — first cycle after one
        # full interval rather than racing the startup burst.
        while not self._stopped:
            try:
                await self._wait(interval)
                if self._stopped:
                    break
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("maintenance_cycle_failed")

    async def _wait(self, interval: float) -> None:
        event = self._wake_event
        if event is None:
            await asyncio.sleep(interval)
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=interval)
        except TimeoutError:
            pass
        event.clear()

    async def _cycle(self) -> None:
        state = self._app.state
        settings = state.settings
        maint = settings.maintenance
        repo = state.corpus

        # Snapshot first, before anything mutates the live file. Only
        # SQLite is backed up: Milvus vectors rebuild from the corpus.
        backup_cfg = settings.backup
        if backup_cfg.enabled:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            dest = Path(backup_cfg.dir) / f"corpus-{stamp}.db"
            t0 = time.perf_counter()
            log.info("backup_start", dest=str(dest))
            await run_in_sqlite(repo.backup_to, dest)
            pruned = await asyncio.to_thread(
                prune_backups, Path(backup_cfg.dir), backup_cfg.retain
            )
            log.info(
                "backup_done",
                duration_seconds=round(time.perf_counter() - t0, 3),
                retain=backup_cfg.retain,
                pruned=pruned,
            )

        free, pages = await run_in_sqlite(repo.fragmentation)
        ratio = free / pages if pages else 0.0
        if ratio >= maint.vacuum_min_free_ratio:
            t0 = time.perf_counter()
            log.info(
                "vacuum_start",
                free_pages=free,
                pages=pages,
                ratio=round(ratio, 3),
            )
            await run_in_sqlite(repo.vacuum)
            log.info(
                "vacuum_done",
                duration_seconds=round(time.perf_counter() - t0, 3),
            )

        if maint.blob_sweep_enabled:
            blob_store = state.blob_store
            on_disk = list(
                await asyncio.to_thread(
                    lambda: list(blob_store.iter_digests())
                )
            )
            referenced = set(
                await run_in_sqlite(repo.all_content_hashes)
            )
            orphans = [d for d in on_disk if d not in referenced]
            for digest in orphans:
                await asyncio.to_thread(blob_store.delete, digest)
            if orphans:
                log.info("blob_sweep", removed=len(orphans))
