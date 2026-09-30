"""Startup recovery for jobs a dead process left mid-stage.

An ungraceful exit (kill -9, crash, power loss) can leave job rows in
``parsing`` / ``chunking`` / ``embedding`` / ``upserting`` — the worker
only writes terminal states itself. :func:`recover_interrupted` runs in
the lifespan *before* the worker starts claiming and resets each row:

- spool bytes present and no cancel pending → half-written state is
  cleaned, the row goes back to ``queued`` with ``attempts + 1`` (or is
  failed ``interrupted`` if that spends ``max_attempts``);
- cancel flag set from the previous process → ``cancelled``, spool
  discarded;
- spool missing (deleted, or a pre-async inline job) → ``failed`` with
  ``interrupted``; the upload can no longer be reconstructed.

Half-written state is cleaned for every row, even the unrecoverable
ones: corpus delete, thin-index delete by ``doc_id`` (best-effort),
artifact discard.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from vector_service.api.ingest import _delete_by_doc_id, _discard_artifacts
from vector_service.core.logging import get_logger
from vector_service.core.threadpools import run_in_sqlite, run_in_store
from vector_service.jobs.worker import IngestWorker

if TYPE_CHECKING:
    from fastapi import FastAPI

#: Terminal error code for jobs that cannot continue after a restart.
JOB_INTERRUPTED = "interrupted"

log = get_logger(__name__)


async def recover_interrupted(app: FastAPI) -> int:
    """Reset every stage-status job left by an earlier process.

    Returns the number of jobs handled. Called once during startup,
    before the worker starts.
    """
    corpus = app.state.corpus
    rows = await run_in_sqlite(corpus.find_interrupted)
    for row in rows:
        # §1: ingest rows only. Later sections dispatch rebuild
        # (§3) and eval_run / gate_check (§5) on ``row["job_type"]``.
        await _recover_one(app, row)
    if rows:
        log.info("job_recovery", recovered=len(rows))
    return len(rows)


async def _recover_one(app: FastAPI, row: dict) -> None:
    state = app.state
    corpus = state.corpus
    job_id = row["job_id"]
    doc_id = row["doc_id"]

    spool_path = row["spool_path"]
    spool_ok = bool(spool_path) and Path(spool_path).is_file()

    # Always undo whatever the dead process committed: corpus rows,
    # thin-index rows, parser artifacts. Each layer is best-effort —
    # a failed cleanup must not leave the job stuck in a stage status.
    if doc_id is not None:
        try:
            await run_in_sqlite(corpus.delete_document, doc_id)
        except Exception as e:  # noqa: BLE001 — recovery continues
            log.warning(
                "job_recovery_corpus_cleanup_failed",
                job_id=job_id,
                doc_id=doc_id,
                error=str(e),
            )
        try:
            await run_in_store(
                _delete_by_doc_id,
                state.store,
                row["database"],
                row["collection"],
                doc_id,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "job_recovery_index_cleanup_failed",
                job_id=job_id,
                doc_id=doc_id,
                error=str(e),
            )
        await asyncio.to_thread(_discard_artifacts, doc_id)

    if not spool_ok:
        # Upload bytes are gone (or never existed for a pre-async inline
        # job) — nothing left to execute.
        await run_in_sqlite(
            corpus.fail_job,
            job_id,
            error_code=JOB_INTERRUPTED,
            error_message=(
                "job was interrupted by a process restart and the upload "
                "spool is no longer available"
            ),
        )
        await IngestWorker._discard_spool(row)
        log.warning("job_recovery_failed_missing_spool", job_id=job_id)
        return

    if row["cancel_requested"]:
        # Honor a cancellation the dead process never got to observe.
        await run_in_sqlite(corpus.mark_cancelled, job_id)
        await IngestWorker._discard_spool(row)
        log.info("job_recovery_cancelled", job_id=job_id)
        return

    attempts = int(row["attempts"]) + 1
    if attempts >= int(row["max_attempts"]):
        await run_in_sqlite(
            corpus.fail_job,
            job_id,
            error_code=JOB_INTERRUPTED,
            error_message=(
                f"job reached max_attempts={row['max_attempts']} after a "
                "process restart"
            ),
            attempts=attempts,
        )
        await IngestWorker._discard_spool(row)
        log.warning(
            "job_recovery_failed_attempts_exhausted",
            job_id=job_id,
            attempts=attempts,
        )
        return

    await run_in_sqlite(corpus.requeue_job, job_id, attempts=attempts)
    log.info("job_recovery_requeued", job_id=job_id, attempts=attempts)
