"""In-process worker that runs queued ingest jobs.

Single process (``workers=1``), no external queue: the worker task
claims the oldest backoff-ready queued row from the corpus, rebuilds
the validated inputs from the spooled upload and ``params_json``, and
runs the shared pipeline.

- **Retry**: pipeline failures surface as :class:`HTTPException`.
  4xx are unrecoverable parameter/state errors → terminal ``failed``;
  5xx (parser/store/embedder unavailable) requeue with exponential
  backoff (``not_before_ts``) until the row's ``max_attempts`` is
  spent.
- **Cancel**: observed at stage boundaries inside the pipeline, which
  rolls back, marks ``cancelled`` and raises :class:`JobCancelled`.
- **Shutdown**: the worker task is cancelled rather than left
  running; blocking work in progress is not force-killed. Startup
  recovery on the next process handles the interrupted row.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

from vector_service.api.ingest import (
    JobCancelled,
    PreparedIngest,
    _run_ingest_pipeline,
)
from vector_service.api.rebuild import run_rebuild_pipeline
from vector_service.core.logging import get_logger
from vector_service.core.threadpools import run_in_sqlite

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


class IngestWorker:
    """Claims queued jobs and runs them, one at a time."""

    def __init__(self, app: FastAPI) -> None:
        self._app = app
        self._wake_event: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._stopped = False

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Launch the worker loop as a background task."""
        if self._task is not None:
            raise RuntimeError("job worker already started")
        self._wake_event = asyncio.Event()
        self._stopped = False
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Stop claiming and cancel an in-flight job.

        A job interrupted here is picked up by startup recovery on the
        next process; blocking worker threads are not force-killed.
        """
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
        """Interrupt the idle wait (new submission, explicit nudge)."""
        if self._wake_event is not None:
            self._wake_event.set()

    def _publish(self, job_id: str, kind: str) -> None:
        """Nudge SSE subscribers after a state change.

        No-op when no bus is attached (unit tests build the worker
        without one). Called from coroutines on the event loop, the same
        thread the bus is mutated from.
        """
        bus = getattr(self._app.state, "job_bus", None)
        if bus is not None:
            bus.publish(job_id, kind)

    # ---- main loop -----------------------------------------------------

    async def run(self) -> None:
        state = self._app.state
        interval = state.settings.jobs.poll_interval_seconds
        while not self._stopped:
            try:
                claimed = await run_in_sqlite(state.corpus.claim_next_queued)
            except Exception:
                log.exception("job_claim_failed")
                await self._wait_idle(interval)
                continue
            if claimed is None:
                await self._wait_idle(interval)
                continue
            try:
                await self._execute(claimed)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job_worker_execute_failed", job_id=claimed["job_id"])

    async def _wait_idle(self, interval: float) -> None:
        event = self._wake_event
        if event is None:
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=interval)
        except TimeoutError:
            pass
        event.clear()

    # ---- per-job execution --------------------------------------------

    async def _execute(self, row: dict) -> None:
        # Dispatch on the row's job_type: ``ingest`` rebuilds the
        # validated upload from the spool, ``rebuild`` derives
        # everything from the existing corpus. Later sections add
        # graph / eval / gate-check dispatch here.
        if row.get("job_type") == "rebuild":
            await self._execute_rebuild(row)
        else:
            await self._execute_ingest(row)

    async def _execute_ingest(self, row: dict) -> None:
        state = self._app.state
        corpus = state.corpus
        job_id = row["job_id"]

        # Rebuild the validated inputs from the spooled upload + params.
        try:
            params = json.loads(row["params_json"])
            # Spool read off the event loop (filesystem, default executor).
            data = await asyncio.to_thread(
                Path(row["spool_path"]).read_bytes
            )
            prepared = PreparedIngest(
                data=data,
                mime=params["mime"],
                extra_metadata=params.get("metadata", {}),
                filename=params.get("filename"),
                profile=params.get("profile", "auto"),
                strategy=params.get("strategy", "recursive"),
                chunk_options=params.get("chunk_options", {}),
                add_context=params.get("add_context", False),
                add_summary=params.get("add_summary", False),
            )
        except Exception as e:  # noqa: BLE001 — self-written row; treat any unreadable state as corrupted
            # Row and spool were written by the service itself: anything
            # unreadable here is unrecoverable, not worth a retry.
            log.warning("job_corrupted", job_id=job_id, error=str(e))
            await run_in_sqlite(
                corpus.fail_job,
                job_id,
                error_code="job_corrupted",
                error_message=str(e),
            )
            self._publish(job_id, "failed")
            await self._discard_spool(row)
            return

        settings = state.settings
        embedder = getattr(state, "embedder", None)
        if embedder is None:
            await self._handle_failure(
                row,
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": (
                                "text embedder is not loaded; call "
                                f"POST /v1/models/{params['embed_model']}/load first"
                            ),
                            "model": params["embed_model"],
                        }
                    },
                ),
            )
            return
        if embedder.model_name != params["embed_model"]:
            await self._handle_failure(
                row,
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": (
                                f"a different embedder is loaded: expected "
                                f"{params['embed_model']!r}, got "
                                f"{embedder.model_name!r}"
                            ),
                            "model": params["embed_model"],
                            "loaded": embedder.model_name,
                        }
                    },
                ),
            )
            return

        async def emit(event: dict) -> None:
            event_type = event.get("type")
            if event_type == "progress":
                await run_in_sqlite(
                    corpus.set_job_progress,
                    job_id,
                    event["page"],
                    event["total"],
                )
                self._publish(job_id, "progress")
            elif event_type == "stage":
                # The pipeline persisted the stage before emitting, so
                # subscribers that reload on this nudge see the new one.
                self._publish(job_id, "stage")

        async def should_cancel() -> bool:
            fresh = await run_in_sqlite(corpus.get_job, job_id)
            return fresh is not None and bool(fresh["cancel_requested"])

        try:
            await _run_ingest_pipeline(
                prepared=prepared,
                settings=settings,
                store=state.store,
                repo=corpus,
                bm25=state.bm25,
                embedder=embedder,
                database=params["database"],
                collection=params["collection"],
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
                embed_model=params["embed_model"],
                inference_timeout_seconds=settings.inference_timeout_seconds,
                emit=emit,
                job_id=job_id,
                doc_id=row["doc_id"],
                should_cancel=should_cancel,
            )
        except JobCancelled:
            # Rollback + cancelled state were already written inside
            # the pipeline's stage-boundary gate.
            self._publish(job_id, "cancelled")
            await self._discard_spool(row)
            return
        except HTTPException as exc:
            await self._handle_failure(row, exc)
            return
        # Pipeline wrote the terminal row itself (finish_job). Finish
        # spool handling before publishing: successful original bytes
        # move into the content-addressed store (a zero-chunk doc never
        # created a document row, so its spool is discarded), so a
        # subscriber that wakes on the done frame already sees the end
        # state instead of racing the cleanup.
        doc = await run_in_sqlite(corpus.get_document, row["doc_id"])
        spool_path = row.get("spool_path")
        if doc is not None and spool_path:
            await asyncio.to_thread(
                state.blob_store.promote, spool_path, doc["content_hash"]
            )
        else:
            await self._discard_spool(row)
        self._publish(job_id, "done")

    async def _execute_rebuild(self, row: dict) -> None:
        """Run a ``rebuild`` job: rederive one physical index from corpus."""
        state = self._app.state
        corpus = state.corpus
        job_id = row["job_id"]

        try:
            params = json.loads(row["params_json"])
        except Exception as e:  # noqa: BLE001 — self-written row; treat unreadable state as corrupted
            log.warning("job_corrupted", job_id=job_id, error=str(e))
            await run_in_sqlite(
                corpus.fail_job,
                job_id,
                error_code="job_corrupted",
                error_message=str(e),
            )
            self._publish(job_id, "failed")
            return

        embedder = getattr(state, "embedder", None)
        if embedder is None:
            await self._handle_failure(
                row,
                HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": (
                                "text embedder is not loaded; vectors cannot "
                                "be recomputed"
                            ),
                        }
                    },
                ),
            )
            return

        async def emit(event: dict) -> None:
            event_type = event.get("type")
            if event_type == "progress":
                await run_in_sqlite(
                    corpus.set_job_progress,
                    job_id,
                    event["page"],
                    event["total"],
                )
                self._publish(job_id, "progress")
            elif event_type == "stage":
                self._publish(job_id, "stage")

        async def should_cancel() -> bool:
            fresh = await run_in_sqlite(corpus.get_job, job_id)
            return fresh is not None and bool(fresh["cancel_requested"])

        try:
            await run_rebuild_pipeline(
                settings=state.settings,
                store=state.store,
                repo=corpus,
                bm25=state.bm25,
                embedder=embedder,
                database=row["database"],
                logical_collection=row["collection"],
                target_ref=params["target_ref"],
                batch_size=params["batch_size"],
                canary_percent=params["canary_percent"],
                embed_model=params.get("target_embed_model") or embedder.model_name,
                inference_timeout_seconds=state.settings.inference_timeout_seconds,
                emit=emit,
                job_id=job_id,
                should_cancel=should_cancel,
            )
        except JobCancelled:
            # Cleanup + cancelled state were written at the cancel gate.
            self._publish(job_id, "cancelled")
            return
        except HTTPException as exc:
            await self._handle_failure(row, exc)
            return
        self._publish(job_id, "done")

    async def _handle_failure(self, row: dict, exc: HTTPException) -> None:
        """Requeue a retryable failure; otherwise mark the job failed."""
        state = self._app.state
        corpus = state.corpus
        job_id = row["job_id"]
        code, message = self._unpack_http_error(exc)
        attempts = int(row["attempts"]) + 1
        max_attempts = int(row["max_attempts"])
        if exc.status_code >= 500 and attempts < max_attempts:
            jobs_cfg = state.settings.jobs
            delay = min(
                jobs_cfg.retry_backoff_base_seconds * 2 ** (attempts - 1),
                jobs_cfg.retry_backoff_max_seconds,
            )
            await run_in_sqlite(
                corpus.requeue_job,
                job_id,
                attempts=attempts,
                not_before=time.time() + delay,
            )
            log.warning(
                "job_requeued",
                job_id=job_id,
                attempts=attempts,
                code=code,
                delay=delay,
            )
            self._publish(job_id, "requeued")
        else:
            await run_in_sqlite(
                corpus.fail_job,
                job_id,
                error_code=code,
                error_message=message,
                attempts=attempts,
            )
            log.warning("job_failed", job_id=job_id, attempts=attempts, code=code)
            self._publish(job_id, "failed")
            await self._discard_spool(row)

    @staticmethod
    def _unpack_http_error(exc: HTTPException) -> tuple[str, str]:
        """Pull ``(code, message)`` out of a raised error envelope."""
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            error = detail["error"]
            return error.get("code", "ingest_failed"), str(
                error.get("message", str(exc))
            )
        return "ingest_failed", str(detail)

    @staticmethod
    async def _discard_spool(row: dict) -> None:
        """Remove the per-job spool directory after a terminal state."""
        spool_path = row.get("spool_path")
        if not spool_path:
            return

        def _rmtree() -> None:
            # The directory only ever holds this job's upload; drop whole.
            shutil.rmtree(Path(spool_path).parent, ignore_errors=True)

        # Filesystem cleanup — default executor.
        await asyncio.to_thread(_rmtree)
