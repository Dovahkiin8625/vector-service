"""``/v1/jobs/*`` — submit and observe background ingest jobs.

Document ingestion is fully asynchronous:

- ``POST /v1/jobs/ingest`` validates and spools the upload, then
  returns ``202`` with a ``job_id`` — no parse/embed work happens in
  the request;
- ``GET /v1/jobs`` / ``GET /v1/jobs/{id}`` return persisted job
  state (stage, progress, attempts, result, error);
- ``GET /v1/jobs/{id}/events`` streams that same state over SSE,
  pushing a frame on every change until the job reaches a terminal
  state.

The in-process worker (``vector_service.jobs.worker``) claims queued
jobs and runs the shared pipeline in :mod:`vector_service.api.ingest`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from vector_service.api.ingest import _prepare_ingest
from vector_service.core.threadpools import run_in_sqlite
from vector_service.corpus import JOB_TERMINAL
from vector_service.schemas.jobs import (
    JobListResponse,
    JobStatus,
    JobSubmitResponse,
)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post(
    "/ingest",
    response_model=JobSubmitResponse,
    status_code=202,
    summary="Submit a document ingest job",
    description=(
        "Same multipart contract as the old synchronous ingest route "
        "(file plus database/collection/chunk_size/chunk_overlap/"
        "embed_model/metadata/profile/strategy/chunk_options/"
        "add_context). Runs every pre-flight check and "
        "spools the raw upload, then returns immediately with the "
        "queued ``job_id``. The embedder does NOT need to be loaded "
        "yet — embedder state is checked when the worker executes the "
        "job. Poll ``GET /v1/jobs/{job_id}`` for stage progress and "
        "the resulting ``doc_id``."
    ),
)
async def submit_ingest_job(
    request: Request,
    file: UploadFile,
    database: str = Form("default"),
    collection: str = Form("ingest"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(75),
    embed_model: str = Form("bge-m3"),
    metadata: str = Form("{}"),
    profile: str = Form("auto"),
    strategy: str = Form("recursive"),
    chunk_options: str = Form("{}"),
    add_context: bool = Form(False),
    add_summary: bool = Form(False),
) -> JobSubmitResponse:
    """Validate, spool the upload, and enqueue."""
    settings = request.app.state.settings
    embedder = getattr(request.app.state, "embedder", None)
    # Every validation except embedder-state runs now; the model may be
    # loaded by the time the worker picks the job up.
    prepared = await _prepare_ingest(
        file=file,
        settings=settings,
        embedder=embedder,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        profile=profile,
        strategy=strategy,
        chunk_options=chunk_options,
        add_context=add_context,
        add_summary=add_summary,
        check_embedder=False,
    )

    job_id = uuid.uuid4().hex
    doc_id = str(uuid.uuid4())

    # Spool the raw bytes BEFORE the row exists so a claimed job never
    # points at a missing file.
    spool_dir = Path(settings.jobs.spool_dir) / job_id
    spool_dir.mkdir(parents=True, exist_ok=False)
    spool_path = spool_dir / "upload"
    try:
        spool_path.write_bytes(prepared.data)
        params = {
            "database": database,
            "collection": collection,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embed_model": embed_model,
            "profile": profile,
            "strategy": strategy,
            "chunk_options": prepared.chunk_options,
            "add_context": add_context,
            "add_summary": add_summary,
            "metadata": prepared.extra_metadata,
            "filename": prepared.filename,
            "mime": prepared.mime,
        }
        await run_in_sqlite(
            request.app.state.corpus.create_job,
            job_id,
            doc_id=doc_id,
            database=database,
            collection=collection,
            filename=prepared.filename,
            mime=prepared.mime,
            params_json=json.dumps(params, ensure_ascii=False),
            spool_path=str(spool_path),
            max_attempts=settings.jobs.max_attempts,
        )
    except Exception:
        # Don't leave an orphan spool if the row didn't land.
        shutil.rmtree(spool_dir, ignore_errors=True)
        raise

    # Nudge the worker (Step 2); absent in tests and while disabled.
    worker = getattr(request.app.state, "job_worker", None)
    if worker is not None and settings.jobs.wake_on_submit:
        worker.wake()

    return JobSubmitResponse(job_id=job_id)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List ingest jobs",
    description="Jobs in reverse submit order; filter by ``status``.",
)
async def list_jobs(
    request: Request,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> JobListResponse:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    rows, total = await run_in_sqlite(
        request.app.state.corpus.list_jobs,
        status=status,
        limit=limit,
        offset=offset,
    )
    return JobListResponse(items=[JobStatus.from_row(row) for row in rows], total=total)


@router.get(
    "/{job_id}",
    response_model=JobStatus,
    summary="Get one ingest job",
    description="Full job state including stage, progress, result and error.",
)
async def get_job(request: Request, job_id: str) -> JobStatus:
    row = await run_in_sqlite(request.app.state.corpus.get_job, job_id)
    if row is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "job_not_found",
                    "message": f"no ingest job {job_id!r}",
                }
            },
        )
    return JobStatus.from_row(row)


# ---- SSE stream --------------------------------------------------------

#: SSE event name for a job-state frame; payload is a JobStatus JSON.
JOB_EVENT = "job"


def _sse_frame(payload: str) -> str:
    """Wrap one JSON payload as an SSE ``job`` event frame."""
    return f"event: {JOB_EVENT}\ndata: {payload}\n\n"


def _row_signature(row: dict) -> tuple:
    """Values whose change warrants a new SSE frame.

    ``updated_ts`` moves on every stage mark / requeue / terminal write,
    so stage transitions are covered even though ``status`` is the raw
    lifecycle value.
    """
    return (
        row["status"],
        row["progress_current"],
        row["progress_total"],
        row["attempts"],
        row["cancel_requested"],
        row["chunk_count"],
        row["page_count"],
        row["tokens_used"],
        row["error_code"],
        row["error_message"],
        row["finished_ts"],
        row["updated_ts"],
    )


@router.get(
    "/{job_id}/events",
    summary="Stream job progress (SSE)",
    description=(
        "Server-sent events stream of full job-state snapshots. A "
        "``job`` event is sent immediately with the current state and "
        "then whenever the job changes — stage transition, progress "
        "tick, retry requeue, cancellation, terminal result. Comment "
        "frames keep the connection alive and a periodic resync tick "
        "guarantees a missed nudge can never leave the view stale. The "
        "stream closes right after the terminal-state frame."
    ),
)
async def stream_job_events(request: Request, job_id: str) -> StreamingResponse:
    """Push job-state frames until the job is terminal."""
    corpus = request.app.state.corpus
    row = await run_in_sqlite(corpus.get_job, job_id)
    if row is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "job_not_found",
                    "message": f"no ingest job {job_id!r}",
                }
            },
        )

    settings = request.app.state.settings
    bus = getattr(request.app.state, "job_bus", None)
    heartbeat = settings.jobs.heartbeat_seconds
    resync = settings.jobs.sse_resync_seconds

    async def event_stream() -> AsyncIterator[str]:
        queue = bus.subscribe(job_id) if bus is not None else None
        last_sig = _row_signature(row)
        try:
            # Initial snapshot — the job may already be terminal.
            yield _sse_frame(JobStatus.from_row(row).model_dump_json())
            if row["status"] in JOB_TERMINAL:
                return

            while True:
                timers: dict[asyncio.Task, str] = {
                    asyncio.create_task(asyncio.sleep(heartbeat)): "heartbeat",
                    asyncio.create_task(asyncio.sleep(resync)): "resync",
                }
                nudge_task: asyncio.Task | None = None
                if queue is not None:
                    nudge_task = asyncio.create_task(queue.get())
                    timers[nudge_task] = "nudge"
                done, pending = await asyncio.wait(
                    timers, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                # Let cancellations settle so no task is destroyed pending.
                await asyncio.gather(*pending, return_exceptions=True)
                kinds = {timers[task] for task in done}

                # Coalesce any further nudges that piled up behind it.
                if "nudge" in kinds and queue is not None:
                    while not queue.empty():
                        queue.get_nowait()

                if "nudge" in kinds or "resync" in kinds:
                    fresh = await run_in_sqlite(corpus.get_job, job_id)
                    if fresh is None:
                        # Row removed out of band — nothing left to stream.
                        return
                    sig = _row_signature(fresh)
                    if sig != last_sig:
                        last_sig = sig
                        yield _sse_frame(JobStatus.from_row(fresh).model_dump_json())
                    if fresh["status"] in JOB_TERMINAL:
                        return

                if "heartbeat" in kinds:
                    # Comment frame: keeps proxies from idling the
                    # connection without reaching client handlers.
                    yield ": keepalive\n\n"
        finally:
            if bus is not None and queue is not None:
                bus.unsubscribe(job_id, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatus,
    summary="Request cancellation of an ingest job",
    description=(
        "Sets the cancel flag; the worker observes it at the next stage "
        "boundary, rolls the partial document back, and marks the job "
        "``cancelled``. A job already in a terminal state returns 409."
    ),
)
async def cancel_job(request: Request, job_id: str) -> JobStatus:
    corpus = request.app.state.corpus
    row = await run_in_sqlite(corpus.get_job, job_id)
    if row is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "job_not_found",
                    "message": f"no ingest job {job_id!r}",
                }
            },
        )
    if row["status"] in JOB_TERMINAL:
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "job_already_terminal",
                    "message": (
                        f"ingest job {job_id!r} is already "
                        f"{row['status']}; nothing to cancel"
                    ),
                    "status": row["status"],
                }
            },
        )
    # The row could reach a terminal state between the check above and
    # the UPDATE; request_cancel reports that race.
    if not await run_in_sqlite(corpus.request_cancel, job_id):
        fresh = await run_in_sqlite(corpus.get_job, job_id)
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "job_already_terminal",
                    "message": (
                        f"ingest job {job_id!r} finished before cancellation "
                        f"took effect"
                    ),
                    "status": fresh["status"] if fresh else None,
                }
            },
        )

    # Let live progress streams show cancel_requested immediately,
    # before the worker reaches its next stage boundary.
    bus = getattr(request.app.state, "job_bus", None)
    if bus is not None:
        bus.publish(job_id, "cancel")
    return JobStatus.from_row(
        await run_in_sqlite(corpus.get_job, job_id)
    )
