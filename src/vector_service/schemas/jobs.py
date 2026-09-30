"""Pydantic schemas for the background ingest-job API.

The service no longer serves a synchronous ingest endpoint; documents
enter through ``POST /v1/jobs/ingest`` and are observed through
``GET /v1/jobs`` / ``GET /v1/jobs/{id}``. ``JobStatus`` is assembled
straight from an ``ingest_jobs`` row via :meth:`JobStatus.from_row`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from vector_service.corpus.models import JOB_STAGES


class JobSubmitResponse(BaseModel):
    """Immediate acknowledgement of ``POST /v1/jobs/ingest``.

    The raw upload is spooled and the row is ``queued``; processing
    happens outside the request. Poll ``GET /v1/jobs/{job_id}``.
    """

    job_id: str = Field(description="Opaque job id used on every jobs route.")
    status: Literal["queued"] = "queued"


class JobProgress(BaseModel):
    """Stage progress counters. Null until the running stage reports."""

    current: int | None = Field(default=None, examples=[3])
    total: int | None = Field(default=None, examples=[12])


class JobError(BaseModel):
    """Terminal error recorded on a failed job."""

    code: str
    message: str


class JobStatus(BaseModel):
    """Full state of one ingest job.

    ``stage`` is the active stage (``parse``/``chunk``/``embed``/
    ``upsert``) while the job runs, ``None`` when queued or terminal;
    ``status`` always carries the raw lifecycle value.
    """

    job_id: str
    doc_id: str | None = Field(
        description="Document id assigned at submit time; populated before any work runs.",
    )
    status: str
    stage: str | None
    database: str
    collection: str
    filename: str | None
    mime: str | None
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    cancel_requested: bool
    progress: JobProgress
    chunk_count: int = Field(ge=0)
    page_count: int | None
    tokens_used: int = Field(ge=0)
    error: JobError | None
    created_ts: float
    updated_ts: float
    finished_ts: float | None

    @classmethod
    def from_row(cls, row: dict) -> JobStatus:
        status = row["status"]
        error_code = row["error_code"]
        return cls(
            job_id=row["job_id"],
            doc_id=row["doc_id"],
            status=status,
            stage=status if status in JOB_STAGES else None,
            database=row["database"],
            collection=row["collection"],
            filename=row["filename"],
            mime=row["mime"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            cancel_requested=bool(row["cancel_requested"]),
            progress=JobProgress(
                current=row["progress_current"], total=row["progress_total"]
            ),
            chunk_count=row["chunk_count"],
            page_count=row["page_count"],
            tokens_used=row["tokens_used"],
            error=JobError(code=error_code, message=row["error_message"])
            if error_code is not None
            else None,
            created_ts=row["created_ts"],
            updated_ts=row["updated_ts"],
            finished_ts=row["finished_ts"],
        )


class JobListResponse(BaseModel):
    """Paginated job listing, newest submit first."""

    items: list[JobStatus]
    total: int = Field(ge=0)
