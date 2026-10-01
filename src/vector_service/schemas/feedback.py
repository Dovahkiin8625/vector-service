"""Pydantic schemas for the retrieval-feedback API."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

FEEDBACK_KINDS = ("up", "down", "click", "adopt")


class FeedbackRequest(BaseModel):
    """One feedback event for a retrieval answer or chunk.

    ``chunk_id`` is null for answer-level (👍/👎) feedback. ``pipeline``
    snapshots the pipeline version the feedback refers to (channels,
    fusion, rewrite, rerank, models, index_ref, routing) — the client
    echoes the context of the retrieval it reacts to.
    """

    database: str = Field("default", min_length=1)
    collection: str = Field("ingest", min_length=1)
    query: str = Field(..., min_length=1)
    chunk_id: str | None = None
    kind: str
    comment: str | None = None
    pipeline: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _kind_known(cls, value):
        value = value.strip().lower()
        if value not in FEEDBACK_KINDS:
            raise ValueError(
                f"kind must be one of {FEEDBACK_KINDS}"
            )
        return value


class FeedbackRecordOut(BaseModel):
    feedback_id: str
    database: str
    collection: str
    query: str
    chunk_id: str | None
    kind: str
    comment: str | None
    pipeline: dict[str, Any]
    created_ts: float


class FeedbackSubmitResponse(BaseModel):
    feedback_id: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackRecordOut]
    total: int
    limit: int
    offset: int


def record_out(row: dict[str, Any]) -> FeedbackRecordOut:
    """Map a repository row onto the API model, parsing the snapshot."""
    try:
        pipeline = json.loads(row.get("pipeline_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pipeline = {}
    return FeedbackRecordOut(
        feedback_id=row["feedback_id"],
        database=row["database"],
        collection=row["collection"],
        query=row["query"],
        chunk_id=row["chunk_id"],
        kind=row["kind"],
        comment=row["comment"],
        pipeline=pipeline if isinstance(pipeline, dict) else {},
        created_ts=float(row["created_ts"]),
    )
