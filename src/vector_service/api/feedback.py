"""Retrieval feedback API: record 👍/👎/click/adopt, list for analysis."""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request

from vector_service.core.threadpools import run_in_sqlite
from vector_service.schemas.feedback import (
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackSubmitResponse,
    record_out,
)

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackSubmitResponse)
async def submit_feedback(
    body: FeedbackRequest, request: Request
) -> FeedbackSubmitResponse:
    repo = request.app.state.corpus
    feedback_id = f"fb_{uuid.uuid4().hex}"
    pipeline_json = json.dumps(body.pipeline, ensure_ascii=False, default=str)
    await run_in_sqlite(
        repo.add_feedback,
        feedback_id,
        database=body.database,
        collection=body.collection,
        query=body.query,
        chunk_id=body.chunk_id,
        kind=body.kind,
        pipeline_json=pipeline_json,
        comment=body.comment,
    )
    return FeedbackSubmitResponse(feedback_id=feedback_id)


@router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    request: Request,
    database: str | None = None,
    collection: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> FeedbackListResponse:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    if kind is not None and kind.strip().lower() not in ("up", "down",
                                                         "click", "adopt"):
        kind = None
    rows, total = await run_in_sqlite(
        request.app.state.corpus.list_feedback,
        database=database,
        collection=collection,
        kind=kind.strip().lower() if kind is not None else None,
        limit=limit,
        offset=offset,
    )
    return FeedbackListResponse(
        items=[record_out(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
