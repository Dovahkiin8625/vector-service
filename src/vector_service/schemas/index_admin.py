"""Pydantic schemas for index rebuild / consistency administration.

These back the routes in :mod:`vector_service.api.rebuild`:

- ``POST .../reindex`` — submit a rebuild job (new physical collection);
- ``POST .../reindex/promote`` — promote the canary, retire the old index;
- ``POST .../consistency`` — compare corpus leaves with physical rows.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):
    """Options for a rebuild job.

    ``embed_model`` defaults to the embedder loaded when the worker
    executes the job. ``canary_percent`` selects how much query traffic
    the rebuilt index receives before promotion; ``0`` keeps it a
    pin-only shadow.
    """

    embed_model: str | None = Field(
        default=None,
        description="Target dense model id; defaults to the currently loaded embedder.",
    )
    canary_percent: int = Field(default=0, ge=0, le=100)
    batch_size: int = Field(default=64, ge=1, le=512)


class ReindexSubmitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    index_ref: str = Field(description="Physical collection being built.")


class PromoteResponse(BaseModel):
    active_ref: str
    retired_ref: str
    model: str


class ConsistencyRequest(BaseModel):
    repair: bool = Field(
        default=False,
        description="Delete orphan rows and rederive missing rows when true.",
    )


class IndexConsistencyItem(BaseModel):
    """One physical index compared against the corpus leaf set."""

    ref: str
    canary: bool
    milvus_rows: int
    missing_in_milvus: list[str]
    orphans_in_milvus: list[str]
    ok: bool


class ConsistencyResponse(BaseModel):
    database: str
    collection: str
    corpus_leaves: int
    indexes: list[IndexConsistencyItem]
    ok: bool
    repaired: bool
