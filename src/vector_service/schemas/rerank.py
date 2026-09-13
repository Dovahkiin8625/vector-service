"""Pydantic schemas for the /v1/rerank endpoints."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RerankRequest(BaseModel):
    """Request body for ``POST /v1/rerank``.

    Schema-level validation is intentionally minimal — length limits
    and ``top_n`` bounds are checked in the route against settings,
    so we can return the canonical error envelope with explicit
    ``code`` strings.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = Field(None, ge=1)
    model: str | None = Field(
        None,
        description="Backend name; defaults to settings.reranker.backend.",
    )


class RerankResultItem(BaseModel):
    """One ranked document, identified by its index in the request."""

    index: int = Field(..., ge=0)
    score: float


class RerankResponse(BaseModel):
    """Response body for ``POST /v1/rerank``.

    ``results`` is sorted by ``score`` descending; length is
    ``min(top_n, len(documents))``.
    """

    model: str
    results: list[RerankResultItem]
    request_id: str