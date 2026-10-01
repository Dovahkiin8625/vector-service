"""Pydantic schemas for the GraphRAG build / status API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GraphBuildRequest(BaseModel):
    """Options for a graph build (all graph rows are replaced on success)."""

    #: Optional entity-type hint appended to the extraction prompt.
    entity_types: list[str] = Field(default_factory=list)

    #: Label-propagation round cap.
    community_iterations: int = Field(20, ge=1, le=100)

    #: Communities smaller than this are not summarized/indexed (members
    #: still appear via the entity index).
    min_community_size: int = Field(2, ge=1, le=10)

    #: Persist extracted claims; false discards them after extraction.
    include_claims: bool = True

    #: Leaf chunks processed per extraction batch (progress tick, cancel
    #: gate granularity).
    batch_size: int = Field(16, ge=1, le=256)


class GraphBuildSubmitResponse(BaseModel):
    """Immediate acknowledgement; the worker runs the build."""

    job_id: str
    status: Literal["queued"] = "queued"
    entity_collection: str
    community_collection: str


class CommunityInfo(BaseModel):
    """One detected community in the graph status response."""

    community_id: str
    community_index: int
    size: int
    summary: str


class GraphStatusResponse(BaseModel):
    """Counts and community listing of one collection's graph."""

    database: str
    collection: str
    entity_collection: str
    community_collection: str
    entities_count: int = Field(ge=0)
    edges_count: int = Field(ge=0)
    claims_count: int = Field(ge=0)
    communities_count: int = Field(ge=0)
    communities: list[CommunityInfo]


class GraphDeleteResponse(BaseModel):
    """Result of deleting the derived graph rows and collections."""

    deleted: bool
    entity_collection: str
    community_collection: str
