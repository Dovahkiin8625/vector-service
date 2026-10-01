"""Shared dataclasses for the retrieval pipeline.

The pipeline turns one user query into per-channel recall legs
(dense ANN and BM25 full-text), fuses them, optionally diversifies
and reranks them. Every stage communicates through the dataclasses
defined here so the orchestrator, channels, and the API layer share
one vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecallSpec:
    """One query going into a recall leg.

    ``vector`` is set when the leg must skip the embedder (HyDE
    pre-mixes the original-query and hypothetical-doc vectors).
    ``hypothetical`` keeps the HyDE document for tracing.
    """

    query: str
    vector: list[float] | None = None
    hypothetical: str | None = None


@dataclass
class MetaPredicate:
    """One inline ``field op value`` condition pulled out of the query."""

    field: str
    op: str
    value: str


@dataclass
class ChannelHit:
    """One raw hit inside a single channel run."""

    chunk_id: str
    score: float
    rank: int
    fields: dict = field(default_factory=dict)


@dataclass
class ChannelRun:
    """The full ranked output of one (channel, query) leg."""

    channel: str
    query: str
    hits: list[ChannelHit] = field(default_factory=list)


@dataclass
class StageTrace:
    """Timing + key parameters for one pipeline stage."""

    stage: str
    duration_ms: int
    detail: dict = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    """The post-rewrite query plan consumed by the recall stage."""

    original_query: str
    dense_specs: list[RecallSpec]
    lexical_queries: list[str]
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """One fused (and optionally reranked) chunk in the final answer."""

    chunk_id: str
    fields: dict
    fusion_score: float
    matched_channels: list[str]
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    """Everything the pipeline observed: answer + plan + raw runs + timing."""

    query: str
    chunks: list[RetrievedChunk]
    plan: RetrievalPlan
    channel_runs: list[ChannelRun]
    traces: list[StageTrace]
    route: Any = None
    #: Pre-rerank order (candidate pool) when reranking ran; lets eval
    #: score the fused order against the final, reranked order.
    rerank_input_ids: list[str] | None = None
