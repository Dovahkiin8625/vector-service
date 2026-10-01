"""Pydantic schemas for the evaluation-set API."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from vector_service.retrieval.expand import CONTEXT_LEVELS
from vector_service.schemas.retrieval import (
    ChannelsSpec,
    ContextSpec,
    FilterSpec,
    FusionSpec,
    MMRSpec,
    RerankSpec,
    RewriteSpec,
    RoutingSpec,
)

# ---- set management ----


class EvalSetCreate(BaseModel):
    database: str = Field("default", min_length=1)
    collection: str = Field("ingest", min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""


class EvalSetOut(BaseModel):
    set_id: str
    database: str
    collection: str
    name: str
    description: str
    question_count: int = 0
    created_ts: float


class EvalSetListResponse(BaseModel):
    items: list[EvalSetOut]
    total: int
    limit: int
    offset: int


# ---- questions ----


class EvalQuestionIn(BaseModel):
    question: str = Field(..., min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None

    @model_validator(mode="after")
    def _expectations_present(self):
        if not (
            self.expected_chunk_ids
            or self.expected_doc_ids
            or (self.expected_answer and self.expected_answer.strip())
        ):
            raise ValueError(
                "at least one expectation required (expected chunks / docs / answer)"
            )
        return self


class EvalQuestionsCreate(BaseModel):
    questions: list[EvalQuestionIn] = Field(..., min_length=1)


class EvalQuestionOut(BaseModel):
    question_id: str
    question: str
    expected_chunk_ids: list[str]
    expected_doc_ids: list[str]
    expected_answer: str | None
    created_ts: float


class EvalQuestionsResponse(BaseModel):
    set_id: str
    items: list[EvalQuestionOut]


# ---- frozen versions ----


class VersionCreate(BaseModel):
    tag: str = Field(..., min_length=1)


class VersionOut(BaseModel):
    version_id: str
    set_id: str
    tag: str
    question_count: int
    created_ts: float


class VersionListResponse(BaseModel):
    set_id: str
    items: list[VersionOut]


# ---- run template ----


class EvalTemplate(BaseModel):
    """Retrieval knobs applied to every question of a run.

    ``database`` / ``collection`` come from the set (not the template),
    and ``query`` is replaced per question. Everything else mirrors
    :class:`~vector_service.schemas.retrieval.RetrievalRequest`.
    """

    index_ref: str | None = None
    top_k: int = Field(10, ge=1, le=100)
    filter: FilterSpec = Field(default_factory=FilterSpec)
    channels: ChannelsSpec = Field(default_factory=ChannelsSpec)
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    rewrite: RewriteSpec = Field(default_factory=RewriteSpec)
    mmr: MMRSpec = Field(default_factory=MMRSpec)
    rerank: RerankSpec = Field(default_factory=RerankSpec)
    routing: RoutingSpec = Field(default_factory=RoutingSpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    #: Expand recalled leaves before recording run ids; mirrors
    #: :class:`~vector_service.schemas.retrieval.RetrievalRequest`.
    context_level: str = "chunk"

    @field_validator("context_level")
    @classmethod
    def _context_level_known(cls, value):
        if value not in CONTEXT_LEVELS:
            raise ValueError(
                f"context_level must be one of {CONTEXT_LEVELS}"
            )
        return value

    @model_validator(mode="after")
    def _pool_covers_top_k(self):
        if self.rerank.enabled and self.rerank.candidate_pool < self.top_k:
            raise ValueError("rerank.candidate_pool must be >= top_k")
        return self


class EvalRunCreate(BaseModel):
    template: EvalTemplate = Field(default_factory=EvalTemplate)
    #: Generate an answer (LLM) from the retrieved chunks per question.
    include_answer: bool = False
    #: Run the questions frozen in this version instead of the live set.
    version_id: str | None = None


class EvalRunSubmitResponse(BaseModel):
    run_id: str
    job_id: str


# ---- regression gates ----


class GateCreate(BaseModel):
    """Gate configuration: scope + frozen dataset version + thresholds.

    Absolute floors (``min_*``) compare the check's summary directly;
    maximum drops compare it against the baseline run and require
    ``baseline_run_id``. At least one criterion is required.
    """

    database: str = Field("default", min_length=1)
    collection: str = Field("ingest", min_length=1)
    set_id: str
    version_id: str
    template: EvalTemplate = Field(default_factory=EvalTemplate)
    min_recall: float | None = Field(default=None, ge=0, le=1)
    min_mrr: float | None = Field(default=None, ge=0, le=1)
    min_ndcg: float | None = Field(default=None, ge=0, le=1)
    max_recall_drop: float | None = Field(default=None, ge=0, le=1)
    max_mrr_drop: float | None = Field(default=None, ge=0, le=1)
    max_ndcg_drop: float | None = Field(default=None, ge=0, le=1)
    baseline_run_id: str | None = None

    @model_validator(mode="after")
    def _validate_criteria(self):
        floors = (
            self.min_recall, self.min_mrr, self.min_ndcg,
        )
        drops = (
            self.max_recall_drop, self.max_mrr_drop, self.max_ndcg_drop,
        )
        if not any(v is not None for v in (*floors, *drops)):
            raise ValueError(
                "gate requires at least one criterion (min metric or max drop)"
            )
        if any(v is not None for v in drops) and not self.baseline_run_id:
            raise ValueError(
                "max_*_drop criteria require baseline_run_id"
            )
        return self


class GateOut(BaseModel):
    gate_id: str
    database: str
    collection: str
    set_id: str
    version_id: str
    template: EvalTemplate
    min_recall: float | None
    min_mrr: float | None
    min_ndcg: float | None
    max_recall_drop: float | None
    max_mrr_drop: float | None
    max_ndcg_drop: float | None
    baseline_run_id: str | None
    created_ts: float


class GateListResponse(BaseModel):
    items: list[GateOut]
    total: int


class GateCheckSubmitResponse(BaseModel):
    check_id: str
    run_id: str
    job_id: str


class GateCheckOut(BaseModel):
    check_id: str
    gate_id: str
    run_id: str | None
    candidate_ref: str
    status: str
    report: dict[str, Any]
    created_ts: float


class GateCheckListResponse(BaseModel):
    gate_id: str
    items: list[GateCheckOut]


# ---- run results ----


class EvalResultOut(BaseModel):
    question_id: str
    question: str
    chunk_ids: list[str]
    metrics: dict[str, Any]
    answer: str | None = None
    answer_error: str | None = None
    created_ts: float


class RerankComparison(BaseModel):
    """Aggregate pre/post-rerank ranking metrics over reranked questions."""

    questions: int
    mean_recall_pre: float | None
    mean_recall_post: float | None
    mean_mrr_pre: float | None
    mean_mrr_post: float | None
    mean_ndcg_pre: float | None
    mean_ndcg_post: float | None


class ChannelCredit(BaseModel):
    """One channel's contribution to expected-chunk hits.

    ``hit_share`` = final-list hits attributed to the channel / all
    final-list hits; ``raw_recall`` = expected chunks the channel
    surfaced anywhere in its runs / all expected chunks.
    """

    hit_share: float | None
    raw_recall: float | None


class ChannelAttributionSummary(BaseModel):
    """Per-channel attribution aggregated over questions with expectations."""

    questions: int
    channels: dict[str, ChannelCredit]


class EvalRunSummary(BaseModel):
    questions: int
    mean_recall: float | None
    mean_mrr: float | None
    mean_ndcg: float | None
    doc_hit_rate: float | None
    rerank: RerankComparison | None = None
    channel_attribution: ChannelAttributionSummary | None = None


class EvalRunOut(BaseModel):
    run_id: str
    set_id: str
    template: EvalTemplate
    include_answer: bool
    version_id: str | None = None
    created_ts: float
    summary: EvalRunSummary
    results: list[EvalResultOut]


class EvalRunListResponse(BaseModel):
    set_id: str
    items: list[dict[str, Any]]


# ---- row mapping helpers ----


def _loads_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def question_out(row: dict[str, Any]) -> EvalQuestionOut:
    return EvalQuestionOut(
        question_id=row["question_id"],
        question=row["question"],
        expected_chunk_ids=_loads_list(row["expected_chunk_ids"]),
        expected_doc_ids=_loads_list(row["expected_doc_ids"]),
        expected_answer=row["expected_answer"],
        created_ts=float(row["created_ts"]),
    )


def set_out(row: dict[str, Any], *, question_count: int = 0) -> EvalSetOut:
    return EvalSetOut(
        set_id=row["set_id"],
        database=row["database"],
        collection=row["collection"],
        name=row["name"],
        description=row["description"],
        question_count=question_count,
        created_ts=float(row["created_ts"]),
    )


def result_out(row: dict[str, Any], *, question: str) -> EvalResultOut:
    try:
        metrics = json.loads(row.get("metrics_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        metrics = {}
    return EvalResultOut(
        question_id=row["question_id"],
        question=question,
        chunk_ids=_loads_list(row["chunk_ids"]),
        metrics=metrics if isinstance(metrics, dict) else {},
        answer=row["answer"],
        answer_error=row["answer_error"],
        created_ts=float(row["created_ts"]),
    )


def gate_out(row: dict[str, Any]) -> GateOut:
    try:
        template = json.loads(row.get("template_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        template = {}
    return GateOut(
        gate_id=row["gate_id"],
        database=row["database"],
        collection=row["collection"],
        set_id=row["set_id"],
        version_id=row["version_id"],
        template=template if isinstance(template, dict) else {},
        min_recall=row["min_recall"],
        min_mrr=row["min_mrr"],
        min_ndcg=row["min_ndcg"],
        max_recall_drop=row["max_recall_drop"],
        max_mrr_drop=row["max_mrr_drop"],
        max_ndcg_drop=row["max_ndcg_drop"],
        baseline_run_id=row["baseline_run_id"],
        created_ts=float(row["created_ts"]),
    )


def check_out(row: dict[str, Any]) -> GateCheckOut:
    try:
        report = json.loads(row.get("report_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        report = {}
    return GateCheckOut(
        check_id=row["check_id"],
        gate_id=row["gate_id"],
        run_id=row["run_id"],
        candidate_ref=row["candidate_ref"],
        status=row["status"],
        report=report if isinstance(report, dict) else {},
        created_ts=float(row["created_ts"]),
    )
