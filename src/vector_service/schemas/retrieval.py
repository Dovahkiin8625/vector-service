"""Pydantic schemas for the retrieval API and pipeline trace."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from vector_service.retrieval.base import (
    ChannelRun,
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from vector_service.retrieval.expand import CONTEXT_LEVELS

REWRITE_METHODS = ("hyde", "multi_query", "step_back", "decompose")


# ---- request ----

class FilterSpec(BaseModel):
    doc_id: str = ""
    filename: str = ""


class ChannelWeights(BaseModel):
    dense: float = 0.5
    bm25: float = 0.5
    summary: float = 0.5
    graph: float = 0.5


class ChannelsSpec(BaseModel):
    dense: bool = True
    bm25: bool = True

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.dense or self.bm25):
            raise ValueError("at least one channel must be enabled (dense / bm25)")
        return self


class FusionSpec(BaseModel):
    method: str = "rrf"
    rrf_k: int = Field(60, ge=1, le=200)
    weights: ChannelWeights = Field(default_factory=ChannelWeights)

    @field_validator("method")
    @classmethod
    def _method_known(cls, value):
        if value not in ("rrf", "weighted"):
            raise ValueError("fusion.method must be 'rrf' or 'weighted'")
        return value

    @model_validator(mode="after")
    def _weights_valid(self):
        values = (self.weights.dense, self.weights.bm25,
                  self.weights.summary, self.weights.graph)
        if any(w < 0 for w in values):
            raise ValueError("fusion weights must be >= 0")
        if all(w == 0 for w in values):
            raise ValueError("at least one fusion weight must be > 0")
        return self


class RewriteSpec(BaseModel):
    enabled: bool = False
    methods: list[str] = Field(default_factory=lambda: list(REWRITE_METHODS))
    hyde_alpha: float = Field(0.7, ge=0.0, le=1.0)
    n_variants: int = Field(3, ge=1, le=5)

    @field_validator("methods")
    @classmethod
    def _methods_known(cls, value):
        unknown = [m for m in value if m not in REWRITE_METHODS]
        if unknown:
            raise ValueError(f"unknown rewrite methods {unknown}")
        if len(value) != len(set(value)):
            raise ValueError("duplicate rewrite methods")
        return value


class MMRSpec(BaseModel):
    enabled: bool = False
    lambda_mult: float = Field(0.7, ge=0.0, le=1.0)


class RerankSpec(BaseModel):
    enabled: bool = True
    candidate_pool: int = Field(25, ge=1, le=64)


class RoutingSpec(BaseModel):
    #: Auto-select channels/predicates from query intent; when off the
    #: explicit ``channels`` spec drives recall.
    enabled: bool = False
    #: Let the LLM classify; silently degrades to the heuristic router.
    use_llm: bool = False


class ContextSpec(BaseModel):
    #: Trim the ranked answer so cumulative chunk tokens stay within this
    #: budget; the top-ranked chunk is always kept even if it exceeds it.
    max_tokens: int | None = Field(None, ge=1, le=100_000)


class RetrievalRequest(BaseModel):
    database: str = Field("default", min_length=1)
    collection: str = Field("ingest", min_length=1)
    #: Pin recall to one physical index (canary/active ref); defaults to
    #: the binding's active_ref (or the logical name when unbound).
    index_ref: str | None = None
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    filter: FilterSpec = Field(default_factory=FilterSpec)
    channels: ChannelsSpec = Field(default_factory=ChannelsSpec)
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    rewrite: RewriteSpec = Field(default_factory=RewriteSpec)
    mmr: MMRSpec = Field(default_factory=MMRSpec)
    rerank: RerankSpec = Field(default_factory=RerankSpec)
    routing: RoutingSpec = Field(default_factory=RoutingSpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    #: Expand recalled leaves up to section/document after reranking;
    #: "chunk" keeps the leaf vocabulary.
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
        # Only relevant when reranking actually runs — otherwise the basic
        # preset (rerank off, top_k up to 100) would always 422.
        if self.rerank.enabled and self.rerank.candidate_pool < self.top_k:
            raise ValueError("rerank.candidate_pool must be >= top_k")
        return self


# ---- response ----

class RecallSpecOut(BaseModel):
    # The vector body is intentionally omitted — it can be large and the
    # trace only needs the wording (+ HyDE hypothetical document).
    query: str
    hypothetical: str | None = None


class PlanOut(BaseModel):
    original_query: str
    dense_specs: list[RecallSpecOut]
    lexical_queries: list[str]
    sub_queries: list[str]


class ChannelHitOut(BaseModel):
    chunk_id: str
    score: float
    rank: int
    fields: dict


class ChannelRunOut(BaseModel):
    channel: str
    query: str
    hits: list[ChannelHitOut]


class StageTraceOut(BaseModel):
    stage: str
    duration_ms: int
    detail: dict


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    fields: dict
    fusion_score: float
    matched_channels: list[str]
    rerank_score: float | None = None


class PredicateOut(BaseModel):
    field: str
    op: str
    value: str


class RouteOut(BaseModel):
    router: str
    intents: list[str]
    signals: list[str]
    dense: bool
    bm25: bool
    summary: bool
    graph: bool
    predicates: list[PredicateOut]
    query: str


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunkOut]
    plan: PlanOut
    channel_runs: list[ChannelRunOut]
    traces: list[StageTraceOut]
    route: RouteOut | None = None


def to_result(result: RetrievalResult) -> RetrievalResponse:
    """Map the pipeline's dataclasses onto the API response model."""
    return RetrievalResponse(
        query=result.query,
        chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                fields=c.fields,
                fusion_score=c.fusion_score,
                matched_channels=c.matched_channels,
                rerank_score=c.rerank_score,
            )
            for c in result.chunks
        ],
        plan=PlanOut(
            original_query=result.plan.original_query,
            dense_specs=[
                RecallSpecOut(query=s.query, hypothetical=s.hypothetical)
                for s in result.plan.dense_specs
            ],
            lexical_queries=list(result.plan.lexical_queries),
            sub_queries=list(result.plan.sub_queries),
        ),
        channel_runs=[
            ChannelRunOut(
                channel=run.channel,
                query=run.query,
                hits=[
                    ChannelHitOut(
                        chunk_id=h.chunk_id, score=h.score,
                        rank=h.rank, fields=h.fields,
                    )
                    for h in run.hits
                ],
            )
            for run in result.channel_runs
        ],
        traces=[
            StageTraceOut(stage=tr.stage, duration_ms=tr.duration_ms,
                          detail=tr.detail)
            for tr in result.traces
        ],
        route=(
            RouteOut(
                router=result.route.router,
                intents=list(result.route.intents),
                signals=list(result.route.signals),
                dense=result.route.dense,
                bm25=result.route.bm25,
                summary=result.route.summary,
                graph=result.route.graph,
                predicates=[
                    PredicateOut(
                        field=p.field, op=p.op, value=p.value
                    )
                    for p in result.route.predicates
                ],
                query=result.route.query,
            )
            if result.route is not None
            else None
        ),
    )
