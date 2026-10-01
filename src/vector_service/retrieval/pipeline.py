"""RetrievalPipeline: transform → recall → fuse → MMR → rerank.

All blocking work (embedding, store RPCs, chat, rerank) is pushed to a
thread executor; recall legs within one stage run concurrently. The
pipeline raises :class:`fastapi.HTTPException` directly — callers
(JSON and NDJSON routes) decide how to surface it.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any, Awaitable, Callable

from fastapi import HTTPException
from opentelemetry import trace

from vector_service.chunking.llm_chunker import (
    get_chat_client,
    is_llm_configured,
)
from vector_service.core.errors import CollectionNotFound
from vector_service.core.metrics import (
    RETRIEVAL_CHANNEL_HIT_REQUESTS_TOTAL,
    RETRIEVAL_CHANNEL_RUNS_TOTAL,
    RETRIEVAL_FUSION_INPUT_TOTAL,
    RETRIEVAL_FUSION_OUTPUT_TOTAL,
    RETRIEVAL_RECALLED_HITS_TOTAL,
    RETRIEVAL_REQUESTS_TOTAL,
    RETRIEVAL_RERANK_CANDIDATES,
    RETRIEVAL_RERANK_INPUT_CHARS,
    RETRIEVAL_RERANK_STAGE_DURATION_SECONDS,
)
from vector_service.graph.names import graph_collection_names
from vector_service.retrieval import transforms as T
from vector_service.retrieval.base import (
    ChannelRun,
    MetaPredicate,
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from vector_service.retrieval.channels import (
    BM25Channel,
    DenseChannel,
    GraphChannel,
    SummaryChannel,
)
from vector_service.retrieval.diversity import mmr
from vector_service.retrieval.expand import expand_to_level
from vector_service.retrieval.fusion import rrf_fuse, weighted_fuse
from vector_service.retrieval.routing import (
    NUM_DOC_FIELDS,
    TEXT_DOC_FIELDS,
    heuristic_route,
    llm_route,
)
from vector_service.schemas.retrieval import (
    FilterSpec,
    RetrievalRequest,
)

_SPARSE_FIELD = "sparse"
_VECTOR_FIELD = "vector"

# With tracing disabled the API's proxy tracer hands back non-recording
# spans, so every ``start_as_current_span`` below stays a cheap no-op.
_TRACER = trace.get_tracer("vector_service.retrieval")

StageEmit = Callable[[dict], Any]


async def _noop_emit(_event: dict) -> None:
    ...


async def _send(emit: StageEmit, event: dict) -> None:
    """Deliver one stage event; tolerate both sync and async sinks."""
    outcome = emit(event)
    if inspect.isawaitable(outcome):
        await outcome


def _http(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code, detail={
        "error": {"code": code, "message": message, **extra},
    })


class _Timer:
    def __init__(self, stage: str):
        self.stage = stage
        self.start = time.perf_counter()

    def done(self, detail: dict | None = None) -> StageTrace:
        duration = int((time.perf_counter() - self.start) * 1000)
        return StageTrace(self.stage, duration, detail or {})


def _escape_eq(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_like(value: str) -> str:
    return _escape_eq(value).replace("%", "\\%").replace("_", "\\_")


def build_filter_expr(spec: FilterSpec) -> str | None:
    """Convert filter.doc_id (eq) into a thin-index Milvus expression.

    Document-level fields (filename, title, ...) are not thin-index
    scalars — the pipeline resolves those predicates against SQLite and
    compiles them onto ``doc_id`` lists. Quotes in user input are
    escaped so a value cannot break out of the expression.
    """
    doc_id = (spec.doc_id or "").strip()
    return f'doc_id == "{_escape_eq(doc_id)}"' if doc_id else None


# Ops the thin index can apply directly per scalar kind.
_NUMERIC_OPS = ("==", "!=", ">", "<", ">=", "<=")
_TEXT_OPS = ("==", "!=", "like")


def thin_predicate_expr(predicate: MetaPredicate) -> str | None:
    """Compile one doc_id / chunk_index predicate onto the thin scalars.

    Returns ``None`` for predicates that cannot be expressed directly
    (document fields are resolved by SQLite instead).
    """
    if predicate.field == "chunk_index":
        if predicate.op not in _NUMERIC_OPS:
            return None
        return f"chunk_index {predicate.op} {int(predicate.value)}"
    if predicate.field == "doc_id":
        if predicate.op not in _TEXT_OPS:
            return None
        if predicate.op == "like":
            return f'doc_id like "%{_escape_like(predicate.value)}%"'
        return f'doc_id {predicate.op} "{_escape_eq(predicate.value)}"'
    return None


def doc_id_in_expr(doc_ids: set[str]) -> str:
    """Build a quoted ``doc_id in [...]`` expression from an id set."""
    quoted = ", ".join(f'"{_escape_eq(doc_id)}"' for doc_id in sorted(doc_ids))
    return f"doc_id in [{quoted}]"


class _RecallFilter:
    """Resolved constraints shared by every recall leg."""

    def __init__(
        self,
        *,
        expr: str | None,
        doc_ids: set[str] | None,
        index_predicates: list[MetaPredicate],
        matches: bool,
    ):
        self.expr = expr
        self.doc_ids = doc_ids
        self.index_predicates = index_predicates
        self.matches = matches


class RetrievalPipeline:
    """One-shot orchestrator; cheap to construct per request."""

    def __init__(
        self,
        *,
        settings: Any,
        store: Any,
        embedder: Any,
        reranker: Any,
        repo: Any = None,
        chat_getter: Callable = get_chat_client,
        chat_check: Callable = is_llm_configured,
    ):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._repo = repo
        self._chat_getter = chat_getter
        self._chat_check = chat_check

    def _resolve_physical(self, req: RetrievalRequest) -> str:
        """Resolve the physical collection recall runs against.

        An explicit ``index_ref`` wins (gate checks pin the canary);
        otherwise the binding's ``active_ref`` applies; an unbound
        logical collection is its own physical name.
        """
        if req.index_ref:
            return req.index_ref
        if self._repo is not None:
            binding = self._repo.get_binding(req.database, req.collection)
            if binding is not None:
                return str(binding["active_ref"])
        return req.collection

    async def validate(self, req: RetrievalRequest) -> dict:
        """Pre-flight checks + schema probe. Shared by JSON/stream routes."""
        if not req.query or not req.query.strip():
            raise _http(422, "retrieval_empty_query", "query must be non-empty")
        if self._embedder is None:
            raise _http(
                503, "embedder_unavailable",
                "text embedder is not loaded; POST /v1/models/{model}/load first",
            )

        loop = asyncio.get_running_loop()
        physical = await loop.run_in_executor(
            None, self._resolve_physical, req
        )
        try:
            info = await loop.run_in_executor(
                None, self._store.collection_info, req.database, physical
            )
        except CollectionNotFound:
            raise _http(
                404, "collection_not_found",
                f"collection {physical!r} does not exist in database "
                f"{req.database!r}",
            ) from None

        schema_field_names = [
            f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            for f in getattr(info, "fields", [])
        ]
        field_names = set(schema_field_names)
        # Recall legs must get an explicit scalar projection: the adapter's
        # output_fields=None legacy default resolves to ["vector"], which
        # after the vector is popped leaves every hit with empty fields
        # (contentless results; MMR/rerank then operate on empty strings).
        output_fields = [
            name for name in schema_field_names
            if name and name not in (_VECTOR_FIELD, _SPARSE_FIELD)
        ]

        # Summary leg needs the summary_vector field; graph legs need the
        # two derived graph collections (plus graph rows when the repo is
        # attached).
        summary_available = "summary_vector" in field_names
        entity_coll, community_coll = graph_collection_names(req.collection)
        try:
            coll_names = await loop.run_in_executor(
                None, self._store.list_collections, req.database
            )
        except CollectionNotFound:
            coll_names = []
        graph_available = entity_coll in coll_names and community_coll in coll_names
        if graph_available and self._repo is not None:
            stats = await loop.run_in_executor(
                None, self._repo.graph_stats, req.database, req.collection
            )
            graph_available = int(stats["entities_count"]) > 0

        if req.rerank.enabled and (
            self._reranker is None or getattr(self._reranker, "_impl", None) is None
        ):
            raise _http(
                503, "reranker_not_loaded",
                "reranker is not loaded; POST /v1/models/{model}/load first",
            )

        rewrite_on = req.rewrite.enabled and any(
            m in req.rewrite.methods
            for m in ("hyde", "multi_query", "step_back", "decompose")
        )
        if rewrite_on and not self._chat_check(self._settings):
            raise _http(
                503, "llm_unavailable",
                "LLM is not configured; set VS_LLM__BASE_URL and VS_LLM__MODEL",
            )
        return {
            "info": info,
            "rewrite_on": rewrite_on,
            "output_fields": output_fields,
            "summary_available": summary_available,
            "graph_available": graph_available,
            "physical": physical,
        }

    async def retrieve(
        self,
        req: RetrievalRequest,
        *,
        emit: StageEmit = _noop_emit,
        prevalidated: dict | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline and return answer + full trace."""
        ctx = prevalidated if prevalidated is not None else await self.validate(req)
        loop = asyncio.get_running_loop()
        traces: list[StageTrace] = []

        RETRIEVAL_REQUESTS_TOTAL.inc()
        rerank_model = (
            getattr(self._reranker, "model_name", "unknown")
            if self._reranker is not None
            else "unknown"
        )

        with _TRACER.start_as_current_span("retrieval") as root_span:
            root_span.set_attributes({
                "vs.database": req.database,
                "vs.collection": req.collection,
                "vs.physical_collection": ctx["physical"],
                "vs.query_length": len(req.query),
                "vs.top_k": req.top_k,
            })

            # ---- route ----
            decision = None
            if req.routing.enabled:
                await _send(emit, {"type": "stage", "stage": "route"})
                timer = _Timer("route")
                with _TRACER.start_as_current_span("route"):
                    decision = await self._route(loop, req)
                traces.append(timer.done({
                    "router": decision.router,
                    "intents": list(decision.intents),
                    "signals": list(decision.signals),
                }))

            effective_query = (
                decision.query if decision is not None else req.query.strip()
            )

            # ---- rewrite ----
            await _send(emit, {"type": "stage", "stage": "rewrite"})
            timer = _Timer("rewrite")
            with _TRACER.start_as_current_span("rewrite"):
                chat_fn = None
                if ctx["rewrite_on"]:
                    chat_client = await loop.run_in_executor(
                        None, self._chat_getter, self._settings
                    )
                    chat_fn = chat_client.as_chat_fn()
                plan = await loop.run_in_executor(
                    None, self._build_plan, req, chat_fn, effective_query
                )
            traces.append(timer.done({
                "dense_queries": len(plan.dense_specs),
                "lexical_queries": len(plan.lexical_queries),
            }))

            # Recall legs may need to feed a wider rerank pool later.
            want_pool = req.rerank.candidate_pool if req.rerank.enabled else 25
            recall_limit = min(64, max(req.top_k, want_pool))

            # ---- resolve filters (predicates + corpus doc-id resolution) ----
            ctx["decision"] = decision
            recall_filter = await loop.run_in_executor(
                None, self._resolve_filter, req, decision
            )

            # ---- recall ----
            await _send(emit, {"type": "stage", "stage": "recall"})
            timer = _Timer("recall")
            with _TRACER.start_as_current_span("recall"):
                runs = await self._recall(
                    loop, plan, req, recall_limit, recall_filter, ctx
                )
            traces.append(timer.done({"legs": len(runs), "per_leg_top_k": recall_limit}))
            for run in runs:
                leg_status = "ok" if run.hits else "empty"
                RETRIEVAL_CHANNEL_RUNS_TOTAL.labels(
                    channel=run.channel, status=leg_status
                ).inc()
                RETRIEVAL_RECALLED_HITS_TOTAL.labels(
                    channel=run.channel
                ).inc(len(run.hits))
            fusion_input_ids = {
                hit.chunk_id for run in runs for hit in run.hits
            }
            RETRIEVAL_FUSION_INPUT_TOTAL.labels(
                method=req.fusion.method
            ).inc(len(fusion_input_ids))

            # ---- fuse ----
            await _send(emit, {"type": "stage", "stage": "fuse"})
            timer = _Timer("fuse")
            with _TRACER.start_as_current_span("fuse"):
                if req.fusion.method == "rrf":
                    chunks = rrf_fuse(runs, rrf_k=req.fusion.rrf_k)
                else:
                    chunks = weighted_fuse(runs, req.fusion.weights.model_dump())

                # ---- hydrate content + citation fields from the corpus ----
                with _TRACER.start_as_current_span("hydrate"):
                    hydrated = 0
                    if self._repo is not None and chunks:
                        rows = await loop.run_in_executor(
                            None,
                            self._repo.hydrate, [c.chunk_id for c in chunks],
                        )
                        for chunk in chunks:
                            row = rows.get(chunk.chunk_id)
                            if row is not None:
                                chunk.fields.update(row)
                        hydrated = len(rows)
            RETRIEVAL_FUSION_OUTPUT_TOTAL.labels(
                method=req.fusion.method
            ).inc(len(chunks))
            traces.append(timer.done({
                "method": req.fusion.method, "chunks": len(chunks),
                "hydrated": hydrated,
            }))

            if not chunks:
                return RetrievalResult(
                    req.query, [], plan, runs, traces, route=decision
                )

            # ---- mmr ----
            if req.mmr.enabled:
                await _send(emit, {"type": "stage", "stage": "mmr"})
                timer = _Timer("mmr")
                select_count = (
                    req.rerank.candidate_pool if req.rerank.enabled else req.top_k
                )
                with _TRACER.start_as_current_span("mmr"):
                    pool = chunks[: min(len(chunks), max(select_count * 2, 30))]
                    query_vector = await loop.run_in_executor(
                        None, self._embedder.embed_query, req.query
                    )
                    doc_vectors = await loop.run_in_executor(
                        None,
                        self._embedder.embed_documents,
                        [str(c.fields.get("text", "")) for c in pool],
                    )
                    chunks = mmr(
                        pool, doc_vectors, query_vector,
                        lambda_mult=req.mmr.lambda_mult, top_k=select_count,
                    )
                traces.append(timer.done({
                    "pool": len(pool), "selected": len(chunks),
                }))

            # ---- rerank ----
            rerank_input_ids = None
            if req.rerank.enabled:
                await _send(emit, {"type": "stage", "stage": "rerank"})
                timer = _Timer("rerank")
                with _TRACER.start_as_current_span("retrieval.rerank"):
                    pool = chunks[: req.rerank.candidate_pool]
                    # Snapshot the fused order before reranking for pre/post
                    # comparison (eval metrics consume this on the result).
                    rerank_input_ids = [c.chunk_id for c in pool]
                    documents = [str(c.fields.get("text", "")) for c in pool]
                    scored = await loop.run_in_executor(
                        None,
                        functools.partial(self._reranker.rerank, req.query, documents,
                                          req.top_k),
                    )
                    reranked: list[RetrievedChunk] = []
                    for hit in scored:
                        chunk = pool[hit.index]
                        chunk.rerank_score = float(hit.score)
                        reranked.append(chunk)
                    chunks = reranked[: req.top_k]
                stage = timer.done({"candidates": len(pool)})
                traces.append(stage)
                RETRIEVAL_RERANK_STAGE_DURATION_SECONDS.labels(
                    model=rerank_model
                ).observe(stage.duration_ms / 1000.0)
                RETRIEVAL_RERANK_INPUT_CHARS.labels(model=rerank_model).observe(
                    sum(len(d) for d in documents)
                )
                RETRIEVAL_RERANK_CANDIDATES.labels(model=rerank_model).observe(
                    len(pool)
                )
            else:
                chunks = chunks[: req.top_k]

            # ---- expand leaves up to the requested context level ----
            if req.context_level != "chunk" and self._repo is not None:
                await _send(emit, {"type": "stage", "stage": "expand"})
                timer = _Timer("expand")
                leaf_count = len(chunks)
                with _TRACER.start_as_current_span("expand"):
                    chunks = await loop.run_in_executor(
                        None,
                        functools.partial(
                            expand_to_level,
                            repo=self._repo, level=req.context_level,
                        ),
                        chunks,
                    )
                traces.append(timer.done({
                    "level": req.context_level,
                    "leaves": leaf_count, "outputs": len(chunks),
                }))

            # ---- compress to token budget ----
            if req.context.max_tokens is not None:
                await _send(emit, {"type": "stage", "stage": "compress"})
                timer = _Timer("compress")
                before = len(chunks)
                with _TRACER.start_as_current_span("compress"):
                    chunks = _compress_to_budget(chunks, req.context.max_tokens)
                traces.append(timer.done({
                    "budget": req.context.max_tokens,
                    "kept": len(chunks), "dropped": before - len(chunks),
                }))

            # Per-request channel hit rate: a channel counts when any
            # final chunk came back through it.
            for channel in {
                ch for chunk in chunks for ch in chunk.matched_channels
            }:
                RETRIEVAL_CHANNEL_HIT_REQUESTS_TOTAL.labels(channel=channel).inc()

            return RetrievalResult(
                req.query, chunks, plan, runs, traces, route=decision,
                rerank_input_ids=rerank_input_ids,
            )

    # ---- internal ----

    async def _route(
        self, loop: asyncio.AbstractEventLoop, req: RetrievalRequest
    ):
        """Classify intent; LLM classification degrades to heuristic."""
        query = req.query.strip()
        if req.routing.use_llm and self._chat_check(self._settings):
            chat_client = await loop.run_in_executor(
                None, self._chat_getter, self._settings
            )
            decision = await loop.run_in_executor(
                None, llm_route, chat_client.as_chat_fn(), query
            )
            if decision is not None:
                return decision
        return await loop.run_in_executor(None, heuristic_route, query)

    def _resolve_filter(
        self, req: RetrievalRequest, decision: Any
    ) -> _RecallFilter:
        """Compile explicit filters + routed predicates into recall constraints."""
        predicates = list(decision.predicates) if decision else []
        expr_parts: list[str] = []
        base_expr = build_filter_expr(req.filter)
        if base_expr:
            expr_parts.append(base_expr)
        index_predicates: list[MetaPredicate] = []
        for predicate in predicates:
            fragment = thin_predicate_expr(predicate)
            if fragment is not None:
                expr_parts.append(fragment)
            if predicate.field == "chunk_index":
                index_predicates.append(predicate)

        # Document-field predicates are resolved against SQLite (the thin
        # Milvus index has no document columns), incl. the explicit
        # filename filter, then compiled onto a doc_id list.
        doc_predicates = [
            p for p in predicates
            if p.field in (*TEXT_DOC_FIELDS, *NUM_DOC_FIELDS)
        ]
        filename = (req.filter.filename or "").strip()
        if filename:
            doc_predicates.append(MetaPredicate(
                field="filename", op="like", value=filename,
            ))

        doc_ids: set[str] | None = None
        matches = True
        if self._repo is not None:
            doc_ids = self._repo.document_ids_for_predicates(
                req.database, req.collection, doc_predicates
            )
            if doc_ids is not None:
                if not doc_ids:
                    matches = False
                else:
                    expr_parts.append(doc_id_in_expr(doc_ids))

        expr = " and ".join(expr_parts) if expr_parts else None
        return _RecallFilter(
            expr=expr,
            doc_ids=doc_ids,
            index_predicates=index_predicates,
            matches=matches,
        )

    def _build_plan(
        self,
        req: RetrievalRequest,
        chat_fn: Any,
        query: str,
    ) -> RetrievalPlan:
        dense = [RecallSpec(query=query)]
        lexical: list[str] = [query]
        sub_queries: list[str] = []

        if chat_fn is not None:
            methods = req.rewrite.methods
            if "hyde" in methods:
                # HyDE replaces the base dense leg with the q/h mix; it is
                # not a lexical query.
                dense = [T.hyde(chat_fn, query, self._embedder,
                                alpha=req.rewrite.hyde_alpha)]
            if "multi_query" in methods:
                for variant in T.multi_query(chat_fn, query, req.rewrite.n_variants):
                    if variant != query:
                        dense.append(RecallSpec(query=variant))
                        lexical.append(variant)
            if "step_back" in methods:
                broader = T.step_back(chat_fn, query)
                if broader != query:
                    dense.append(RecallSpec(query=broader))
                    lexical.append(broader)
            if "decompose" in methods:
                parts = T.decompose(chat_fn, query)
                sub_queries = list(parts)
                for part in parts:
                    if part and part != query:
                        dense.append(RecallSpec(query=part))
                        lexical.append(part)

        return RetrievalPlan(
            original_query=query,
            dense_specs=_dedupe_specs(dense),
            lexical_queries=_dedupe_strings(lexical),
            sub_queries=sub_queries,
        )

    async def _recall(
        self,
        loop: asyncio.AbstractEventLoop,
        plan: RetrievalPlan,
        req: RetrievalRequest,
        top_k: int,
        recall_filter: _RecallFilter,
        ctx: dict,
    ) -> list[ChannelRun]:
        # Routing (when on) drives the channel set; otherwise the
        # explicit channels spec applies.
        use_dense = req.channels.dense
        use_bm25 = req.channels.bm25
        use_summary = False
        use_graph = False
        if req.routing.enabled:
            decision = ctx["decision"]
            use_dense = bool(decision.dense)
            use_bm25 = bool(decision.bm25)
            use_summary = bool(decision.summary and ctx["summary_available"])
            use_graph = bool(decision.graph and ctx["graph_available"])

        output_fields = ctx["output_fields"]
        physical = ctx["physical"]

        if not recall_filter.matches:
            # A resolved predicate set matched no document: keep the
            # planned legs visible in the trace as empty runs.
            return self._empty_runs(
                plan,
                use_dense=use_dense,
                use_bm25=use_bm25,
                use_summary=use_summary,
                use_graph=use_graph,
            )

        dense_ch = DenseChannel(self._store, req.database, physical,
                                self._embedder, _VECTOR_FIELD)
        bm25_ch = BM25Channel(self._store, req.database, physical,
                              _SPARSE_FIELD)
        summary_ch = SummaryChannel(self._store, req.database, physical,
                                    self._embedder)
        # (channel, query, awaitable): legs are scheduled as before,
        # and gathered each inside its own ``recall.<channel>`` span.
        legs: list[tuple[str, str, Awaitable]] = []
        if use_dense:
            for spec in plan.dense_specs:
                legs.append(("dense", spec.query, loop.run_in_executor(
                    None,
                    functools.partial(dense_ch.recall, spec, top_k,
                                      recall_filter.expr, output_fields),
                )))
        if use_summary:
            for spec in plan.dense_specs:
                legs.append(("summary", spec.query, loop.run_in_executor(
                    None,
                    functools.partial(summary_ch.recall, spec, top_k,
                                      recall_filter.expr, output_fields),
                )))
        if use_bm25:
            for query_text in plan.lexical_queries:
                spec = RecallSpec(query=query_text)
                legs.append(("bm25", query_text, loop.run_in_executor(
                    None,
                    functools.partial(bm25_ch.recall, spec, top_k,
                                      recall_filter.expr, output_fields),
                )))
        if use_graph:
            entity_coll, community_coll = graph_collection_names(req.collection)
            graph_ch = GraphChannel(
                store=self._store,
                repo=self._repo,
                database=req.database,
                entity_collection=entity_coll,
                community_collection=community_coll,
                embedder=self._embedder,
            )
            legs.append(("graph", plan.original_query, loop.run_in_executor(
                None,
                functools.partial(
                    graph_ch.recall,
                    RecallSpec(query=plan.original_query),
                    top_k,
                    doc_ids=recall_filter.doc_ids,
                    index_predicates=recall_filter.index_predicates,
                ),
            )))

        async def _run_leg(channel: str, query: str, coro: Awaitable):
            with _TRACER.start_as_current_span(f"recall.{channel}") as leg_span:
                leg_span.set_attribute("vs.query_length", len(query))
                return await coro

        return list(await asyncio.gather(*(
            _run_leg(channel, query, coro)
            for channel, query, coro in legs
        )))

    @staticmethod
    def _empty_runs(
        plan: RetrievalPlan,
        *,
        use_dense: bool,
        use_bm25: bool,
        use_summary: bool,
        use_graph: bool,
    ) -> list[ChannelRun]:
        runs: list[ChannelRun] = []
        if use_dense:
            runs.extend(
                ChannelRun(channel="dense", query=spec.query)
                for spec in plan.dense_specs
            )
        if use_summary:
            runs.extend(
                ChannelRun(channel="summary", query=spec.query)
                for spec in plan.dense_specs
            )
        if use_bm25:
            runs.extend(
                ChannelRun(channel="bm25", query=query)
                for query in plan.lexical_queries
            )
        if use_graph:
            runs.append(ChannelRun(
                channel="graph", query=plan.original_query,
            ))
        return runs


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dedupe_specs(specs: list[RecallSpec]) -> list[RecallSpec]:
    # Keep the first spec per wording — HyDE (vector set) comes first and
    # must not be shadowed by a later plain-vector spec for the same query.
    seen: set[str] = set()
    out: list[RecallSpec] = []
    for spec in specs:
        if spec.query not in seen:
            seen.add(spec.query)
            out.append(spec)
    return out


def _chunk_tokens(chunk: RetrievedChunk) -> int:
    raw = chunk.fields.get("token_count")
    return int(raw) if raw is not None else 0


def _compress_to_budget(
    chunks: list[RetrievedChunk], max_tokens: int
) -> list[RetrievedChunk]:
    """Greedy token-budget cut over the ranked chunks.

    The top-ranked chunk is always kept even if it alone exceeds the
    budget; in that case its overflow is forgiven and the following
    chunks pack against the full budget. Otherwise packing is
    cumulative. Zero-token chunks (missing ``token_count``) are never
    dropped; the first positive-token chunk that would reach the budget
    is where the cut happens — it and everything after it are dropped.
    """
    if not chunks:
        return []
    kept: list[RetrievedChunk] = [chunks[0]]
    first_tokens = _chunk_tokens(chunks[0])
    used = 0 if first_tokens > max_tokens else first_tokens
    for chunk in chunks[1:]:
        tokens = _chunk_tokens(chunk)
        if tokens > 0 and used + tokens >= max_tokens:
            break
        kept.append(chunk)
        used += tokens
    return kept
