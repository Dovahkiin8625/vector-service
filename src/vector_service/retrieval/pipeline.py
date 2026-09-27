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

from vector_service.chunking.llm_chunker import (
    get_chat_client,
    is_llm_configured,
)
from vector_service.core.errors import CollectionNotFound
from vector_service.retrieval import transforms as T
from vector_service.retrieval.base import (
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from vector_service.retrieval.channels import BM25Channel, DenseChannel
from vector_service.retrieval.diversity import mmr
from vector_service.retrieval.fusion import rrf_fuse, weighted_fuse
from vector_service.schemas.retrieval import (
    FilterSpec,
    RetrievalRequest,
)

_SPARSE_FIELD = "sparse"
_VECTOR_FIELD = "vector"

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
    """Convert filter.doc_id (eq) / filename (like) into a Milvus expression.

    Empty fields are omitted; LIKE wildcards and quotes in user input
    are escaped so a filter value cannot break out of the expression.
    """
    parts: list[str] = []
    doc_id = (spec.doc_id or "").strip()
    filename = (spec.filename or "").strip()
    if doc_id:
        parts.append(f'doc_id == "{_escape_eq(doc_id)}"')
    if filename:
        parts.append(f'filename like "%{_escape_like(filename)}%"')
    return " and ".join(parts) if parts else None


class RetrievalPipeline:
    """One-shot orchestrator; cheap to construct per request."""

    def __init__(
        self,
        *,
        settings: Any,
        store: Any,
        embedder: Any,
        reranker: Any,
        chat_getter: Callable = get_chat_client,
        chat_check: Callable = is_llm_configured,
    ):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._chat_getter = chat_getter
        self._chat_check = chat_check

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
        try:
            info = await loop.run_in_executor(
                None, self._store.collection_info, req.database, req.collection
            )
        except CollectionNotFound:
            raise _http(
                404, "collection_not_found",
                f"collection {req.collection!r} does not exist in database "
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
        if req.channels.bm25 and _SPARSE_FIELD not in field_names:
            raise _http(
                422, "retrieval_channel_unsupported",
                "BM25 channel requires schema v2; migrate the ingest collection",
                channels=["bm25"], migration_available=True,
            )

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

        # ---- rewrite ----
        await _send(emit, {"type": "stage", "stage": "rewrite"})
        timer = _Timer("rewrite")
        chat_fn = None
        if ctx["rewrite_on"]:
            chat_client = await loop.run_in_executor(
                None, self._chat_getter, self._settings
            )
            chat_fn = chat_client.as_chat_fn()
        plan = await loop.run_in_executor(
            None, self._build_plan, req, chat_fn
        )
        traces.append(timer.done({
            "dense_queries": len(plan.dense_specs),
            "lexical_queries": len(plan.lexical_queries),
        }))

        # Recall legs may need to feed a wider rerank pool later.
        want_pool = req.rerank.candidate_pool if req.rerank.enabled else 25
        recall_limit = min(64, max(req.top_k, want_pool))

        # ---- recall ----
        await _send(emit, {"type": "stage", "stage": "recall"})
        timer = _Timer("recall")
        filter_expr = build_filter_expr(req.filter)
        runs = await self._recall(
            loop, plan, req, recall_limit, filter_expr, ctx["output_fields"]
        )
        traces.append(timer.done({"legs": len(runs), "per_leg_top_k": recall_limit}))

        # ---- fuse ----
        await _send(emit, {"type": "stage", "stage": "fuse"})
        timer = _Timer("fuse")
        if req.fusion.method == "rrf":
            chunks = rrf_fuse(runs, rrf_k=req.fusion.rrf_k)
        else:
            chunks = weighted_fuse(runs, req.fusion.weights.model_dump())
        traces.append(timer.done({
            "method": req.fusion.method, "chunks": len(chunks),
        }))

        if not chunks:
            return RetrievalResult(req.query, [], plan, runs, traces)

        # ---- mmr ----
        if req.mmr.enabled:
            await _send(emit, {"type": "stage", "stage": "mmr"})
            timer = _Timer("mmr")
            select_count = (
                req.rerank.candidate_pool if req.rerank.enabled else req.top_k
            )
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
        if req.rerank.enabled:
            await _send(emit, {"type": "stage", "stage": "rerank"})
            timer = _Timer("rerank")
            pool = chunks[: req.rerank.candidate_pool]
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
            traces.append(timer.done({"candidates": len(pool)}))
        else:
            chunks = chunks[: req.top_k]

        return RetrievalResult(req.query, chunks, plan, runs, traces)

    # ---- internal ----

    def _build_plan(
        self, req: RetrievalRequest, chat_fn: Any
    ) -> RetrievalPlan:
        query = req.query.strip()
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
        filter_expr: str | None,
        output_fields: list[str],
    ) -> list:
        dense_ch = DenseChannel(self._store, req.database, req.collection,
                                self._embedder, _VECTOR_FIELD)
        bm25_ch = BM25Channel(self._store, req.database, req.collection,
                              _SPARSE_FIELD)
        tasks: list[Awaitable] = []
        if req.channels.dense:
            for spec in plan.dense_specs:
                tasks.append(loop.run_in_executor(
                    None,
                    functools.partial(dense_ch.recall, spec, top_k,
                                      filter_expr, output_fields),
                ))
        if req.channels.bm25:
            for query_text in plan.lexical_queries:
                spec = RecallSpec(query=query_text)
                tasks.append(loop.run_in_executor(
                    None,
                    functools.partial(bm25_ch.recall, spec, top_k,
                                      filter_expr, output_fields),
                ))
        return list(await asyncio.gather(*tasks))


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
