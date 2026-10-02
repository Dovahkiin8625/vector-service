"""Evaluation sets: expected answers + batch retrieval/generation runs.

Evaluation data is corpus metadata (schema v9) and never enters Milvus:
``POST .../runs`` submits an ``eval_run`` job that runs the shared
:class:`~vector_service.retrieval.pipeline.RetrievalPipeline` over every
question of the set, records the ranked chunk ids and metrics
(Recall / MRR / nDCG, expected-doc hit, rerank pre/post comparison and
per-channel attribution), and — when ``include_answer`` is on —
generates an answer from the retrieved chunks via the LLM.

Run results are replaced wholesale on a retry (a run is derivable from
the corpus); no migration or compatibility branches.
"""
from __future__ import annotations

import asyncio
import json
import math
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.api.ingest import JobCancelled
from vector_service.chunking.llm_chunker import get_chat_client, is_llm_configured
from vector_service.core.logging import get_logger
from vector_service.core.threadpools import run_in_sqlite
from vector_service.corpus import JOB_EVALUATING
from vector_service.retrieval.pipeline import RetrievalPipeline
from vector_service.schemas.evaluation import (
    ChannelAttributionSummary,
    ChannelCredit,
    EvalQuestionOut,
    EvalQuestionsCreate,
    EvalQuestionsResponse,
    EvalRunCreate,
    EvalRunOut,
    EvalRunSubmitResponse,
    EvalRunSummary,
    EvalSetCreate,
    EvalSetListResponse,
    EvalSetOut,
    GateCheckListResponse,
    GateCheckSubmitResponse,
    GateCreate,
    GateListResponse,
    RerankComparison,
    VersionCreate,
    VersionListResponse,
    VersionOut,
    check_out,
    gate_out,
    question_out,
    result_out,
    set_out,
)
from vector_service.schemas.retrieval import RetrievalRequest

router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])
log = get_logger(__name__)


def _build_pipeline(state: Any) -> RetrievalPipeline:
    return RetrievalPipeline(
        settings=state.settings,
        store=state.store,
        repo=state.corpus,
        embedder=state.embedder,
        reranker=state.reranker,
        bm25=getattr(state, "bm25", None),
    )


# ---- answer generation ----

_ANSWER_SYSTEM = (
    "你是问答助手。仅根据下面给出的检索片段回答用户问题；"
    "片段不足以回答时，明确说明无法从资料中确认，不要编造。"
)


def _compose_answer(chat_fn: Any, question: str, chunks: list[Any]) -> str:
    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        text = str(chunk.fields.get("text", "")).strip()
        if text:
            context_parts.append(f"[片段{index}] {text}")
    context = "\n\n".join(context_parts) or "（无检索片段）"
    return chat_fn([
        {"role": "system", "content": _ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"问题：{question}\n\n检索片段：\n{context}",
        },
    ])


# ---- metrics ----


def _ranking_metrics(
    ids: list[str], wanted: set[str]
) -> tuple[float, float, float]:
    """Return ``(recall, mrr, ndcg)`` for one ranked id list.

    Binary relevance: a position is relevant iff its id is in ``wanted``.
    nDCG discounts with ``1/log2(rank+1)``; IDCG treats every wanted
    chunk as rankable (capped at the list length).
    """
    hits = 0
    mrr = 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(ids, start=1):
        if chunk_id in wanted:
            hits += 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            dcg += 1.0 / math.log2(rank + 1)

    ideal_count = min(len(wanted), len(ids))
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    ndcg = dcg / idcg if idcg else 0.0
    return hits / len(wanted), mrr, ndcg


def _channel_attribution(
    retrieval: Any, wanted: set[str], final_ids: list[str]
) -> dict[str, Any]:
    """Attribute expected-chunk hits to recall channels.

    ``hits`` counts, per channel, final-list hit chunks that carry the
    channel in ``matched_channels``. ``found`` counts expected chunks
    the channel surfaced anywhere in its raw channel runs — including
    chunks fusion or rerank later dropped. ``hits_total`` /
    ``wanted_total`` are the per-question denominators used when the
    run-level summary aggregates shares.
    """
    hit_ids = wanted.intersection(final_ids)
    hits: dict[str, int] = {}
    for chunk in retrieval.chunks:
        if chunk.chunk_id in hit_ids:
            for name in chunk.matched_channels:
                hits[name] = hits.get(name, 0) + 1

    found_sets: dict[str, set[str]] = {}
    for run in retrieval.channel_runs:
        bucket = found_sets.setdefault(run.channel, set())
        for hit in run.hits:
            if hit.chunk_id in wanted:
                bucket.add(hit.chunk_id)
    found = {name: len(ids) for name, ids in found_sets.items()}

    return {
        "hits": dict(sorted(hits.items())),
        "found": dict(sorted(found.items())),
        "hits_total": len(hit_ids),
        "wanted_total": len(wanted),
    }


def _question_metrics(
    retrieval: Any,
    *,
    expected_chunks: list[str],
    expected_docs: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Per-question ranking metrics and channel attribution.

    ``recall`` / ``mrr`` / ``ndcg`` score the final ranking; when the
    run reranked, ``rerank`` carries the same metrics on the pre-rerank
    order (truncated to ``top_k``) for before/after comparison;
    ``channels`` attributes the hits to the recall channels.
    ``doc_hit`` marks whether any returned chunk belongs to an expected
    document.
    """
    metrics: dict[str, Any] = {}
    final_ids = [c.chunk_id for c in retrieval.chunks]
    if expected_chunks:
        wanted = set(expected_chunks)
        recall, mrr, ndcg = _ranking_metrics(final_ids, wanted)
        metrics["recall"] = recall
        metrics["mrr"] = mrr
        metrics["ndcg"] = ndcg

        if retrieval.rerank_input_ids is not None:
            pre_recall, pre_mrr, pre_ndcg = _ranking_metrics(
                retrieval.rerank_input_ids[:top_k], wanted
            )
            metrics["rerank"] = {
                "pre_recall": pre_recall,
                "pre_mrr": pre_mrr,
                "pre_ndcg": pre_ndcg,
            }
        metrics["channels"] = _channel_attribution(
            retrieval, wanted, final_ids
        )

    if expected_docs:
        wanted_docs = set(expected_docs)
        doc_ids = {
            str(c.fields.get("doc_id", "")) for c in retrieval.chunks
        }
        metrics["doc_hit"] = bool(wanted_docs.intersection(doc_ids))
    return metrics


def _metrics_of(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a stored ``metrics_json`` (internal row or result row)."""
    raw = row.get("metrics_json", "{}")
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _summarize(results: list[dict[str, Any]]) -> EvalRunSummary:
    parsed = [_metrics_of(r) for r in results]
    recalls = [m["recall"] for m in parsed if "recall" in m]
    mrrs = [m["mrr"] for m in parsed if "mrr" in m]
    ndcgs = [m["ndcg"] for m in parsed if "ndcg" in m]
    doc_hits = [m["doc_hit"] for m in parsed if "doc_hit" in m]

    rerank = None
    reranked = [m for m in parsed if "rerank" in m]
    if reranked:
        rerank = RerankComparison(
            questions=len(reranked),
            mean_recall_pre=_mean([m["rerank"]["pre_recall"] for m in reranked]),
            mean_recall_post=_mean([m["recall"] for m in reranked]),
            mean_mrr_pre=_mean([m["rerank"]["pre_mrr"] for m in reranked]),
            mean_mrr_post=_mean([m["mrr"] for m in reranked]),
            mean_ndcg_pre=_mean([m["rerank"]["pre_ndcg"] for m in reranked]),
            mean_ndcg_post=_mean([m["ndcg"] for m in reranked]),
        )

    channel_attribution = None
    channel_rows = [m["channels"] for m in parsed if "channels" in m]
    if channel_rows:
        names = {
            name
            for row in channel_rows
            for name in (*row["hits"], *row["found"])
        }
        total_hits = sum(row["hits_total"] for row in channel_rows)
        total_wanted = sum(row["wanted_total"] for row in channel_rows)
        channels: dict[str, ChannelCredit] = {}
        for name in sorted(names):
            hit_count = sum(
                row["hits"].get(name, 0) for row in channel_rows
            )
            found_count = sum(
                row["found"].get(name, 0) for row in channel_rows
            )
            channels[name] = ChannelCredit(
                hit_share=(hit_count / total_hits) if total_hits else None,
                raw_recall=(found_count / total_wanted)
                if total_wanted
                else None,
            )
        channel_attribution = ChannelAttributionSummary(
            questions=len(channel_rows), channels=channels
        )

    return EvalRunSummary(
        questions=len(results),
        mean_recall=_mean(recalls),
        mean_mrr=_mean(mrrs),
        mean_ndcg=_mean(ndcgs),
        doc_hit_rate=(sum(1 for h in doc_hits if h) / len(doc_hits))
        if doc_hits
        else None,
        rerank=rerank,
        channel_attribution=channel_attribution,
    )


# ---- run pipeline (worker side) ----


async def run_eval_pipeline(
    *,
    state: Any,
    run_id: str,
    emit: Any,
    job_id: str,
    should_cancel: Any,
) -> None:
    """Run the retrieval template over every question of the set.

    Questions come from the live set, or from a frozen version snapshot
    when the run stored ``version_id``.
    """
    corpus = state.corpus

    run = await run_in_sqlite(corpus.get_eval_run, run_id)
    if run is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_run_not_found",
                    "message": f"eval run {run_id!r} no longer exists",
                }
            },
        )
    params = json.loads(run["params_json"] or "{}")
    set_id = run["set_id"]

    eval_set = await run_in_sqlite(corpus.get_eval_set, set_id)
    if eval_set is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} no longer exists",
                }
            },
        )

    version_id = params.get("version_id")
    if version_id:
        version = await run_in_sqlite(corpus.get_eval_version, version_id)
        if version is None or version["set_id"] != set_id:
            raise HTTPException(
                404,
                detail={
                    "error": {
                        "code": "eval_version_not_found",
                        "message": (
                            f"version {version_id!r} does not exist for set "
                            f"{set_id!r}"
                        ),
                    }
                },
            )
        questions = _snapshot_questions(version["snapshot_json"])
    else:
        questions = await run_in_sqlite(
            corpus.list_eval_questions, set_id
        )
    if not questions:
        raise HTTPException(
            422,
            detail={
                "error": {
                    "code": "eval_set_empty",
                    "message": f"eval set {set_id!r} has no questions to run",
                }
            },
        )

    results = await _run_question_set(
        state=state,
        run_id=run_id,
        job_id=job_id,
        emit=emit,
        should_cancel=should_cancel,
        questions=questions,
        scope_database=eval_set["database"],
        scope_collection=eval_set["collection"],
        template=params.get("template", {}),
        include_answer=bool(params.get("include_answer")),
    )
    total = len(questions)
    await run_in_sqlite(
        corpus.finish_job,
        job_id,
        chunk_count=total,
        page_count=None,
        tokens_used=0,
    )
    log.info(
        "eval_run_done",
        job_id=job_id,
        run_id=run_id,
        questions=total,
        summary=_summarize(results).model_dump(),
    )


async def _run_question_set(
    *,
    state: Any,
    run_id: str,
    job_id: str,
    emit: Any,
    should_cancel: Any,
    questions: list[dict[str, Any]],
    scope_database: str,
    scope_collection: str,
    template: dict[str, Any],
    include_answer: bool,
) -> list[dict[str, Any]]:
    """Shared per-question loop for eval runs and gate checks.

    Marks the job evaluating, optionally prepares the answer LLM, runs
    the full pipeline per question with the template applied, and
    incrementally replaces the run results. Does not finish the job —
    callers own terminal state.
    """
    corpus = state.corpus
    await run_in_sqlite(corpus.mark_job, job_id, JOB_EVALUATING)
    await emit({"type": "stage", "stage": "evaluate"})

    chat_fn = None
    if include_answer:
        try:
            chat_client = await asyncio.to_thread(
                get_chat_client, state.settings
            )
        except ValueError as e:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "llm_unavailable",
                        "message": str(e),
                    }
                },
            ) from e
        chat_fn = chat_client.as_chat_fn()

    pipeline = _build_pipeline(state)
    total = len(questions)
    results: list[dict[str, Any]] = []

    for index, question_row in enumerate(questions, start=1):
        fresh = await run_in_sqlite(corpus.get_job, job_id)
        if fresh is not None and fresh["cancel_requested"]:
            await run_in_sqlite(corpus.mark_cancelled, job_id)
            raise JobCancelled(job_id)

        req = RetrievalRequest(
            database=scope_database,
            collection=scope_collection,
            query=question_row["question"],
            **template,
        )
        retrieval = await pipeline.retrieve(req)
        chunks = retrieval.chunks

        answer: str | None = None
        answer_error: str | None = None
        if include_answer:
            try:
                answer = await asyncio.to_thread(
                    _compose_answer,
                    chat_fn,
                    question_row["question"],
                    chunks,
                )
            except Exception as e:  # noqa: BLE001 — one bad answer must not kill the batch
                answer_error = str(e) or type(e).__name__
                log.warning(
                    "eval_answer_failed",
                    run_id=run_id,
                    question_id=question_row["question_id"],
                    error=answer_error,
                )

        expected_chunks = _loads(question_row["expected_chunk_ids"])
        expected_docs = _loads(question_row["expected_doc_ids"])
        metrics = _question_metrics(
            retrieval,
            expected_chunks=expected_chunks,
            expected_docs=expected_docs,
            top_k=req.top_k,
        )
        results.append({
            "question_id": question_row["question_id"],
            "chunk_ids": json.dumps(
                [c.chunk_id for c in chunks], ensure_ascii=False
            ),
            "metrics_json": json.dumps(metrics, ensure_ascii=False),
            "answer": answer,
            "answer_error": answer_error,
        })

        await run_in_sqlite(
            corpus.set_job_progress, job_id, index, total
        )
        await run_in_sqlite(corpus.replace_eval_results, run_id, results)

    return results


# ---- gate checks (worker side) ----


def _snapshot_questions(snapshot_json: str) -> list[dict[str, Any]]:
    """Parse a frozen version snapshot as question-shaped rows."""
    try:
        value = json.loads(snapshot_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _gate_report_metrics(summary: EvalRunSummary) -> dict[str, Any]:
    return {
        "recall": summary.mean_recall,
        "mrr": summary.mean_mrr,
        "ndcg": summary.mean_ndcg,
    }


def _evaluate_gate(
    gate: dict[str, Any],
    summary: EvalRunSummary,
    baseline: EvalRunSummary | None,
    *,
    candidate_ref: str,
) -> tuple[dict[str, Any], bool]:
    """Evaluate a check summary against the gate's thresholds.

    Absolute floors compare the summary directly; drop criteria compare
    it against the baseline summary. Returns ``(report, passed)``.
    """
    metrics = _gate_report_metrics(summary)
    violations: list[str] = []

    floors = (
        ("recall", gate["min_recall"]),
        ("mrr", gate["min_mrr"]),
        ("ndcg", gate["min_ndcg"]),
    )
    for name, floor in floors:
        if floor is None:
            continue
        value = metrics[name]
        if value is None or value < floor:
            violations.append(f"{name}_below_min")

    baseline_metrics = (
        _gate_report_metrics(baseline) if baseline is not None else None
    )
    deltas: dict[str, float] = {}
    drops = (
        ("recall", gate["max_recall_drop"]),
        ("mrr", gate["max_mrr_drop"]),
        ("ndcg", gate["max_ndcg_drop"]),
    )
    for name, max_drop in drops:
        if max_drop is None:
            continue
        value = metrics[name]
        base_value = baseline_metrics[name] if baseline_metrics else None
        if value is None or base_value is None:
            violations.append(f"{name}_unmeasured")
            continue
        delta = value - base_value
        deltas[name] = delta
        if delta < -max_drop:
            violations.append(f"{name}_drop_exceeds")

    passed = not violations
    report = {
        "candidate_ref": candidate_ref,
        "metrics": metrics,
        "baseline": baseline_metrics,
        "deltas": deltas,
        "violations": violations,
        "passed": passed,
    }
    return report, passed


async def run_gate_check(
    *,
    state: Any,
    gate_id: str,
    check_id: str,
    run_id: str,
    emit: Any,
    job_id: str,
    should_cancel: Any,
) -> None:
    """Run the frozen questions against the canary and evaluate the gate."""
    corpus = state.corpus

    gate = await run_in_sqlite(corpus.get_regression_gate, gate_id)
    if gate is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "gate_not_found",
                    "message": f"gate {gate_id!r} no longer exists",
                }
            },
        )

    binding = await run_in_sqlite(
        corpus.get_binding, gate["database"], gate["collection"]
    )
    candidate = binding["canary_ref"] if binding else None
    if candidate is None:
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "gate_no_canary",
                    "message": (
                        f"no canary parked for {gate['database']!r}/"
                        f"{gate['collection']!r}; submit a reindex first"
                    ),
                }
            },
        )

    version = await run_in_sqlite(
        corpus.get_eval_version, gate["version_id"]
    )
    if version is None or version["set_id"] != gate["set_id"]:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_version_not_found",
                    "message": (
                        f"version {gate['version_id']!r} no longer exists"
                    ),
                }
            },
        )
    questions = _snapshot_questions(version["snapshot_json"])
    if not questions:
        raise HTTPException(
            422,
            detail={
                "error": {
                    "code": "eval_set_empty",
                    "message": "gate version has no questions",
                }
            },
        )

    # Pin the check's runs to the canary, whatever the gate template says.
    template = json.loads(gate["template_json"] or "{}")
    template["index_ref"] = candidate
    run_params = {
        "template": template,
        "include_answer": False,
        "version_id": gate["version_id"],
    }
    # A retried check reuses its run: results are replaced wholesale as
    # the questions run, so partial rows from the dead attempt are harmless.
    if await run_in_sqlite(corpus.get_eval_run, run_id) is None:
        await run_in_sqlite(
            corpus.create_eval_run,
            run_id,
            set_id=gate["set_id"],
            params_json=json.dumps(run_params, ensure_ascii=False),
        )

    try:
        results = await _run_question_set(
            state=state,
            run_id=run_id,
            job_id=job_id,
            emit=emit,
            should_cancel=should_cancel,
            questions=questions,
            scope_database=gate["database"],
            scope_collection=gate["collection"],
            template=template,
            include_answer=False,
        )
    except JobCancelled:
        await run_in_sqlite(
            corpus.update_gate_check,
            check_id,
            status="cancelled",
            report_json=json.dumps(
                {"candidate_ref": candidate}, ensure_ascii=False
            ),
        )
        raise

    summary = _summarize(results)
    baseline = None
    if gate["baseline_run_id"]:
        baseline_rows = await run_in_sqlite(
            corpus.list_eval_results, gate["baseline_run_id"]
        )
        baseline = _summarize(baseline_rows)

    report, passed = _evaluate_gate(
        gate, summary, baseline, candidate_ref=candidate
    )
    await run_in_sqlite(
        corpus.update_gate_check,
        check_id,
        run_id=run_id,
        status="passed" if passed else "failed",
        report_json=json.dumps(report, ensure_ascii=False),
    )

    await run_in_sqlite(
        corpus.finish_job,
        job_id,
        chunk_count=len(questions),
        page_count=None,
        tokens_used=0,
    )
    log.info(
        "gate_check_done",
        job_id=job_id,
        gate_id=gate_id,
        run_id=run_id,
        passed=passed,
        violations=report["violations"],
    )


def _loads(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


# ---- routes: sets ----


@router.post(
    "/sets",
    response_model=EvalSetOut,
    status_code=201,
    summary="Create an evaluation set",
)
async def create_set(body: EvalSetCreate, request: Request) -> EvalSetOut:
    set_id = f"evs_{uuid.uuid4().hex}"
    await run_in_sqlite(
        request.app.state.corpus.create_eval_set,
        set_id,
        database=body.database,
        collection=body.collection,
        name=body.name,
        description=body.description,
    )
    row = await run_in_sqlite(
        request.app.state.corpus.get_eval_set, set_id
    )
    return set_out(row, question_count=0)


@router.get(
    "/sets",
    response_model=EvalSetListResponse,
    summary="List evaluation sets",
)
async def list_sets(
    request: Request,
    database: str | None = None,
    collection: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> EvalSetListResponse:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    rows, total = await run_in_sqlite(
        request.app.state.corpus.list_eval_sets,
        database=database,
        collection=collection,
        limit=limit,
        offset=offset,
    )
    items = [
        set_out(row, question_count=await _count_questions(request, row))
        for row in rows
    ]
    return EvalSetListResponse(
        items=items, total=total, limit=limit, offset=offset
    )


async def _count_questions(
    request: Request, set_row: dict[str, Any]
) -> int:
    questions = await run_in_sqlite(
        request.app.state.corpus.list_eval_questions, set_row["set_id"]
    )
    return len(questions)


@router.get(
    "/sets/{set_id}",
    response_model=EvalSetOut,
    summary="Get an evaluation set",
)
async def get_set(set_id: str, request: Request) -> EvalSetOut:
    row = await run_in_sqlite(
        request.app.state.corpus.get_eval_set, set_id
    )
    if row is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    return set_out(
        row, question_count=await _count_questions(request, row)
    )


@router.delete(
    "/sets/{set_id}",
    summary="Delete an evaluation set",
    description="Questions, runs and run results are deleted with the set.",
)
async def delete_set(set_id: str, request: Request) -> dict:
    row = await run_in_sqlite(
        request.app.state.corpus.get_eval_set, set_id
    )
    if row is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    await run_in_sqlite(
        request.app.state.corpus.delete_eval_set, set_id
    )
    return {"deleted": True, "set_id": set_id}


# ---- routes: questions ----


@router.post(
    "/sets/{set_id}/questions",
    response_model=EvalQuestionsResponse,
    status_code=201,
    summary="Add questions to an evaluation set",
)
async def add_questions(
    set_id: str, body: EvalQuestionsCreate, request: Request
) -> EvalQuestionsResponse:
    repo = request.app.state.corpus
    if await run_in_sqlite(repo.get_eval_set, set_id) is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    payload = [
        {
            "question_id": f"evq_{uuid.uuid4().hex}",
            "set_id": set_id,
            "question": item.question,
            "expected_chunk_ids": json.dumps(
                item.expected_chunk_ids, ensure_ascii=False
            ),
            "expected_doc_ids": json.dumps(
                item.expected_doc_ids, ensure_ascii=False
            ),
            "expected_answer": item.expected_answer,
        }
        for item in body.questions
    ]
    await run_in_sqlite(repo.add_eval_questions, payload)
    rows = await run_in_sqlite(repo.list_eval_questions, set_id)
    return EvalQuestionsResponse(
        set_id=set_id,
        items=[question_out(row) for row in rows],
    )


@router.get(
    "/sets/{set_id}/questions",
    response_model=EvalQuestionsResponse,
    summary="List questions of an evaluation set",
)
async def list_questions(
    set_id: str, request: Request
) -> EvalQuestionsResponse:
    repo = request.app.state.corpus
    if await run_in_sqlite(repo.get_eval_set, set_id) is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    rows = await run_in_sqlite(repo.list_eval_questions, set_id)
    return EvalQuestionsResponse(
        set_id=set_id,
        items=[EvalQuestionOut(**question_out(row).model_dump()) for row in rows],
    )


# ---- routes: versions ----


@router.post(
    "/sets/{set_id}/versions",
    response_model=VersionOut,
    status_code=201,
    summary="Freeze the set's questions into an immutable version",
)
async def create_version(
    set_id: str, body: VersionCreate, request: Request
) -> VersionOut:
    repo = request.app.state.corpus
    if await run_in_sqlite(repo.get_eval_set, set_id) is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    if (
        await run_in_sqlite(repo.get_eval_version_by_tag, set_id, body.tag)
    ) is not None:
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "version_tag_exists",
                    "message": (
                        f"version tag {body.tag!r} already exists for set "
                        f"{set_id!r}"
                    ),
                }
            },
        )

    questions = await run_in_sqlite(repo.list_eval_questions, set_id)
    if not questions:
        raise HTTPException(
            422,
            detail={
                "error": {
                    "code": "eval_set_empty",
                    "message": "cannot freeze a set with no questions",
                }
            },
        )
    snapshot = [
        {
            "question_id": q["question_id"],
            "question": q["question"],
            "expected_chunk_ids": q["expected_chunk_ids"],
            "expected_doc_ids": q["expected_doc_ids"],
            "expected_answer": q["expected_answer"],
        }
        for q in questions
    ]
    version_id = f"evv_{uuid.uuid4().hex}"
    await run_in_sqlite(
        repo.create_eval_version,
        version_id,
        set_id=set_id,
        tag=body.tag,
        question_count=len(questions),
        snapshot_json=json.dumps(snapshot, ensure_ascii=False),
    )
    row = await run_in_sqlite(repo.get_eval_version, version_id)
    return VersionOut(
        version_id=row["version_id"],
        set_id=row["set_id"],
        tag=row["tag"],
        question_count=row["question_count"],
        created_ts=float(row["created_ts"]),
    )


@router.get(
    "/sets/{set_id}/versions",
    response_model=VersionListResponse,
    summary="List frozen versions of an evaluation set",
)
async def list_versions(
    set_id: str, request: Request
) -> VersionListResponse:
    repo = request.app.state.corpus
    if await run_in_sqlite(repo.get_eval_set, set_id) is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    rows = await run_in_sqlite(repo.list_eval_versions, set_id)
    return VersionListResponse(
        set_id=set_id,
        items=[
            VersionOut(
                version_id=r["version_id"],
                set_id=r["set_id"],
                tag=r["tag"],
                question_count=r["question_count"],
                created_ts=float(r["created_ts"]),
            )
            for r in rows
        ],
    )


# ---- routes: runs ----


@router.post(
    "/sets/{set_id}/runs",
    response_model=EvalRunSubmitResponse,
    status_code=202,
    summary="Submit a batch retrieval/generation run",
    description=(
        "Runs the retrieval template over every question, records ranked "
        "chunk ids, Recall/MRR and expected-doc hits, and optionally "
        "generates an answer per question."
    ),
)
async def submit_run(
    set_id: str, body: EvalRunCreate, request: Request
) -> EvalRunSubmitResponse:
    repo = request.app.state.corpus
    eval_set = await run_in_sqlite(repo.get_eval_set, set_id)
    if eval_set is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    version = None
    if body.version_id is not None:
        version = await run_in_sqlite(
            repo.get_eval_version, body.version_id
        )
        if version is None or version["set_id"] != set_id:
            raise HTTPException(
                404,
                detail={
                    "error": {
                        "code": "eval_version_not_found",
                        "message": (
                            f"version {body.version_id!r} does not exist for "
                            f"set {set_id!r}"
                        ),
                    }
                },
            )
        if not _snapshot_questions(version["snapshot_json"]):
            raise HTTPException(
                422,
                detail={
                    "error": {
                        "code": "eval_set_empty",
                        "message": (
                            f"version {body.version_id!r} has no questions "
                            "to run"
                        ),
                    }
                },
            )
    elif not await run_in_sqlite(repo.list_eval_questions, set_id):
        raise HTTPException(
            422,
            detail={
                "error": {
                    "code": "eval_set_empty",
                    "message": f"eval set {set_id!r} has no questions to run",
                }
            },
        )
    if body.include_answer and not is_llm_configured(request.app.state.settings):
        raise HTTPException(
            503,
            detail={
                "error": {
                    "code": "llm_unavailable",
                    "message": (
                        "include_answer needs an LLM; set VS_LLM__BASE_URL "
                        "and VS_LLM__MODEL"
                    ),
                }
            },
        )

    run_id = f"evr_{uuid.uuid4().hex}"
    job_id = uuid.uuid4().hex
    params = {
        "template": body.template.model_dump(),
        "include_answer": body.include_answer,
        "version_id": body.version_id,
    }
    await run_in_sqlite(
        repo.create_eval_run,
        run_id,
        set_id=set_id,
        params_json=json.dumps(params, ensure_ascii=False),
    )
    await run_in_sqlite(
        repo.create_job,
        job_id,
        job_type="eval_run",
        database=eval_set["database"],
        collection=eval_set["collection"],
        filename=None,
        mime=None,
        params_json=json.dumps({"run_id": run_id}),
        max_attempts=request.app.state.settings.jobs.max_attempts,
    )

    worker = getattr(request.app.state, "job_worker", None)
    if worker is not None and request.app.state.settings.jobs.wake_on_submit:
        worker.wake()

    return EvalRunSubmitResponse(run_id=run_id, job_id=job_id)


@router.get(
    "/sets/{set_id}/runs",
    summary="List runs of an evaluation set",
)
async def list_runs(set_id: str, request: Request) -> dict:
    repo = request.app.state.corpus
    if await run_in_sqlite(repo.get_eval_set, set_id) is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_set_not_found",
                    "message": f"eval set {set_id!r} does not exist",
                }
            },
        )
    rows = await run_in_sqlite(repo.list_eval_runs, set_id)
    return {"set_id": set_id, "items": rows}


@router.get(
    "/runs/{run_id}",
    response_model=EvalRunOut,
    summary="Get a run with per-question results",
)
async def get_run(run_id: str, request: Request) -> EvalRunOut:
    repo = request.app.state.corpus
    run = await run_in_sqlite(repo.get_eval_run, run_id)
    if run is None:
        raise HTTPException(
            404,
            detail={
                "error": {
                    "code": "eval_run_not_found",
                    "message": f"eval run {run_id!r} does not exist",
                }
            },
        )
    params = json.loads(run["params_json"] or "{}")
    rows = await run_in_sqlite(repo.list_eval_results, run_id)
    questions = {
        q["question_id"]: q["question"]
        for q in await run_in_sqlite(
            repo.list_eval_questions, run["set_id"]
        )
    }
    version_id = params.get("version_id")
    if version_id is not None:
        # The live questions may have changed since the run; the frozen
        # snapshot owns the wording for versioned runs.
        version = await run_in_sqlite(
            repo.get_eval_version, version_id
        )
        if version is not None and version["set_id"] == run["set_id"]:
            for snap in _snapshot_questions(version["snapshot_json"]):
                questions.setdefault(snap["question_id"], snap["question"])
    results = [
        result_out(row, question=questions.get(row["question_id"], ""))
        for row in rows
    ]
    return EvalRunOut(
        run_id=run["run_id"],
        set_id=run["set_id"],
        template=params.get("template", {}),
        include_answer=bool(params.get("include_answer")),
        version_id=version_id,
        created_ts=float(run["created_ts"]),
        summary=_summarize(rows),
        results=results,
    )


# ---- routes: regression gates ----


def _gate_error(code: str, message: str, status: int) -> HTTPException:
    return HTTPException(
        status,
        detail={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


@router.post(
    "/gates",
    response_model=GateListResponse,
    status_code=201,
    summary="Configure the regression gate for a (database, collection)",
)
async def create_gate(
    body: GateCreate, request: Request
) -> GateListResponse:
    repo = request.app.state.corpus
    eval_set = await run_in_sqlite(repo.get_eval_set, body.set_id)
    if eval_set is None:
        raise _gate_error(
            "eval_set_not_found",
            f"eval set {body.set_id!r} does not exist",
            404,
        )
    version = await run_in_sqlite(
        repo.get_eval_version, body.version_id
    )
    if version is None or version["set_id"] != body.set_id:
        raise _gate_error(
            "eval_version_not_found",
            (
                f"version {body.version_id!r} does not exist for set "
                f"{body.set_id!r}"
            ),
            404,
        )
    if (
        await run_in_sqlite(
            repo.get_gate_for_collection, body.database, body.collection
        )
    ) is not None:
        raise _gate_error(
            "gate_exists",
            (
                f"a gate already exists for {body.database!r}/"
                f"{body.collection!r}; delete it before reconfiguring"
            ),
            409,
        )

    gate_id = f"gat_{uuid.uuid4().hex}"
    now = None
    await run_in_sqlite(
        repo.create_regression_gate,
        gate_id,
        database=body.database,
        collection=body.collection,
        set_id=body.set_id,
        version_id=body.version_id,
        template_json=json.dumps(
            body.template.model_dump(), ensure_ascii=False
        ),
        min_recall=body.min_recall,
        min_mrr=body.min_mrr,
        min_ndcg=body.min_ndcg,
        max_recall_drop=body.max_recall_drop,
        max_mrr_drop=body.max_mrr_drop,
        max_ndcg_drop=body.max_ndcg_drop,
        baseline_run_id=body.baseline_run_id,
        now=now,
    )
    row = await run_in_sqlite(repo.get_regression_gate, gate_id)
    out = gate_out(row)
    return GateListResponse(items=[out], total=1)


@router.get(
    "/gates",
    response_model=GateListResponse,
    summary="List regression gates",
)
async def list_gates(
    request: Request,
    database: str | None = None,
    collection: str | None = None,
) -> GateListResponse:
    rows = await run_in_sqlite(
        request.app.state.corpus.list_regression_gates,
        database=database,
        collection=collection,
    )
    return GateListResponse(
        items=[gate_out(row) for row in rows], total=len(rows)
    )


@router.get(
    "/gates/{gate_id}",
    response_model=GateListResponse,
    summary="Get a regression gate",
)
async def get_gate(gate_id: str, request: Request) -> GateListResponse:
    row = await run_in_sqlite(
        request.app.state.corpus.get_regression_gate, gate_id
    )
    if row is None:
        raise _gate_error(
            "gate_not_found", f"gate {gate_id!r} does not exist", 404
        )
    return GateListResponse(items=[gate_out(row)], total=1)


@router.delete(
    "/gates/{gate_id}",
    summary="Delete a regression gate and its checks",
)
async def delete_gate(gate_id: str, request: Request) -> dict:
    repo = request.app.state.corpus
    if (
        await run_in_sqlite(repo.get_regression_gate, gate_id) is None
    ):
        raise _gate_error(
            "gate_not_found", f"gate {gate_id!r} does not exist", 404
        )
    await run_in_sqlite(repo.delete_regression_gate, gate_id)
    return {"deleted": True, "gate_id": gate_id}


@router.post(
    "/gates/{gate_id}/checks",
    response_model=GateCheckSubmitResponse,
    status_code=202,
    summary="Run the gate's frozen questions against the parked canary",
    description=(
        "Pins the retrieval template to the canary physical index, runs "
        "every frozen question, evaluates absolute floors and max drops "
        "against the baseline run, and records passed/failed with a "
        "detailed report. A passed check is required to promote the canary."
    ),
)
async def submit_gate_check(
    gate_id: str, request: Request
) -> GateCheckSubmitResponse:
    repo = request.app.state.corpus
    gate = await run_in_sqlite(repo.get_regression_gate, gate_id)
    if gate is None:
        raise _gate_error(
            "gate_not_found", f"gate {gate_id!r} does not exist", 404
        )

    binding = await run_in_sqlite(
        repo.get_binding, gate["database"], gate["collection"]
    )
    candidate = binding["canary_ref"] if binding else None
    if candidate is None:
        raise _gate_error(
            "gate_no_canary",
            (
                f"no canary parked for {gate['database']!r}/"
                f"{gate['collection']!r}; submit a reindex first"
            ),
            409,
        )

    check_id = f"evc_{uuid.uuid4().hex}"
    run_id = f"evr_{uuid.uuid4().hex}"
    job_id = uuid.uuid4().hex
    await run_in_sqlite(
        repo.create_gate_check,
        check_id,
        gate_id=gate_id,
        run_id=None,
        candidate_ref=candidate,
        status="running",
    )
    await run_in_sqlite(
        repo.create_job,
        job_id,
        job_type="gate_check",
        database=gate["database"],
        collection=gate["collection"],
        filename=None,
        mime=None,
        params_json=json.dumps(
            {
                "gate_id": gate_id,
                "check_id": check_id,
                "run_id": run_id,
            }
        ),
        max_attempts=request.app.state.settings.jobs.max_attempts,
    )

    worker = getattr(request.app.state, "job_worker", None)
    if worker is not None and request.app.state.settings.jobs.wake_on_submit:
        worker.wake()

    return GateCheckSubmitResponse(
        check_id=check_id, run_id=run_id, job_id=job_id
    )


@router.get(
    "/gates/{gate_id}/checks",
    response_model=GateCheckListResponse,
    summary="List checks of a regression gate",
)
async def list_gate_checks(
    gate_id: str, request: Request
) -> GateCheckListResponse:
    repo = request.app.state.corpus
    if (
        await run_in_sqlite(repo.get_regression_gate, gate_id) is None
    ):
        raise _gate_error(
            "gate_not_found", f"gate {gate_id!r} does not exist", 404
        )
    rows = await run_in_sqlite(repo.list_gate_checks, gate_id)
    return GateCheckListResponse(
        gate_id=gate_id, items=[check_out(row) for row in rows]
    )
