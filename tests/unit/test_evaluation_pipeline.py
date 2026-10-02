"""Worker-side evaluation pipelines: run_eval_pipeline / run_gate_check.

Real :class:`CorpusRepository` (jobs, runs, results, bindings, checks)
combined with the same in-memory store/model fakes used by the
retrieval pipeline tests. Pins the §5 contracts:

- eval runs score the live question set and finish the job;
- cancellation mid-batch marks the job cancelled;
- gate checks pin retrieval to the canary via ``index_ref``;
- the check row ends passed/failed with a full report;
- a gate with no parked canary fails fast (409).
"""
from __future__ import annotations

import asyncio
import functools
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tests.unit.test_retrieval_pipeline_advanced import (
    FakeBM25,
    FakeEmbedder,
    FakeSettings,
    FakeStore,
)
from vector_service.api.evaluation import run_eval_pipeline, run_gate_check
from vector_service.api.ingest import JobCancelled
from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)

_CANARY = "ingest__rebuild_aaaaaaaaaaaa"
_TEMPLATE = {"rerank": {"enabled": False}}


def async_test(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


# ---- fixtures / builders ------------------------------------------------


def _seeded_corpus(tmp_path):
    repo = CorpusRepository(tmp_path / "corpus.db")
    repo.initialize()
    doc = DocumentRecord(
        doc_id="d1", database="default", collection="ingest",
        filename="年报.pdf", mime="application/pdf",
        content_hash="a" * 64, title="年度报告", author="张三", page_count=10,
    )
    chunks = [
        ChunkRecord(chunk_id="d1_0", doc_id="d1", database="default",
                    collection="ingest", chunk_index=0, text="第一段内容",
                    section_header="引言", page_number=1,
                    char_start=0, char_end=50, token_count=5),
        ChunkRecord(chunk_id="d1_1", doc_id="d1", database="default",
                    collection="ingest", chunk_index=1, text="第二段内容",
                    section_header="正文", page_number=2,
                    char_start=50, char_end=100, token_count=5),
    ]
    repo.store_document(doc, chunks)
    return repo


def _make_set(repo, *, questions):
    repo.create_eval_set("evs_1", database="default", collection="ingest",
                         name="s")
    repo.add_eval_questions([
        {
            "question_id": f"eq_{i + 1}", "set_id": "evs_1",
            "question": item["question"],
            "expected_chunk_ids": json.dumps(item.get("chunks", [])),
            "expected_doc_ids": json.dumps(item.get("docs", [])),
            "expected_answer": None,
        }
        for i, item in enumerate(questions)
    ])


def _state(repo, *, dense_hits):
    store = FakeStore(
        info_fields=[{"name": "id"}, {"name": "text"}, {"name": "sparse"}],
        dense_hits=dense_hits,
    )
    return SimpleNamespace(
        corpus=repo, settings=FakeSettings(), store=store,
        embedder=FakeEmbedder(), reranker=None, bm25=FakeBM25(),
    )


async def _no_emit(_event):
    pass


def _never_cancel():
    async def should_cancel():
        return False

    return should_cancel


# ---- run_eval_pipeline --------------------------------------------------


@async_test
async def test_run_eval_pipeline_scores_questions_and_finishes_job(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[
        {"question": "营收？", "chunks": ["d1_0"]},
    ])
    repo.create_job(
        "job_1", database="default", collection="ingest",
        filename=None, mime=None, job_type="eval_run",
        params_json=json.dumps({"run_id": "evr_1", "template": _TEMPLATE}),
    )
    repo.create_eval_run(
        "evr_1", set_id="evs_1",
        params_json=json.dumps({"template": _TEMPLATE}),
    )
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])

    await run_eval_pipeline(
        state=state, run_id="evr_1", emit=_no_emit, job_id="job_1",
        should_cancel=_never_cancel(),
    )

    job = repo.get_job("job_1")
    assert job["status"] == "done"
    rows = repo.list_eval_results("evr_1")
    assert len(rows) == 1
    metrics = json.loads(rows[0]["metrics_json"])
    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0


@async_test
async def test_run_eval_pipeline_cancel_mid_batch(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[
        {"question": "q1", "chunks": ["d1_0"]},
        {"question": "q2", "chunks": ["d1_1"]},
    ])
    repo.create_job(
        "job_1", database="default", collection="ingest",
        filename=None, mime=None, job_type="eval_run",
        params_json=json.dumps({"run_id": "evr_1", "template": _TEMPLATE}),
    )
    repo.create_eval_run(
        "evr_1", set_id="evs_1",
        params_json=json.dumps({"template": _TEMPLATE}),
    )
    repo.request_cancel("job_1")
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])

    with pytest.raises(JobCancelled):
        await run_eval_pipeline(
            state=state, run_id="evr_1", emit=_no_emit, job_id="job_1",
            should_cancel=_never_cancel(),
        )
    assert repo.get_job("job_1")["status"] == "cancelled"


# ---- run_gate_check -----------------------------------------------------


def _make_gate(repo, *, min_recall):
    questions = repo.list_eval_questions("evs_1")
    snapshot = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "expected_chunk_ids": row["expected_chunk_ids"],
            "expected_doc_ids": row["expected_doc_ids"],
        }
        for row in questions
    ]
    repo.create_eval_version(
        "evv_1", set_id="evs_1", tag="v1",
        question_count=len(snapshot),
        snapshot_json=json.dumps(snapshot, ensure_ascii=False),
    )
    repo.create_regression_gate(
        "gat_1", database="default", collection="ingest",
        set_id="evs_1", version_id="evv_1",
        template_json=json.dumps(_TEMPLATE), min_recall=min_recall,
    )
    repo.create_job(
        "job_1", database="default", collection="ingest",
        filename=None, mime=None, job_type="gate_check",
        params_json=json.dumps({
            "gate_id": "gat_1", "check_id": "chk_1", "run_id": "evr_1",
        }),
    )
    repo.create_gate_check(
        "chk_1", gate_id="gat_1", run_id=None,
        candidate_ref=_CANARY, status="running",
    )


@async_test
async def test_run_gate_check_passes_and_pins_canary(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    repo.set_canary(
        "default", "ingest", canary_ref=_CANARY, canary_percent=10,
        model="fake",
    )
    _make_gate(repo, min_recall=0.8)
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])

    await run_gate_check(
        state=state, gate_id="gat_1", check_id="chk_1", run_id="evr_1",
        emit=_no_emit, job_id="job_1", should_cancel=_never_cancel(),
    )

    check = repo.get_latest_gate_check("gat_1")
    assert check["status"] == "passed"
    report = json.loads(check["report_json"])
    assert report["candidate_ref"] == _CANARY
    assert report["passed"] is True
    assert report["violations"] == []
    # The run was created by the gate flow and pinned to the canary.
    run = repo.get_eval_run("evr_1")
    run_template = json.loads(run["params_json"])["template"]
    assert run_template["index_ref"] == _CANARY
    # Recall really ran against the canary physical collection.
    searched = {call["collection"] for call in state.store.search_calls}
    assert searched == {_CANARY}


@async_test
async def test_run_gate_check_fails_below_floor(tmp_path):
    repo = _seeded_corpus(tmp_path)
    # Expectation is a chunk the dense leg does not surface.
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_9"]}])
    repo.set_canary(
        "default", "ingest", canary_ref=_CANARY, canary_percent=10,
        model="fake",
    )
    _make_gate(repo, min_recall=0.8)
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])

    await run_gate_check(
        state=state, gate_id="gat_1", check_id="chk_1", run_id="evr_1",
        emit=_no_emit, job_id="job_1", should_cancel=_never_cancel(),
    )

    check = repo.get_latest_gate_check("gat_1")
    assert check["status"] == "failed"
    report = json.loads(check["report_json"])
    assert report["violations"] == ["recall_below_min"]


@async_test
async def test_run_gate_check_no_canary_raises_409(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    _make_gate(repo, min_recall=0.8)
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])

    with pytest.raises(HTTPException) as exc_info:
        await run_gate_check(
            state=state, gate_id="gat_1", check_id="chk_1", run_id="evr_1",
            emit=_no_emit, job_id="job_1", should_cancel=_never_cancel(),
        )
    assert exc_info.value.status_code == 409
