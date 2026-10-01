"""IngestWorker dispatch for ``eval_run`` / ``gate_check`` jobs.

Reuses the builders from test_evaluation_pipeline (real corpus,
in-memory store/model fakes) and invokes ``worker._execute`` directly.
Pins the §5 contracts:

- eval_run rows drive run_eval_pipeline to a done job;
- gate_check rows drive run_gate_check to a done job;
- unreadable params fail the row as ``job_corrupted`` with no attempt;
- a 4xx from the pipeline (gate without canary) fails immediately.
"""
from __future__ import annotations

import asyncio
import functools
import json
from types import SimpleNamespace

from tests.unit.test_evaluation_pipeline import (
    _CANARY,
    _make_gate,
    _make_set,
    _seeded_corpus,
    _state,
)
from vector_service.jobs import IngestWorker


def async_test(coro):
    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


def _worker(state):
    app = SimpleNamespace(state=state)
    return IngestWorker(app)


# ---- eval_run dispatch --------------------------------------------------


@async_test
async def test_worker_eval_run_dispatches_to_pipeline(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    template = {"rerank": {"enabled": False}}
    repo.create_job(
        "job_1", database="default", collection="ingest",
        filename=None, mime=None, job_type="eval_run",
        params_json=json.dumps({"run_id": "evr_1"}),
    )
    repo.create_eval_run(
        "evr_1", set_id="evs_1",
        params_json=json.dumps({"template": template}),
    )
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])
    worker = _worker(state)

    await worker._execute(repo.get_job("job_1"))

    assert repo.get_job("job_1")["status"] == "done"
    assert len(repo.list_eval_results("evr_1")) == 1


@async_test
async def test_worker_eval_run_corrupted_params_fails_job(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    repo.create_job(
        "job_1", database="default", collection="ingest",
        filename=None, mime=None, job_type="eval_run",
        params_json="not-json",
    )
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])
    worker = _worker(state)

    await worker._execute(repo.get_job("job_1"))

    row = repo.get_job("job_1")
    assert row["status"] == "failed"
    assert row["error_code"] == "job_corrupted"
    assert row["attempts"] == 0


# ---- gate_check dispatch ------------------------------------------------


@async_test
async def test_worker_gate_check_dispatches_to_pipeline(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    repo.set_canary(
        "default", "ingest", canary_ref=_CANARY, canary_percent=10,
        model="fake",
    )
    _make_gate(repo, min_recall=0.8)
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])
    worker = _worker(state)

    await worker._execute(repo.get_job("job_1"))

    assert repo.get_job("job_1")["status"] == "done"
    check = repo.get_latest_gate_check("gat_1")
    assert check["status"] == "passed"


@async_test
async def test_worker_gate_check_without_canary_fails_4xx(tmp_path):
    repo = _seeded_corpus(tmp_path)
    _make_set(repo, questions=[{"question": "q1", "chunks": ["d1_0"]}])
    # _make_gate prepares gate + check + job, but no canary is parked.
    _make_gate(repo, min_recall=0.8)
    state = _state(repo, dense_hits=[("d1_0", 0.9, {})])
    worker = _worker(state)

    await worker._execute(repo.get_job("job_1"))

    row = repo.get_job("job_1")
    assert row["status"] == "failed"
    assert row["attempts"] == 1  # 4xx: no retry
    assert row["error_code"] == "gate_no_canary"
