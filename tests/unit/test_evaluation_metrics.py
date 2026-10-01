"""Evaluation metric functions: ranking, attribution, gate verdict, summary.

Pins the §5 contracts:

- ``_ranking_metrics`` binary-relevance Recall/MRR/nDCG;
- ``_question_metrics`` final-order scoring + pre-rerank comparison +
  channel attribution + expected-doc hit;
- ``_evaluate_gate`` absolute floors and max-drop criteria;
- ``_summarize`` run-level means and aggregate attribution.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from vector_service.api.evaluation import (
    _channel_attribution,
    _evaluate_gate,
    _question_metrics,
    _ranking_metrics,
    _summarize,
)
from vector_service.schemas.evaluation import EvalRunSummary

# ---- ranking metrics ----


def test_ranking_metrics_perfect_partial_positions():
    recall, mrr, ndcg = _ranking_metrics(
        ["a", "b", "c"], {"a", "c"}
    )
    assert recall == 1.0
    assert mrr == 1.0
    # DCG = 1/log2(2) + 1/log2(4) = 1.5
    # IDCG = 1/log2(2) + 1/log2(3)
    expected_ndcg = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert ndcg == pytest.approx(expected_ndcg, abs=1e-9)


def test_ranking_metrics_no_overlap():
    recall, mrr, ndcg = _ranking_metrics(["x", "y"], {"a"})
    assert (recall, mrr, ndcg) == (0.0, 0.0, 0.0)


def test_ranking_metrics_mrr_follows_first_hit():
    # "b" is noise; the only wanted chunk "a" sits at position 2.
    recall, mrr, ndcg = _ranking_metrics(["b", "a"], {"a"})
    assert recall == 1.0
    assert mrr == 0.5
    assert ndcg == pytest.approx(1.0 / math.log2(3))


def test_ranking_metrics_empty_id_list():
    recall, mrr, ndcg = _ranking_metrics([], {"a"})
    assert (recall, mrr, ndcg) == (0.0, 0.0, 0.0)


# ---- per-question metrics + attribution ----


def _retrieval(*, final_ids, rerank_input_ids=None):
    def _chunk(chunk_id):
        return SimpleNamespace(
            chunk_id=chunk_id, fields={"doc_id": "d1"},
            matched_channels=["dense", "bm25"] if chunk_id == "a"
            else ["dense"],
        )

    return SimpleNamespace(
        chunks=[_chunk(cid) for cid in final_ids],
        channel_runs=[
            SimpleNamespace(channel="dense", hits=[
                SimpleNamespace(chunk_id="a"),
                SimpleNamespace(chunk_id="b"),
            ]),
            SimpleNamespace(channel="bm25", hits=[
                SimpleNamespace(chunk_id="a"),
            ]),
        ],
        rerank_input_ids=rerank_input_ids,
    )


def test_question_metrics_scores_final_and_prev_rerank():
    retrieval = _retrieval(final_ids=["a", "b"],
                           rerank_input_ids=["b", "a"])
    metrics = _question_metrics(
        retrieval, expected_chunks=["a", "c"],
        expected_docs=["d1"], top_k=10,
    )
    assert metrics["recall"] == 0.5  # one of two wanted
    assert metrics["mrr"] == 1.0     # a ranks first in the final list
    assert metrics["doc_hit"] is True

    rerank = metrics["rerank"]
    assert rerank["pre_recall"] == 0.5
    assert rerank["pre_mrr"] == 0.5  # pre order b,a: a is second
    assert rerank["pre_ndcg"] < metrics["ndcg"]

    channels = metrics["channels"]
    assert channels["hits"] == {"bm25": 1, "dense": 1}
    assert channels["found"] == {"bm25": 1, "dense": 1}
    assert channels["hits_total"] == 1
    assert channels["wanted_total"] == 2


def test_question_metrics_without_rerank_has_no_rerank_block():
    metrics = _question_metrics(
        _retrieval(final_ids=["a"]),
        expected_chunks=["a"], expected_docs=[], top_k=10,
    )
    assert "rerank" not in metrics
    assert "channels" in metrics


def test_question_metrics_doc_miss():
    metrics = _question_metrics(
        _retrieval(final_ids=["a"]),
        expected_chunks=[], expected_docs=["d9"], top_k=10,
    )
    assert metrics["doc_hit"] is False


def test_channel_attribution_direct():
    retrieval = _retrieval(final_ids=["a", "b"])
    attribution = _channel_attribution(retrieval, {"a", "c"}, ["a", "b"])
    assert attribution["hits"] == {"bm25": 1, "dense": 1}
    # Dense raw runs surfaced b too, but only a is wanted: one credit.
    assert attribution["found"] == {"bm25": 1, "dense": 1}
    assert attribution["hits_total"] == 1
    assert attribution["wanted_total"] == 2


# ---- gate evaluation ----


def _gate(**criteria):
    base = {
        "min_recall": None, "min_mrr": None, "min_ndcg": None,
        "max_recall_drop": None, "max_mrr_drop": None,
        "max_ndcg_drop": None,
    }
    base.update(criteria)
    return base


def _summary(recall=0.9, mrr=0.8, ndcg=0.7):
    return EvalRunSummary(
        questions=1, mean_recall=recall, mean_mrr=mrr, mean_ndcg=ndcg,
        doc_hit_rate=None,
    )


def test_gate_passes_above_floors():
    report, passed = _evaluate_gate(
        _gate(min_recall=0.8, min_mrr=0.7),
        _summary(), None, candidate_ref="coll__rebuild_abc",
    )
    assert passed is True
    assert report["violations"] == []
    assert report["candidate_ref"] == "coll__rebuild_abc"
    assert report["metrics"] == {"recall": 0.9, "mrr": 0.8, "ndcg": 0.7}


def test_gate_fails_below_floor():
    report, passed = _evaluate_gate(
        _gate(min_recall=0.95), _summary(), None, candidate_ref="c",
    )
    assert passed is False
    assert report["violations"] == ["recall_below_min"]


def test_gate_unmeasured_metric_fails_floor():
    report, passed = _evaluate_gate(
        _gate(min_recall=0.5), _summary(recall=None), None,
        candidate_ref="c",
    )
    assert passed is False
    assert report["violations"] == ["recall_below_min"]


def test_gate_drop_within_tolerance_passes():
    report, passed = _evaluate_gate(
        _gate(max_recall_drop=0.2),
        _summary(recall=0.8), _summary(recall=0.9),
        candidate_ref="c",
    )
    assert passed is True
    assert report["deltas"] == {"recall": pytest.approx(-0.1)}


def test_gate_drop_exceeds_tolerance_fails():
    report, passed = _evaluate_gate(
        _gate(max_recall_drop=0.05),
        _summary(recall=0.8), _summary(recall=0.9),
        candidate_ref="c",
    )
    assert passed is False
    assert report["violations"] == ["recall_drop_exceeds"]


def test_gate_drop_unmeasured_without_baseline():
    report, passed = _evaluate_gate(
        _gate(max_recall_drop=0.1), _summary(recall=0.9), None,
        candidate_ref="c",
    )
    assert passed is False
    assert report["violations"] == ["recall_unmeasured"]


# ---- run summary ----


def test_summarize_aggregates_means_and_attribution():
    rows = [
        {
            "metrics_json": json.dumps({
                "recall": 1.0, "mrr": 1.0, "ndcg": 1.0,
                "doc_hit": True,
                "rerank": {"pre_recall": 0.5, "pre_mrr": 0.5,
                           "pre_ndcg": 0.5},
                "channels": {"hits": {"dense": 1}, "found": {"dense": 1},
                             "hits_total": 1, "wanted_total": 1},
            }),
        },
        {
            "metrics_json": json.dumps({
                "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
                "doc_hit": False,
                "channels": {"hits": {}, "found": {"bm25": 1},
                             "hits_total": 0, "wanted_total": 1},
            }),
        },
    ]
    summary = _summarize(rows)
    assert summary.questions == 2
    assert summary.mean_recall == 0.5
    assert summary.mean_mrr == 0.5
    assert summary.doc_hit_rate == 0.5

    assert summary.rerank is not None
    assert summary.rerank.questions == 1
    assert summary.rerank.mean_recall_pre == 0.5
    assert summary.rerank.mean_recall_post == 1.0

    attribution = summary.channel_attribution
    assert attribution.questions == 2
    dense = attribution.channels["dense"]
    bm25 = attribution.channels["bm25"]
    assert dense.hit_share == 1.0
    assert dense.raw_recall == 0.5
    # bm25 never sat on a final-list hit: zero share of the one hit
    # (share is None only when there were no hits at all).
    assert bm25.hit_share == 0.0
    assert bm25.raw_recall == 0.5


def test_summarize_empty_results():
    summary = _summarize([])
    assert summary.questions == 0
    assert summary.mean_recall is None
    assert summary.doc_hit_rate is None
    assert summary.rerank is None
    assert summary.channel_attribution is None
