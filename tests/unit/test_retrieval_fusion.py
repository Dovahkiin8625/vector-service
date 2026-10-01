"""Rank-fusion pure functions: RRF and per-channel weighted fusion."""
from __future__ import annotations

from vector_service.retrieval.base import ChannelHit, ChannelRun
from vector_service.retrieval.fusion import rrf_fuse, weighted_fuse


def _run(channel, pairs):
    """pairs: list of (chunk_id, score, fields)."""
    return ChannelRun(
        channel=channel,
        query=f"{channel}-q",
        hits=[
            ChannelHit(chunk_id=cid, score=s, rank=i + 1, fields=f or {})
            for i, (cid, s, f) in enumerate(pairs)
        ],
    )


def test_rrf_single_run_scores():
    out = rrf_fuse([_run("dense", [("c1", 0.9, None), ("c2", 0.8, None)])])
    assert [c.chunk_id for c in out] == ["c1", "c2"]
    assert out[0].fusion_score == 1.0 / 61
    assert out[1].fusion_score == 1.0 / 62
    assert out[0].matched_channels == ["dense"]


def test_rrf_combines_two_runs_dedup():
    out = rrf_fuse([
        _run("dense", [("c1", 0.9, {"text": "a"}), ("c2", 0.8, None)]),
        _run("bm25", [("c2", 5.0, None), ("c3", 3.0, None)]),
    ])
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c2"] == 1.0 / 62 + 1.0 / 61
    assert scores["c1"] == 1.0 / 61
    assert scores["c3"] == 1.0 / 62
    c2 = next(c for c in out if c.chunk_id == "c2")
    assert c2.matched_channels == ["dense", "bm25"]
    # c1 was first seen in dense, which carried fields.
    c1 = next(c for c in out if c.chunk_id == "c1")
    assert c1.fields == {"text": "a"}
    assert out[0].chunk_id == "c2"


def test_rrf_respects_k():
    out = rrf_fuse([_run("dense", [("c1", 1.0, None)])], rrf_k=1)
    assert out[0].fusion_score == 0.5


def test_weighted_fuse_minmax_normalizes():
    out = weighted_fuse(
        [_run("dense", [("c1", 10.0, None), ("c2", 0.0, None)])],
        weights={"dense": 1.0, "bm25": 0.0},
    )
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c1"] == 1.0
    assert scores["c2"] == 0.0
    assert out[0].chunk_id == "c1"


def test_weighted_fuse_constant_scores_become_half():
    out = weighted_fuse(
        [_run("bm25", [("c1", 3.0, None), ("c2", 3.0, None)])],
        weights={"dense": 0.0, "bm25": 1.0},
    )
    scores = {c.chunk_id: round(c.fusion_score, 6) for c in out}
    assert scores == {"c1": 0.5, "c2": 0.5}


def test_weighted_fuse_blends_channels():
    out = weighted_fuse([
        _run("dense", [("c1", 10.0, None), ("c2", 0.0, None)]),
        _run("bm25", [("c1", 0.0, None), ("c2", 10.0, None)]),
    ], weights={"dense": 0.5, "bm25": 0.5})
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c1"] == scores["c2"] == 0.5
