"""MMR selection over fused candidates."""
from __future__ import annotations

from vector_service.retrieval.diversity import mmr


def _chunks(n):
    from vector_service.retrieval.base import RetrievedChunk

    return [
        RetrievedChunk(chunk_id=f"c{i}", fields={}, fusion_score=1.0 - 0.1 * i,
                       matched_channels=["dense"])
        for i in range(n)
    ]


def test_lambda_one_picks_query_similarity_order():
    # query ~ x-axis; docs from most-aligned to least, last doc on y-axis
    docs = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=1.0)
    assert [c.chunk_id for c in out] == ["c0", "c1", "c2"]


def test_lambda_zero_second_pick_is_most_dissimilar():
    docs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=0.0, top_k=2)
    assert [c.chunk_id for c in out] == ["c0", "c2"]


def test_top_k_truncates():
    docs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=1.0, top_k=2)
    assert len(out) == 2


def test_first_pick_is_highest_query_similarity():
    docs = [[0.1, 0.9], [0.9, 0.1]]
    out = mmr(_chunks(2), docs, [1.0, 0.0], lambda_mult=0.7, top_k=1)
    assert out[0].chunk_id == "c1"
