"""Rank fusion: Reciprocal Rank Fusion and per-channel weighted fusion.

Both functions are pure — they take ranked channel runs and return
fused chunks, which makes the fusion contract trivial to unit test.
"""
from __future__ import annotations

from typing import Any

from vector_service.retrieval.base import (
    ChannelRun,
    RetrievedChunk,
)


def _collect(runs: list[ChannelRun]) -> tuple[
    dict[str, float], dict[str, dict], dict[str, list[str]]
]:
    """Initialise per-id score/fields/channels by scanning runs in order."""
    scores: dict[str, float] = {}
    fields: dict[str, dict] = {}
    channels: dict[str, list[str]] = {}
    for run in runs:
        for hit in run.hits:
            cid = hit.chunk_id
            if cid not in scores:
                scores[cid] = 0.0
                fields[cid] = dict(hit.fields)
                channels[cid] = []
            if run.channel not in channels[cid]:
                channels[cid].append(run.channel)
    return scores, fields, channels


def _finalize(
    scores: dict[str, float],
    fields: dict[str, dict],
    channels: dict[str, list[str]],
) -> list[RetrievedChunk]:
    out = [
        RetrievedChunk(
            chunk_id=cid,
            fields=fields[cid],
            fusion_score=scores[cid],
            matched_channels=channels[cid],
        )
        for cid in scores
    ]
    out.sort(key=lambda c: c.fusion_score, reverse=True)
    return out


def rrf_fuse(runs: list[ChannelRun], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: ``score(d) = Σ 1 / (rrf_k + rank_i(d))``.

    Ranks start at 1; a chunk missing from a leg contributes nothing
    for that leg. Robust to wildly different per-channel score scales.
    """
    scores, fields, channels = _collect(runs)
    for run in runs:
        for hit in run.hits:
            scores[hit.chunk_id] += 1.0 / (rrf_k + hit.rank)
    return _finalize(scores, fields, channels)


def _normalize_run(run: ChannelRun) -> dict[str, float]:
    """Min-max normalize one leg's raw scores to [0, 1].

    A leg whose scores are all equal contributes 0.5 to every hit —
    picking 0 (or 1) would silently silence (or dominate) that leg.
    """
    vals: list[tuple[str, float]] = [(h.chunk_id, h.score) for h in run.hits]
    if not vals:
        return {}
    lo = min(s for _, s in vals)
    hi = max(s for _, s in vals)
    if hi == lo:
        return {cid: 0.5 for cid, _ in vals}
    span = hi - lo
    return {cid: (s - lo) / span for cid, s in vals}


def weighted_fuse(
    runs: list[ChannelRun], weights: dict[str, Any]
) -> list[RetrievedChunk]:
    """Weighted sum of per-leg min-max-normalized scores.

    ``weights`` maps channel name ("dense" / "bm25") to a non-negative
    weight; legs whose channel has weight 0 still appear in the trace
    but contribute nothing to the fused score.
    """
    scores, fields, channels = _collect(runs)
    for run in runs:
        weight = float(weights.get(run.channel, 0.0) or 0.0)
        if weight == 0.0:
            continue
        for cid, norm in _normalize_run(run).items():
            scores[cid] += weight * norm
    return _finalize(scores, fields, channels)
