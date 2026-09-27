"""Maximal Marginal Relevance.

MMR trades relevance against redundancy: after taking the top-relevant
document, each next pick maximizes

    λ · sim(d, q) − (1 − λ) · max sim(d, selected)

so near-duplicate fused candidates are pushed apart. Similarities are
cosine — every vector is L2-normalized before scoring.
"""
from __future__ import annotations

import math

from vector_service.retrieval.base import RetrievedChunk


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr(
    chunks: list[RetrievedChunk],
    doc_vectors: list[list[float]],
    query_vector: list[float],
    *,
    lambda_mult: float = 0.7,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``top_k`` MMR-selected chunks in selection order."""
    if not chunks:
        return []
    if top_k is None:
        top_k = len(chunks)
    if top_k <= 0:
        return []
    docs = [_l2_normalize(v) for v in doc_vectors]
    query = _l2_normalize(query_vector)
    query_sim = [_cosine(d, query) for d in docs]

    # First pick is always the single most query-relevant candidate. This
    # must be an explicit argmax: at lambda_mult == 0 every first-round
    # MMR score is identically zero, so the generic loop would leave the
    # pick to set iteration order rather than query similarity.
    selected_idx: list[int] = [
        max(range(len(chunks)), key=query_sim.__getitem__)
    ]
    remaining = set(range(len(chunks)))
    remaining.discard(selected_idx[0])
    while remaining and len(selected_idx) < top_k:
        best_idx, best_score = None, None
        for i in remaining:
            redundancy = max(_cosine(docs[i], docs[j]) for j in selected_idx)
            score = lambda_mult * query_sim[i] - (1.0 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best_idx, best_score = i, score
        selected_idx.append(best_idx)
        remaining.discard(best_idx)
    return [chunks[i] for i in selected_idx]
