"""Pure-Python similarity primitives.

Lives next to the embedder base classes (``embeddings/base.py`` etc.) so
``api/similarity.py`` can call ``pairwise_scores`` after embedding.

Implemented in pure Python (``math`` / builtins) on purpose — the codebase
declares ``torch`` as an *optional* dep and never imports ``numpy``, so
adding a hard dependency here would surprise downstream installers. The
hot paths only touch short dense float lists (dim <= 1024 in the current
model set), so a tight Python loop is comfortably under a millisecond per
thousand pairs.

Supported metrics mirror the literals already used by the vector store
schemas (``Literal["cosine", "ip", "l2"]`` in ``schemas/management.py``):

* ``cosine`` — cosine similarity, range ``[-1, 1]``; higher is better.
* ``ip``     — inner / dot product; higher is better for normalised vectors
  (otherwise unbounded). Reuses the Milvus label so the similarity score
  the user sees here matches what the store returns for the same vectors.
* ``l2``     — Euclidean distance; **lower is better**.

Sort order convention:

* For ``cosine`` and ``ip`` the result list is sorted descending by score.
* For ``l2`` the result list is sorted ascending by score.

This matches the contract documented on ``HitResponse.score`` in
``schemas/management.py:539-552``.
"""
from __future__ import annotations

import math
from typing import Iterable, Literal

Metric = Literal["cosine", "ip", "l2"]
"""Supported similarity metrics (mirrors ``schemas/management.py`` literals)."""

_VALID_METRICS: frozenset[str] = frozenset({"cosine", "ip", "l2"})


def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _norm(a: list[float]) -> float:
    s = 0.0
    for x in a:
        s += x * x
    return math.sqrt(s)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Returns ``0.0`` on zero-norm input."""
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def ip(a: list[float], b: list[float]) -> float:
    """Inner / dot product. Higher = more similar (for normalised vectors)."""
    return _dot(a, b)


def l2(a: list[float], b: list[float]) -> float:
    """Euclidean (L2) distance. Lower = more similar."""
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


_SCORE_FNS = {"cosine": cosine, "ip": ip, "l2": l2}


def validate_metric(metric: str) -> Metric:
    """Coerce / validate a metric string; raise ``ValueError`` on unknown."""
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"unsupported metric {metric!r}; "
            f"expected one of {sorted(_VALID_METRICS)}"
        )
    return metric  # type: ignore[return-value]


def pairwise_scores(
    query_vec: list[float],
    vectors: Iterable[list[float]],
    metric: str,
) -> list[float]:
    """Compute similarity between ``query_vec`` and each entry in ``vectors``.

    Args:
        query_vec: Reference vector. Length must equal every entry in
            ``vectors``; mismatched lengths fall through to ``zip`` and
            silently truncate — callers are expected to have validated
            ``dim`` upstream (the embedder API surfaces a 422
            ``dimension_mismatch`` for cross-family comparisons).
        vectors: Iterable of candidate vectors in original input order.
        metric: One of ``cosine``, ``ip``, ``l2``.

    Returns:
        List of float scores, one per input vector, in input order.
    """
    fn = _SCORE_FNS[validate_metric(metric)]
    return [fn(query_vec, v) for v in vectors]


def arg_sort(scores: list[float], metric: str) -> list[int]:
    """Return indices that sort ``scores`` from most-similar to least-similar.

    For ``cosine`` and ``ip`` (higher is better) sorts descending; for
    ``l2`` (lower is better) sorts ascending. Stable on ties (Python's
    ``sorted`` is stable).
    """
    reverse = validate_metric(metric) != "l2"
    return [i for i, _ in sorted(enumerate(scores), key=lambda kv: kv[1], reverse=reverse)]