"""Reranker abstract base class and result dataclass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredHit:
    """A single document with a relevance score from the reranker.

    ``index`` is the position of the document in the input ``documents``
    list — never a database primary key. ``score`` is whatever the
    underlying reranker produces (typically a sigmoid-normalised logit
    in [0, 1] for sentence-transformers CrossEncoder, but the value
    is NOT guaranteed normalised; clients must not assume a range).
    """

    index: int
    score: float


class Reranker(ABC):
    """Abstract base for reranker backends.

    The contract mirrors ``Embedder``: concrete subclasses expose a
    ``model_name`` class attribute and implement ``load()`` (eager,
    idempotent, called from ``lifespan``) plus ``rerank()``.

    Implementations are CPU/GPU-bound and are expected to be called from
    a thread executor by the route layer — they themselves do NOT need
    to be async.
    """

    model_name: str  # class attribute; e.g. "bge-reranker-v2-m3"

    @abstractmethod
    def load(self) -> None:
        """Load weights / connect to backend. Idempotent."""

    def unload(self) -> None:
        """Release weights and any associated resources.

        Mirrors ``Embedder.unload``: idempotent, default no-op.
        Subclasses that load weights onto GPU should override to
        drop references and call ``torch.cuda.empty_cache()``.
        """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[ScoredHit]:
        """Return ``documents`` reordered by relevance.

        The returned list MUST be sorted by ``score`` descending.
        If ``top_n`` is given, the list is truncated to at most
        ``top_n`` entries; ``None`` means return everything.
        """
