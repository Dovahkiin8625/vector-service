"""Embedder abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base for embedding models.

    Concrete subclasses must:
    - Set `dim` (int) and `model_name` (str)
    - Implement `embed_documents` and `embed_query`

    Both methods are synchronous. Async dispatch is the caller's job
    (use `loop.run_in_executor` from FastAPI routes).
    """

    dim: int
    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Eagerly load the model into memory and run any one-time warmup.

        Called once during application startup (see `core.lifespan`).
        Subclasses that lazily defer heavy work to first use must
        implement this so the load happens up front and readiness
        probes can observe it. May raise `ModelNotLoaded`.
        """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Return one vector per input."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. May differ from embed_documents (e.g. BGE prefix)."""
