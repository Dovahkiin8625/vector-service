"""Reranker registry: name → implementation class.

Mirrors ``embeddings/registry.py``. Concrete classes are registered
here to avoid forcing every backend's import at package import time.
"""
from __future__ import annotations

from vector_service.rerankers.base import Reranker

RERANKER_REGISTRY: dict[str, type[Reranker]] = {}


def get_reranker_class(name: str) -> type[Reranker]:
    """Return the registered reranker class for ``name``.

    Raises ``KeyError`` if no reranker is registered under that name.
    """
    try:
        return RERANKER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown reranker backend: {name!r}; "
            f"registered: {sorted(RERANKER_REGISTRY)}"
        ) from exc


def list_reranker_names() -> list[str]:
    """Return all registered reranker backend names (sorted by insertion)."""
    return list(RERANKER_REGISTRY)


from . import cross_encoder  # noqa: F401  # side-effect: registers CrossEncoderReranker