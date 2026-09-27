"""Pluggable text-chunking subsystem.

Importing the package registers every built-in strategy. Build a
chunker by name with :func:`build_chunker`; registered names are
available via :func:`list_strategies`.
"""
from __future__ import annotations

from vector_service.chunking.base import (
    Chunk,
    Chunker,
    get_strategy_class,
    list_strategies,
    register_strategy,
)
from vector_service.chunking.factory import build_chunker

# Import concrete chunkers for their registration side effects.
from vector_service.chunking import (
    fixed_chunker,
    paragraph_chunker,
    recursive_chunker,
    semantic_chunker,
    llm_chunker,
)

__all__ = [
    "Chunk",
    "Chunker",
    "build_chunker",
    "get_strategy_class",
    "list_strategies",
    "register_strategy",
]
