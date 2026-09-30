"""Chunker abstractions and the strategy registry.

Every splitting strategy implements the :class:`Chunker` interface
and registers itself under a stable name with
:func:`register_strategy`. The API layer never imports a concrete
chunker class — it asks :func:`vector_service.chunking.factory.build_chunker`
for the named strategy, so adding a new strategy is a local change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

from vector_service.chunking.tokens import count_tokens

# Hierarchy level tags — the values are persisted in ``chunks.level``
# and carried on the wire, so treat renames like an API version.
LEVEL_CHUNK = "chunk"
LEVEL_SECTION = "section"
LEVEL_DOCUMENT = "document"


@dataclass
class Chunk:
    """A single emitted chunk.

    ``section_header`` is the breadcrumb header path (e.g. ``"1.
    Introduction > 1.1 Background"``) at the time the chunk was
    finalised; ``""`` for header-less input. ``page_number`` is
    copied from section metadata when present; ``None`` for formats
    that don't have pages. ``context`` holds the optional
    LLM-generated situating prefix (Anthropic contextual retrieval);
    it is prepended at embed time but is NOT part of the stored
    chunk text.

    Hierarchy fields: ``level`` is this row's level; ``key`` /
    ``parent_key`` are document-local link keys (``"document"`` /
    ``"section:{ord}"`` / ``"chunk:{i}"``) resolved to real
    ``chunk_id`` values at persist time. ``section_ord`` tags a leaf
    with its source section; ``char_start`` / ``char_end`` are the
    source offsets of the row's *own* (non-reconstructed) text.
    ``summary`` holds the optional LLM chunk summary, embedded as
    ``summary_vector``.
    """

    text: str
    chunk_index: int
    token_count: int
    section_header: str
    page_number: int | None = None
    context: str | None = None
    summary: str | None = None
    key: str | None = None
    parent_key: str | None = None
    level: str = LEVEL_CHUNK
    section_ord: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class Chunker(ABC):
    """Common chunker contract.

    Concrete constructors accept (at minimum) ``chunk_size``,
    ``chunk_overlap`` and an optional ``token_counter`` override.
    """

    chunk_size: int
    chunk_overlap: int
    token_counter: Callable[[str], int]

    @abstractmethod
    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Split ``markdown`` into chunks in source order."""


def validate_size_overlap(chunk_size: int, chunk_overlap: int) -> None:
    """Validate the shared chunk-size / overlap invariants.

    ``chunk_size`` must be positive; ``chunk_overlap`` must be
    non-negative and strictly smaller than ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
            f"({chunk_size})"
        )


class BaseChunker(Chunker):
    """Base class storing the shared chunker configuration.

    Also provides the ``token_count_for`` helper so subclasses don't
    have to reach around the injected counter.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        token_counter: Callable[[str], int] | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        validate_size_overlap(chunk_size, chunk_overlap)
        if min_chunk_size is not None:
            if min_chunk_size < 0:
                raise ValueError(
                    f"min_chunk_size must be >= 0, got {min_chunk_size}"
                )
            if min_chunk_size >= chunk_size:
                raise ValueError(
                    f"min_chunk_size ({min_chunk_size}) must be < "
                    f"chunk_size ({chunk_size})"
                )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = token_counter if token_counter is not None else count_tokens
        self._configured_min_chunk_size = min_chunk_size

    @property
    def min_chunk_size(self) -> int:
        """Chunks below this many tokens get merged into neighbours.

        Defaults to 10% of ``chunk_size`` when not explicitly set.
        """
        if self._configured_min_chunk_size:
            return self._configured_min_chunk_size
        return max(1, self.chunk_size // 10)

    @property
    def _chunk_size(self) -> int:
        # Compatibility alias for older code/tests.
        return self.chunk_size

    @property
    def _chunk_overlap(self) -> int:
        return self.chunk_overlap

    def _count(self, text: str) -> int:
        return self.token_counter(text)


# ---- strategy registry -------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type[Chunker]] = {}


def register_strategy(name: str) -> Callable[[type[Chunker]], type[Chunker]]:
    """Class decorator registering a chunker under ``name``.

    Strategy names are the public API contract of ``POST /v1/chunk``
    and ``POST /v1/ingest`` — treat renames like an API version.
    """

    def _decorator(cls: type[Chunker]) -> type[Chunker]:
        _STRATEGY_REGISTRY[name] = cls
        return cls

    return _decorator


def list_strategies() -> list[str]:
    """Return the registered strategy names in registration order."""
    return list(_STRATEGY_REGISTRY)


def get_strategy_class(name: str) -> type[Chunker]:
    """Look up the chunker class registered as ``name``.

    Raises ``KeyError`` for unknown strategies.
    """
    return _STRATEGY_REGISTRY[name]
