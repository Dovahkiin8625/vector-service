"""Small-to-large context expansion.

Recall returns leaf chunks. After the final top-k truncation,
:func:`expand_to_level` replaces each leaf with its ancestor at the
requested context level:

- ``"chunk"`` — no expansion;
- ``"section"`` — the containing section parent;
- ``"document"`` — the document root.

Sibling leaves whose chain reaches the same ancestor collapse into
one output (adjacent-overlap dedup), in first-hit order, so two
neighbouring leaves expanded to their section return the section
once. Ancestor content is read from the corpus
(:meth:`CorpusRepository.get_chunk_ancestor_rows`); a leaf the
corpus cannot resolve passes through unchanged.
"""
from __future__ import annotations

from typing import Any

from vector_service.retrieval.base import RetrievedChunk

#: Supported expansion targets, smallest to largest.
CONTEXT_LEVELS = ("chunk", "section", "document")


def expand_to_level(
    chunks: list[RetrievedChunk],
    *,
    repo: Any,
    level: str,
) -> list[RetrievedChunk]:
    """Fetch ancestors at ``level`` and expand.

    Runs the (blocking) repository lookup inline — callers run this
    inside a worker thread / executor.
    """
    if level == "chunk" or not chunks:
        return chunks
    if level not in CONTEXT_LEVELS:
        raise ValueError(f"unknown context level {level!r}")
    rows = repo.get_chunk_ancestor_rows(
        [c.chunk_id for c in chunks], level
    )
    return expand_chunks(chunks, rows)


def expand_chunks(
    chunks: list[RetrievedChunk],
    ancestor_rows: dict[str, dict[str, Any]],
) -> list[RetrievedChunk]:
    """Replace leaves with their ancestor rows, collapsing siblings.

    ``ancestor_rows`` maps each leaf chunk id to
    ``{"chunk_id": ancestor_id, "text": ..., "level": ..., ...}``.
    Every distinct ancestor appears once, at the position of its
    first contributing leaf; leaves missing from the map pass
    through unchanged.
    """
    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        row = ancestor_rows.get(chunk.chunk_id)
        if row is None:
            expanded.append(chunk)
            continue
        ancestor_id = str(row["chunk_id"])
        if ancestor_id in seen:
            continue
        seen.add(ancestor_id)
        fields = dict(chunk.fields)
        fields.update({k: v for k, v in row.items() if k != "chunk_id"})
        expanded.append(
            RetrievedChunk(
                chunk_id=ancestor_id,
                fields=fields,
                fusion_score=chunk.fusion_score,
                matched_channels=chunk.matched_channels,
                rerank_score=chunk.rerank_score,
            )
        )
    return expanded
