"""Dataclasses exchanged between the corpus layer and its callers."""

from __future__ import annotations

from dataclasses import dataclass

# ---- ingest job lifecycle ----

JOB_QUEUED = "queued"
JOB_PARSING = "parsing"
JOB_CHUNKING = "chunking"
JOB_EMBEDDING = "embedding"
JOB_UPSERTING = "upserting"
# Graph builds reuse the jobs table: LLM extraction over every leaf, then
# community detection / summary / vector indexing.
JOB_EXTRACTING = "extracting"
JOB_GRAPHING = "graphing"
# Eval runs reuse the jobs table: one RetrievalPipeline run per question.
JOB_EVALUATING = "evaluating"
JOB_DONE = "done"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

#: Active stage statuses — a job sitting in one of these is in flight.
JOB_STAGES = (
    JOB_PARSING,
    JOB_CHUNKING,
    JOB_EXTRACTING,
    JOB_GRAPHING,
    JOB_EVALUATING,
    JOB_EMBEDDING,
    JOB_UPSERTING,
)

#: Terminal statuses. From any of these the job never moves again.
JOB_TERMINAL = (JOB_DONE, JOB_FAILED, JOB_CANCELLED)

#: Every status an ingest job may move through, in stage order.
JOB_STATUSES = (
    JOB_QUEUED,
    *JOB_STAGES,
    *JOB_TERMINAL,
)


@dataclass
class DocumentRecord:
    """One source document registered in the corpus.

    ``database`` / ``collection`` name the vector index this document's
    chunks are derived into. ``content_hash`` is the SHA256 of the raw
    uploaded bytes (future dedupe / change-detection key).
    """

    doc_id: str
    database: str
    collection: str
    filename: str | None
    mime: str | None
    content_hash: str
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    status: str = "active"
    created_ts: float = 0.0


@dataclass
class ChunkRecord:
    """One chunk: content plus the structure fields the hierarchy will use.

    ``parent_id`` / ``level`` / ``char_start`` / ``char_end`` are
    scaffolding for hierarchical (small-to-large) retrieval; they are
    ``None`` / the default while chunks stay single-level.

    ``context`` is the LLM-generated contextual prefix used at embed
    time. It is persisted so an index can be rebuilt later, but is never
    returned as part of chunk content.
    """

    chunk_id: str
    doc_id: str
    database: str
    collection: str
    chunk_index: int
    text: str
    section_header: str = ""
    page_number: int | None = None
    token_count: int = 0
    context: str | None = None
    summary: str | None = None
    parent_id: str | None = None
    level: str = "chunk"
    char_start: int | None = None
    char_end: int | None = None
    created_ts: float = 0.0


@dataclass
class IndexEntry:
    """Registry row recording that one chunk carries one derived index.

    ``index_kind`` is ``dense`` / ``sparse`` / ``summary`` (extensible).
    ``model`` is the exact model id that produced the index, so switching
    embedding models is a registry change plus a rebuild, never a guess.
    ``index_ref`` identifies the concrete index (``"<collection>:<field>"``).
    """

    chunk_id: str
    doc_id: str
    database: str
    collection: str
    index_kind: str
    model: str
    index_ref: str
    created_ts: float = 0.0
