"""Corpus layer: SQLite system of record for documents and chunks.

This package is the authoritative store for *content and structure*:
parsed documents, their chunks, the chunk hierarchy, and the registry of
derived indexes (dense / sparse / summary vectors) built on top of them.

The vector store (Milvus) is a **derived index**: its rows can be rebuilt
at any time from the records here. Online retrieval hits the vector index
first, then hydrates content via :class:`~vector_service.corpus.repository.CorpusRepository`.
"""

from vector_service.corpus.models import (
    JOB_CANCELLED,
    JOB_CHUNKING,
    JOB_DONE,
    JOB_EMBEDDING,
    JOB_EVALUATING,
    JOB_EXTRACTING,
    JOB_FAILED,
    JOB_GRAPHING,
    JOB_PARSING,
    JOB_QUEUED,
    JOB_STAGES,
    JOB_TERMINAL,
    JOB_UPSERTING,
    ChunkRecord,
    DocumentRecord,
    IndexEntry,
)
from vector_service.corpus.repository import CorpusRepository

__all__ = [
    "JOB_CANCELLED",
    "JOB_CHUNKING",
    "JOB_DONE",
    "JOB_EMBEDDING",
    "JOB_EVALUATING",
    "JOB_EXTRACTING",
    "JOB_FAILED",
    "JOB_GRAPHING",
    "JOB_PARSING",
    "JOB_QUEUED",
    "JOB_STAGES",
    "JOB_TERMINAL",
    "JOB_UPSERTING",
    "ChunkRecord",
    "CorpusRepository",
    "DocumentRecord",
    "IndexEntry",
]
