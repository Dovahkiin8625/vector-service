"""SQLite connection management and schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with the corpus PRAGMAs.

    - ``journal_mode=WAL`` + ``synchronous=NORMAL``: readers do not block
      on a concurrent writer; safe for a single host.
    - ``busy_timeout=5000``: lock contention waits instead of raising
      immediately.
    - ``foreign_keys=ON``: cascade deletes rely on this (default is off).
    """
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Destructive changes keyed by the schema generation that introduced
# them. When an existing database predates ``version`` the statement
# runs BEFORE the idempotent DDL recreates what was dropped. No
# compatibility branches: old data is discarded by design.
_BREAKING_CHANGES: tuple[tuple[int, str], ...] = (
    # v2 -> v3: ingest_jobs gained doc_id/params/spool/attempts/cancel/
    # progress columns; v3 -> v4: not_before_ts joined in. Job history
    # is disposable, rebuild the table at each generation.
    (3, "DROP TABLE IF EXISTS ingest_jobs"),
    (4, "DROP TABLE IF EXISTS ingest_jobs"),
    # v5: job_type joined the row; worker dispatch reads it.
    (5, "DROP TABLE IF EXISTS ingest_jobs"),
)


def initialize(path: str | Path) -> None:
    """Ensure the parent directory and every corpus table exist.

    Reads ``PRAGMA user_version`` first so an older database gets the
    pending destructive changes before DDL runs. A fresh database
    (version 0) skips straight to the current DDL.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    from vector_service.corpus.schema import DDL, SCHEMA_VERSION

    conn = connect(db_path)
    try:
        existing = conn.execute("PRAGMA user_version").fetchone()[0]
        for version, stmt in _BREAKING_CHANGES:
            if existing < version:
                conn.execute(stmt)
        for stmt in DDL:
            conn.execute(stmt)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()
