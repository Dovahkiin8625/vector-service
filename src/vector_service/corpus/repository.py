"""Repository over the corpus SQLite database.

Every method opens its own short-lived connection (SQLite connections are
cheap and are not shared across the worker threads that call this layer).
Multi-row writes that must be atomic take an explicit ``conn`` so they can
share one transaction.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vector_service.corpus.db import connect, initialize
from vector_service.corpus.models import (
    JOB_CANCELLED,
    JOB_DONE,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_STAGES,
    JOB_TERMINAL,
    ChunkRecord,
    DocumentRecord,
    IndexEntry,
)

# Client-facing chunk fields, assembled from chunks + documents. Kept here
# so retrieval hydration and content browsing return one vocabulary.
CONTENT_FIELDS = (
    "doc_id",
    "chunk_index",
    "text",
    "section_header",
    "page_number",
    "char_start",
    "char_end",
    "token_count",
    "summary",
    "title",
    "author",
    "page_count",
    "filename",
)


class CorpusRepository:
    """System-of-record gateway for documents, chunks, jobs, index registry."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        initialize(self.path)

    @contextmanager
    def _txn(self) -> Iterator[Any]:
        conn = connect(self.path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- ingest jobs ----------------------------------------------------

    def create_job(
        self,
        job_id: str,
        *,
        database: str,
        collection: str,
        filename: str | None,
        mime: str | None,
        doc_id: str | None = None,
        job_type: str = "ingest",
        params_json: str = "{}",
        spool_path: str | None = None,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO ingest_jobs
                    (job_id, job_type, doc_id, database, collection, filename,
                     mime, status, params_json, spool_path, max_attempts,
                     created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    doc_id,
                    database,
                    collection,
                    filename,
                    mime,
                    JOB_QUEUED,
                    params_json,
                    spool_path,
                    max_attempts,
                    now,
                    now,
                ),
            )

    def mark_job(self, job_id: str, status: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                "UPDATE ingest_jobs SET status = ?, updated_ts = ? WHERE job_id = ?",
                (status, now, job_id),
            )

    def set_job_doc_id(self, job_id: str, doc_id: str) -> None:
        """Persist the doc id minted for a job (needed for recovery)."""
        with self._txn() as conn:
            conn.execute(
                "UPDATE ingest_jobs SET doc_id = ? WHERE job_id = ?",
                (doc_id, job_id),
            )

    def set_job_progress(self, job_id: str, current: int, total: int) -> None:
        """Update the progress counters shown while a stage runs."""
        now = time.time()
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET progress_current = ?, progress_total = ?, updated_ts = ?
                WHERE job_id = ?
                """,
                (current, total, now, job_id),
            )

    def request_cancel(self, job_id: str) -> bool:
        """Ask the worker to cancel a job at the next stage boundary.

        Returns ``True`` when the flag was set, ``False`` when the job
        is already terminal (nothing to cancel).
        """
        placeholders = ",".join("?" for _ in JOB_TERMINAL)
        with self._txn() as conn:
            cursor = conn.execute(
                f"""
                UPDATE ingest_jobs
                SET cancel_requested = 1, updated_ts = ?
                WHERE job_id = ? AND status NOT IN ({placeholders})
                """,
                (time.time(), job_id, *JOB_TERMINAL),
            )
        return cursor.rowcount > 0

    def claim_next_queued(self, *, now: float | None = None) -> dict[str, Any] | None:
        """Return the oldest claimable queued job of any worker type.

        The worker dispatches on the row's ``job_type``: ``ingest``
        rows carry an upload spool, ``rebuild`` rows derive everything
        from the existing corpus. A job parked for retry backoff
        (``not_before_ts`` in the future) is skipped. The caller marks
        the job into its first stage itself — single-process
        ``workers=1`` makes a separate claim state unnecessary.
        """
        now = time.time() if now is None else now
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT * FROM ingest_jobs
                WHERE status = ?
                      AND (not_before_ts IS NULL OR not_before_ts <= ?)
                ORDER BY created_ts, job_id
                LIMIT 1
                """,
                (JOB_QUEUED, now),
            ).fetchone()
        return dict(row) if row is not None else None

    def requeue_job(
        self,
        job_id: str,
        *,
        attempts: int,
        not_before: float | None = None,
        now: float | None = None,
    ) -> None:
        """Put a failed attempt back on the queue with an attempt count.

        Clears transient execution state (progress, error, cancel flag)
        while keeping the original row — and its ``created_ts`` — so
        ordering still reflects submit time. ``not_before`` parks the
        job for retry backoff; ``claim_next_queued`` keeps skipping it
        until then.
        """
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, attempts = ?, cancel_requested = 0,
                    progress_current = NULL, progress_total = NULL,
                    error_code = NULL, error_message = NULL,
                    not_before_ts = ?, updated_ts = ?
                WHERE job_id = ?
                """,
                (JOB_QUEUED, attempts, not_before, now, job_id),
            )

    def mark_cancelled(self, job_id: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, updated_ts = ?, finished_ts = ?
                WHERE job_id = ?
                """,
                (JOB_CANCELLED, now, now, job_id),
            )

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs newest-first, optionally filtered by status."""
        with self._txn() as conn:
            if status is None:
                total = conn.execute("SELECT COUNT(*) FROM ingest_jobs").fetchone()[0]
                rows = conn.execute(
                    """
                    SELECT * FROM ingest_jobs
                    ORDER BY created_ts DESC, job_id
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM ingest_jobs WHERE status = ?",
                    (status,),
                ).fetchone()[0]
                rows = conn.execute(
                    """
                    SELECT * FROM ingest_jobs
                    WHERE status = ?
                    ORDER BY created_ts DESC, job_id
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
        return [dict(row) for row in rows], int(total)

    def finish_job(
        self,
        job_id: str,
        *,
        chunk_count: int,
        page_count: int | None,
        tokens_used: int,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, chunk_count = ?, page_count = ?,
                    tokens_used = ?, updated_ts = ?, finished_ts = ?
                WHERE job_id = ?
                """,
                (JOB_DONE, chunk_count, page_count, tokens_used, now, now, job_id),
            )

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        attempts: int | None = None,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            if attempts is None:
                conn.execute(
                    """
                    UPDATE ingest_jobs
                    SET status = ?, error_code = ?, error_message = ?,
                        updated_ts = ?, finished_ts = ?
                    WHERE job_id = ?
                    """,
                    (JOB_FAILED, error_code, error_message, now, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE ingest_jobs
                    SET status = ?, error_code = ?, error_message = ?,
                        attempts = ?, updated_ts = ?, finished_ts = ?
                    WHERE job_id = ?
                    """,
                    (
                        JOB_FAILED,
                        error_code,
                        error_message,
                        attempts,
                        now,
                        now,
                        job_id,
                    ),
                )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def find_interrupted(self) -> list[dict[str, Any]]:
        """Return jobs left in a running stage by a dead process.

        Only an ungraceful exit leaves rows with a stage status — the
        worker marks terminal states itself. Startup recovery decides
        per row whether it can be requeued (spool present) or must be
        failed (upload bytes gone).
        """
        placeholders = ",".join("?" for _ in JOB_STAGES)
        with self._txn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM ingest_jobs
                WHERE status IN ({placeholders})
                ORDER BY created_ts, job_id
                """,
                JOB_STAGES,
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- document + chunks ---------------------------------------------

    def store_document(
        self, document: DocumentRecord, chunks: list[ChunkRecord]
    ) -> None:
        """Insert one document and all of its chunks in a single transaction.

        Called by the ingest pipeline *before* any vectors are written:
        content lands first, so the vector rows are always a derivation of
        something the corpus already holds.
        """
        with self._txn() as conn:
            self._put_document(conn, document)
            self._put_chunks(conn, chunks)

    @staticmethod
    def _put_document(conn: Any, document: DocumentRecord) -> None:
        conn.execute(
            """
            INSERT INTO documents
                (doc_id, database, collection, filename, mime, content_hash,
                 title, author, page_count, status, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.database,
                document.collection,
                document.filename,
                document.mime,
                document.content_hash,
                document.title,
                document.author,
                document.page_count,
                document.status,
                document.created_ts,
            ),
        )

    @staticmethod
    def _put_chunks(conn: Any, chunks: list[ChunkRecord]) -> None:
        conn.executemany(
            """
            INSERT INTO chunks
                (chunk_id, doc_id, database, collection, chunk_index, text,
                 section_header, page_number, token_count, context, summary,
                 parent_id, level, char_start, char_end, created_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.database,
                    c.collection,
                    c.chunk_index,
                    c.text,
                    c.section_header,
                    c.page_number,
                    c.token_count,
                    c.context,
                    c.summary,
                    c.parent_id,
                    c.level,
                    c.char_start,
                    c.char_end,
                    c.created_ts,
                )
                for c in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> list[str]:
        """Remove every corpus row belonging to a document.

        Covers the chunk index registry, chunks (FK cascade would also do
        it), and the document itself. Job history is intentionally kept so
        failed ingests stay queryable. Used by ingest rollback and the
        document-delete flows.

        Returns the content hashes that lost their *last* referencing
        document — callers drop the matching content-addressed blobs.
        """
        with self._txn() as conn:
            digest = conn.execute(
                "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            chunk_rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            conn.execute("DELETE FROM chunk_indexes WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._prune_graph_for_chunks(
                conn, [row["chunk_id"] for row in chunk_rows]
            )
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return self._orphan_hashes(
                conn, [digest[0]] if digest is not None else []
            )

    @staticmethod
    def _orphan_hashes(conn: Any, digests: list[str]) -> list[str]:
        """Filter to hashes no surviving document references."""
        orphaned: list[str] = []
        for digest in digests:
            count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE content_hash = ?",
                (digest,),
            ).fetchone()[0]
            if count == 0:
                orphaned.append(digest)
        return orphaned

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    # ---- deletions mirrored from the derived index ----------------------

    def delete_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Delete index rows for the given chunks from the system of record.

        Mirrors a vector-index row delete: chunk_indexes and chunks go
        first; any document left with zero chunks is deleted too. Returns
        the hashes that lost their last referencing document so callers
        drop the matching content-addressed blobs.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._txn() as conn:
            affected = conn.execute(
                f"SELECT DISTINCT doc_id FROM chunks "
                f"WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            doc_ids = [row["doc_id"] for row in affected]
            doc_ph = ",".join("?" for _ in doc_ids)
            hashes = [
                row[0]
                for row in conn.execute(
                    f"SELECT DISTINCT content_hash FROM documents "
                    f"WHERE doc_id IN ({doc_ph})",
                    doc_ids,
                ).fetchall()
            ]
            conn.execute(
                f"DELETE FROM chunk_indexes WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            conn.execute(
                f"DELETE FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            self._prune_graph_for_chunks(conn, chunk_ids)
            if doc_ids:
                conn.execute(
                    f"DELETE FROM documents WHERE doc_id IN ({doc_ph}) "
                    f"AND NOT EXISTS (SELECT 1 FROM chunks c "
                    f"WHERE c.doc_id = documents.doc_id)",
                    doc_ids,
                )
            return self._orphan_hashes(conn, hashes)

    def delete_for_collection(
        self, database: str, collection: str
    ) -> list[str]:
        """Drop every corpus row in one (database, collection).

        Returns hashes that lost their last referencing document — the
        same hash may survive via a document in another collection.
        """
        with self._txn() as conn:
            hashes = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT content_hash FROM documents "
                    "WHERE database = ? AND collection = ?",
                    (database, collection),
                ).fetchall()
            ]
            conn.execute(
                "DELETE FROM chunk_indexes WHERE database = ? AND collection = ?",
                (database, collection),
            )
            conn.execute(
                "DELETE FROM chunks WHERE database = ? AND collection = ?",
                (database, collection),
            )
            conn.execute(
                "DELETE FROM documents WHERE database = ? AND collection = ?",
                (database, collection),
            )
            conn.execute(
                "DELETE FROM index_bindings WHERE database = ? AND collection = ?",
                (database, collection),
            )
            self._delete_graph_scope(conn, database=database, collection=collection)
            return self._orphan_hashes(conn, hashes)

    def delete_for_database(self, database: str) -> list[str]:
        """Drop every corpus row belonging to one database.

        Returns hashes that lost their last referencing document.
        """
        with self._txn() as conn:
            hashes = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT content_hash FROM documents "
                    "WHERE database = ?",
                    (database,),
                ).fetchall()
            ]
            conn.execute(
                "DELETE FROM chunk_indexes WHERE database = ?",
                (database,),
            )
            conn.execute(
                "DELETE FROM chunks WHERE database = ?",
                (database,),
            )
            conn.execute(
                "DELETE FROM documents WHERE database = ?",
                (database,),
            )
            conn.execute(
                "DELETE FROM index_bindings WHERE database = ?",
                (database,),
            )
            self._delete_graph_scope(conn, database=database, collection=None)
            return self._orphan_hashes(conn, hashes)

    # ---- storage maintenance -------------------------------------------

    def fragmentation(self) -> tuple[int, int]:
        """Return ``(freelist_count, page_count)`` for the corpus DB.

        The free-page ratio decides whether the maintenance worker runs a
        VACUUM: deletes and updates otherwise leave the file holding
        unreclaimed pages forever.
        """
        with self._txn() as conn:
            free = conn.execute("PRAGMA freelist_count").fetchone()[0]
            pages = conn.execute("PRAGMA page_count").fetchone()[0]
        return int(free), int(pages)

    def vacuum(self) -> None:
        """Rebuild the database file, dropping free pages.

        VACUUM cannot run inside a transaction; ``isolation_level=None``
        on the dedicated connection makes sqlite3 pass it straight
        through. A TRUNCATE checkpoint afterwards shrinks the WAL the
        rebuild just wrote. Requires no concurrent reader in WAL mode
        beyond the usual — operation is best-effort and scheduled.
        """
        import sqlite3

        conn = sqlite3.connect(str(self.path), isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("VACUUM")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()

    def all_content_hashes(self) -> list[str]:
        """Every hash a live document references (blob-sweep input)."""
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT content_hash FROM documents"
            ).fetchall()
        return [row[0] for row in rows]

    def backup_to(self, dest: str | Path) -> Path:
        """Write a consistent snapshot of the live corpus to ``dest``.

        Uses the sqlite online backup API rather than copying the db /
        WAL files: it is safe while writers are active and the result
        is one complete, independently openable database file.
        """
        import sqlite3

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with connect(self.path) as source, sqlite3.connect(str(dest)) as target:
            source.backup(target)
        return dest

    def corpus_collections(self, database: str) -> list[str]:
        """Distinct corpus collection names in a database (derived-state
        cleanup needs them when the database is dropped)."""
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT collection FROM documents WHERE database = ?",
                (database,),
            ).fetchall()
        return [row["collection"] for row in rows]

    # ---- derived-index registry ----------------------------------------

    def record_indexes(self, entries: list[IndexEntry]) -> None:
        if not entries:
            return
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_indexes
                    (chunk_id, doc_id, database, collection, index_kind,
                     model, index_ref, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.chunk_id,
                        e.doc_id,
                        e.database,
                        e.collection,
                        e.index_kind,
                        e.model,
                        e.index_ref,
                        e.created_ts,
                    )
                    for e in entries
                ],
            )

    # ---- physical index bindings ----------------------------------------

    def get_binding(
        self, database: str, collection: str
    ) -> dict[str, Any] | None:
        """Return the active/canary binding for one logical collection."""
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM index_bindings WHERE database = ? AND collection = ?",
                (database, collection),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_bindings(self, database: str) -> list[dict[str, Any]]:
        """Every binding in a database (drop cleanup needs the refs)."""
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT * FROM index_bindings WHERE database = ?",
                (database,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_canary(
        self,
        database: str,
        collection: str,
        *,
        canary_ref: str,
        canary_percent: int,
        model: str,
        now: float | None = None,
    ) -> None:
        """Point the canary slot at a freshly rebuilt physical collection.

        The active slot keeps serving. A first binding takes the logical
        collection name as its active ref — the physical layout ingests
        built before bindings existed.
        """
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO index_bindings
                    (database, collection, active_ref, canary_ref,
                     canary_percent, model, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(database, collection) DO UPDATE SET
                    canary_ref = excluded.canary_ref,
                    canary_percent = excluded.canary_percent,
                    model = excluded.model,
                    updated_ts = excluded.updated_ts
                """,
                (
                    database,
                    collection,
                    collection,
                    canary_ref,
                    canary_percent,
                    model,
                    now,
                    now,
                ),
            )

    def promote_index(
        self, database: str, collection: str, *, now: float | None = None
    ) -> dict[str, Any]:
        """Promote the canary physical collection and rewrite the registry.

        One transaction: the canary becomes active (canary cleared),
        and every ``chunk_indexes`` row of this logical collection gets
        the new ``index_ref`` — the dense rows also record the new
        model. Returns ``{"old_ref", "new_ref", "model"}``. Raises
        ``LookupError`` when no canary is parked.
        """
        now = time.time() if now is None else now
        with self._txn() as conn:
            binding = conn.execute(
                "SELECT * FROM index_bindings WHERE database = ? AND collection = ?",
                (database, collection),
            ).fetchone()
            if binding is None or binding["canary_ref"] is None:
                raise LookupError(
                    f"no canary index to promote for {database!r}/{collection!r}"
                )
            old_ref = binding["active_ref"]
            new_ref = binding["canary_ref"]
            model = binding["model"]
            conn.execute(
                """
                UPDATE index_bindings
                SET active_ref = ?, canary_ref = NULL, canary_percent = 0,
                    updated_ts = ?
                WHERE database = ? AND collection = ?
                """,
                (new_ref, now, database, collection),
            )
            conn.execute(
                """
                UPDATE chunk_indexes
                SET index_ref = ?, model = ?, created_ts = ?
                WHERE database = ? AND collection = ? AND index_kind = 'dense'
                """,
                (new_ref, model, now, database, collection),
            )
            # Sparse rows keep their analyzer model tag; only the
            # physical pointer moves.
            conn.execute(
                """
                UPDATE chunk_indexes
                SET index_ref = ?, created_ts = ?
                WHERE database = ? AND collection = ? AND index_kind = 'sparse'
                """,
                (new_ref, now, database, collection),
            )
        return {"old_ref": old_ref, "new_ref": new_ref, "model": model}

    def delete_binding(self, database: str, collection: str) -> None:
        """Remove the binding row for one logical collection (drop)."""
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM index_bindings WHERE database = ? AND collection = ?",
                (database, collection),
            )

    def delete_bindings_for_database(self, database: str) -> None:
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM index_bindings WHERE database = ?",
                (database,),
            )

    # ---- reads: retrieval hydration -------------------------------------

    @staticmethod
    def _assemble(row: Any) -> dict[str, Any]:
        """Map a joined chunks+documents row onto the content-field shape."""
        return {name: row[name] for name in CONTENT_FIELDS}

    def get_chunk_ancestor_rows(
        self, chunk_ids: list[str], level: str
    ) -> dict[str, dict[str, Any]]:
        """Map each leaf chunk id to an ancestor content row at ``level``.

        Walks ``parent_id`` chains (leaf → section → document) inside
        one connection, fetching chain rows in IN-list rounds. Returns
        ``leaf_chunk_id -> {"chunk_id": ancestor_id, ...content
        fields, "level", "char_start", "char_end"}``. A leaf whose row
        already sits at ``level`` maps to itself. Missing chain links
        fall back to the deepest row that exists.
        """
        if not chunk_ids:
            return {}
        cache: dict[str, dict[str, Any]] = {}
        pending = set(chunk_ids)
        while pending:
            placeholders = ",".join("?" for _ in pending)
            cache_size_before = len(cache)
            with self._txn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {_ANCESTOR_PROJECTION}
                    FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id IN ({placeholders})
                    """,
                    tuple(pending),
                ).fetchall()
            for row in rows:
                cache[row["chunk_id"]] = dict(row)
            if len(cache) == cache_size_before:
                # Every remaining id points at a row that does not
                # exist; another round could only repeat these fetches.
                break
            next_pending: set[str] = set()
            for row in cache.values():
                if row["level"] == level or not row["parent_id"]:
                    continue
                if row["parent_id"] not in cache:
                    next_pending.add(row["parent_id"])
            pending = next_pending - set(cache)

        out: dict[str, dict[str, Any]] = {}
        for leaf_id in chunk_ids:
            current = cache.get(leaf_id)
            if current is None:
                continue
            seen = {leaf_id}
            while current["level"] != level and current["parent_id"]:
                parent = cache.get(current["parent_id"])
                if parent is None or current["parent_id"] in seen:
                    break
                seen.add(current["parent_id"])
                current = parent
            fields = {name: current[name] for name in CONTENT_FIELDS}
            fields.update(
                level=current["level"],
                char_start=current["char_start"],
                char_end=current["char_end"],
            )
            out[leaf_id] = {"chunk_id": current["chunk_id"], **fields}
        return out

    def hydrate(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return content fields for many chunks, keyed by ``chunk_id``."""
        if not chunk_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self._txn() as conn:
            # Chunk in arbitrary order; fetch in one query with an IN list,
            # then key the result.
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                f"""
                SELECT {_JOIN_PROJECTION}
                FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        for row in rows:
            out[row["chunk_id"]] = self._assemble(row)
        return out

    def list_leaf_chunks(
        self, database: str, collection: str
    ) -> list[dict[str, Any]]:
        """Enumerate every leaf chunk of one logical collection.

        Rebuild/repair input — one row per ``level = 'chunk'`` chunk
        with the content needed to rederive every vector: the stored
        context prefix (dense embedding input), the summary (summary
        vector input, text fallback) and token accounting. Source
        order matches ``leaf_chunk_texts``.
        """
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, text, context, summary,
                       token_count
                FROM chunks
                WHERE database = ? AND collection = ? AND level = 'chunk'
                ORDER BY doc_id, chunk_index
                """,
                (database, collection),
            ).fetchall()
        return [dict(row) for row in rows]

    def leaf_chunk_texts(self, database: str, collection: str) -> list[str]:
        """All leaf-chunk texts in source order (BM25 fit input).

        Parent rows (``document`` / ``section``) are excluded: BM25
        statistics describe the leaf population that actually enters
        the sparse index, so parent text never double-counts a term.
        """
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT text FROM chunks
                WHERE database = ? AND collection = ? AND level = 'chunk'
                ORDER BY doc_id, chunk_index
                """,
                (database, collection),
            ).fetchall()
        return [row["text"] for row in rows]

    def doc_ids_for_filename(
        self, database: str, collection: str, filename_needle: str
    ) -> list[str]:
        """Resolve a filename LIKE needle to the matching document ids.

        Lets retrieval keep its filter on the thin ``doc_id`` scalar in the
        vector index while the filename itself lives only in the corpus.
        """
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT doc_id FROM documents
                WHERE database = ? AND collection = ? AND filename LIKE ?
                """,
                (database, collection, f"%{filename_needle}%"),
            ).fetchall()
        return [row["doc_id"] for row in rows]

    def doc_ids_for_metadata(
        self,
        database: str,
        collection: str,
        predicates: list[dict[str, str]],
    ) -> list[str]:
        """Resolve structured document predicates to matching doc ids.

        Predicates target documents-table columns: text columns
        (filename / title / author / status / mime) with ``like`` become
        substring matches (other ops compare exact values); page_count
        comparisons use the integer column. Conditions combine with AND.
        Keeps retrieval's filter on the thin ``doc_id`` scalar while the
        metadata itself lives only in the corpus.
        """
        clauses: list[str] = []
        params: list[Any] = []
        text_fields = ("filename", "title", "author", "status", "mime")
        for predicate in predicates:
            field = predicate["field"]
            op = predicate["op"]
            value = predicate["value"]
            if field not in (*text_fields, "page_count"):
                raise ValueError(f"metadata field {field!r} not resolvable")
            if op not in ("like", "==", "!=", ">=", "<=", ">", "<"):
                raise ValueError(f"metadata op {op!r} not supported")
            if field in text_fields:
                if op == "like":
                    clauses.append(f"{field} LIKE ?")
                    params.append(f"%{value}%")
                else:
                    clauses.append(f"{field} {op} ?")
                    params.append(value)
            else:
                clauses.append(f"page_count {op} CAST(? AS INTEGER)")
                params.append(int(value))

        where = " AND ".join(clauses) if clauses else "1 = 1"
        query = f"""
            SELECT DISTINCT doc_id FROM documents
            WHERE database = ? AND collection = ? AND {where}
        """
        with self._txn() as conn:
            rows = conn.execute(query, [database, collection, *params]).fetchall()
        return [row["doc_id"] for row in rows]

    # ---- retrieval feedback ----

    def add_feedback(
        self,
        feedback_id: str,
        *,
        database: str,
        collection: str,
        query: str,
        chunk_id: str | None,
        kind: str,
        pipeline_json: str = "{}",
        comment: str | None = None,
        now: float | None = None,
    ) -> None:
        """Persist one retrieval feedback event.

        ``chunk_id`` is null for answer-level (👍/👎) feedback;
        ``pipeline_json`` snapshots the pipeline version (channels,
        models, params, index_ref, routing) the feedback refers to.
        """
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO feedback
                    (feedback_id, database, collection, query, chunk_id,
                     kind, pipeline_json, comment, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id, database, collection, query, chunk_id,
                    kind, pipeline_json, comment, now,
                ),
            )

    def list_feedback(
        self,
        *,
        database: str | None = None,
        collection: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List feedback newest first with optional scope/kind filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if database is not None:
            clauses.append("database = ?")
            params.append(database)
        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._txn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM feedback {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM feedback {where}
                ORDER BY created_ts DESC, feedback_id
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    # ---- evaluation sets -----------------------------------------------

    def create_eval_set(
        self,
        set_id: str,
        *,
        database: str,
        collection: str,
        name: str,
        description: str = "",
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO eval_sets
                    (set_id, database, collection, name, description, created_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (set_id, database, collection, name, description, now),
            )

    def get_eval_set(self, set_id: str) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_sets WHERE set_id = ?", (set_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_eval_sets(
        self,
        *,
        database: str | None = None,
        collection: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List eval sets newest-first with optional scope filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if database is not None:
            clauses.append("database = ?")
            params.append(database)
        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._txn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM eval_sets {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM eval_sets {where}
                ORDER BY created_ts DESC, set_id
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def delete_eval_set(self, set_id: str) -> None:
        """Delete the set; questions, runs and results cascade."""
        with self._txn() as conn:
            conn.execute("DELETE FROM eval_sets WHERE set_id = ?", (set_id,))

    def add_eval_questions(
        self, questions: list[dict[str, Any]]
    ) -> None:
        """Bulk insert questions for one set.

        Each item carries ``question_id``, ``set_id``, ``question``,
        ``expected_chunk_ids`` / ``expected_doc_ids`` (already JSON
        strings) and ``expected_answer``.
        """
        now = time.time()
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT INTO eval_questions
                    (question_id, set_id, question, expected_chunk_ids,
                     expected_doc_ids, expected_answer, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        q["question_id"], q["set_id"], q["question"],
                        q["expected_chunk_ids"], q["expected_doc_ids"],
                        q["expected_answer"], now,
                    )
                    for q in questions
                ],
            )

    def list_eval_questions(self, set_id: str) -> list[dict[str, Any]]:
        """Questions of a set in insertion order."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM eval_questions
                WHERE set_id = ?
                ORDER BY rowid
                """,
                (set_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- evaluation runs -----------------------------------------------

    def create_eval_run(
        self,
        run_id: str,
        *,
        set_id: str,
        params_json: str = "{}",
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO eval_runs (run_id, set_id, params_json, created_ts)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, set_id, params_json, now),
            )

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_eval_runs(self, set_id: str) -> list[dict[str, Any]]:
        """Runs of a set newest-first."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM eval_runs
                WHERE set_id = ?
                ORDER BY created_ts DESC, run_id
                """,
                (set_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def replace_eval_results(
        self, run_id: str, results: list[dict[str, Any]]
    ) -> None:
        """Replace a run's results wholesale (a retry restarts the run).

        Each item carries ``question_id``, ``chunk_ids`` (JSON string),
        ``metrics_json``, ``answer`` / ``answer_error``.
        """
        now = time.time()
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM eval_results WHERE run_id = ?", (run_id,)
            )
            conn.executemany(
                """
                INSERT INTO eval_results
                    (run_id, question_id, chunk_ids, metrics_json, answer,
                     answer_error, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id, r["question_id"], r["chunk_ids"],
                        r["metrics_json"], r["answer"], r["answer_error"], now,
                    )
                    for r in results
                ],
            )

    def list_eval_results(self, run_id: str) -> list[dict[str, Any]]:
        """Results of a run in question-loop order.

        Ordered by the results' own rowid rather than a join on
        eval_questions: versioned runs snapshot questions that may no
        longer exist as live rows.
        """
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT r.*
                FROM eval_results r
                WHERE r.run_id = ?
                ORDER BY r.rowid
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- frozen versions ------------------------------------------------

    def create_eval_version(
        self,
        version_id: str,
        *,
        set_id: str,
        tag: str,
        question_count: int,
        snapshot_json: str,
        now: float | None = None,
    ) -> None:
        """Freeze a set's questions into an immutable version snapshot."""
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO eval_set_versions
                    (version_id, set_id, tag, question_count,
                     snapshot_json, created_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id, set_id, tag, question_count,
                    snapshot_json, now,
                ),
            )

    def get_eval_version(
        self, version_id: str
    ) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM eval_set_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_eval_version_by_tag(
        self, set_id: str, tag: str
    ) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT * FROM eval_set_versions
                WHERE set_id = ? AND tag = ?
                """,
                (set_id, tag),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_eval_versions(self, set_id: str) -> list[dict[str, Any]]:
        """Frozen versions of a set, newest first."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM eval_set_versions
                WHERE set_id = ?
                ORDER BY created_ts DESC, version_id
                """,
                (set_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ---- regression gates -----------------------------------------------

    def create_regression_gate(
        self,
        gate_id: str,
        *,
        database: str,
        collection: str,
        set_id: str,
        version_id: str,
        template_json: str = "{}",
        min_recall: float | None = None,
        min_mrr: float | None = None,
        min_ndcg: float | None = None,
        max_recall_drop: float | None = None,
        max_mrr_drop: float | None = None,
        max_ndcg_drop: float | None = None,
        baseline_run_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Register a regression gate for one logical collection."""
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO regression_gates
                    (gate_id, database, collection, set_id, version_id,
                     template_json, min_recall, min_mrr, min_ndcg,
                     max_recall_drop, max_mrr_drop, max_ndcg_drop,
                     baseline_run_id, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate_id, database, collection, set_id, version_id,
                    template_json, min_recall, min_mrr, min_ndcg,
                    max_recall_drop, max_mrr_drop, max_ndcg_drop,
                    baseline_run_id, now, now,
                ),
            )

    def get_regression_gate(
        self, gate_id: str
    ) -> dict[str, Any] | None:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM regression_gates WHERE gate_id = ?",
                (gate_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_gate_for_collection(
        self, database: str, collection: str
    ) -> dict[str, Any] | None:
        """The registered gate for a logical collection, if any."""
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT * FROM regression_gates
                WHERE database = ? AND collection = ?
                """,
                (database, collection),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_regression_gates(
        self, *, database: str | None = None, collection: str | None = None
    ) -> list[dict[str, Any]]:
        """Gates with optional scope filter, newest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if database is not None:
            clauses.append("database = ?")
            params.append(database)
        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._txn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM regression_gates
                {where}
                ORDER BY created_ts DESC, gate_id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_regression_gate(self, gate_id: str) -> None:
        """Delete a gate and its checks."""
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM regression_gates WHERE gate_id = ?",
                (gate_id,),
            )

    # ---- gate checks ----------------------------------------------------

    def create_gate_check(
        self,
        check_id: str,
        *,
        gate_id: str,
        run_id: str | None,
        candidate_ref: str,
        status: str,
        report_json: str = "{}",
        now: float | None = None,
    ) -> None:
        """Record a gate check (running/passed/failed)."""
        now = time.time() if now is None else now
        with self._txn() as conn:
            conn.execute(
                """
                INSERT INTO gate_checks
                    (check_id, gate_id, run_id, candidate_ref, status,
                     report_json, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    check_id, gate_id, run_id, candidate_ref, status,
                    report_json, now,
                ),
            )

    def update_gate_check(
        self,
        check_id: str,
        *,
        run_id: str | None = None,
        status: str,
        report_json: str,
    ) -> None:
        """Finish a check with its status and report."""
        with self._txn() as conn:
            conn.execute(
                """
                UPDATE gate_checks
                SET run_id = ?, status = ?, report_json = ?
                WHERE check_id = ?
                """,
                (run_id, status, report_json, check_id),
            )

    def get_latest_gate_check(
        self, gate_id: str
    ) -> dict[str, Any] | None:
        """Most recent check for a gate, if any."""
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT * FROM gate_checks
                WHERE gate_id = ?
                ORDER BY created_ts DESC, check_id
                LIMIT 1
                """,
                (gate_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_gate_checks(self, gate_id: str) -> list[dict[str, Any]]:
        """Checks of a gate, newest first."""
        with self._txn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM gate_checks
                WHERE gate_id = ?
                ORDER BY created_ts DESC, check_id
                """,
                (gate_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def is_corpus_collection(self, database: str, collection: str) -> bool:
        with self._txn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM documents
                WHERE database = ? AND collection = ? LIMIT 1
                """,
                (database, collection),
            ).fetchone()
        return row is not None

    def browse(
        self, database: str, collection: str, *, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Page over chunks with content assembled from the corpus.

        Returns rows shaped ``{"id": chunk_id, "fields": {content fields}}``
        plus a total count, matching the generic management browse output.
        """
        with self._txn() as conn:
            total = conn.execute(
                """
                SELECT COUNT(*) FROM chunks
                WHERE database = ? AND collection = ?
                """,
                (database, collection),
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT {_JOIN_PROJECTION}
                FROM chunks c JOIN documents d ON c.doc_id = d.doc_id
                WHERE c.database = ? AND c.collection = ?
                ORDER BY c.rowid
                LIMIT ? OFFSET ?
                """,
                (database, collection, limit, offset),
            ).fetchall()
        items = [{"id": row["chunk_id"], "fields": self._assemble(row)} for row in rows]
        return items, int(total)

    # ---- GraphRAG graph -------------------------------------------------

    @staticmethod
    def _prune_graph_for_chunks(conn: Any, chunk_ids: list[str]) -> None:
        """Drop graph provenance for the chunks, then graph rows left orphan.

        Mentions go first; edges with no remaining provenance are deleted;
        entities with no remaining mentions are deleted and cascade their
        claims, incident edges and community memberships; communities
        whose last member disappeared are deleted too.
        """
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(
            f"DELETE FROM entity_mentions WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        conn.execute(
            f"DELETE FROM edge_mentions WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        conn.execute(
            """
            DELETE FROM edges
            WHERE NOT EXISTS (
                SELECT 1 FROM edge_mentions m WHERE m.edge_id = edges.edge_id
            )
            """
        )
        conn.execute(
            """
            DELETE FROM entities
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.entity_id
            )
            """
        )
        conn.execute(
            """
            DELETE FROM communities
            WHERE NOT EXISTS (
                SELECT 1 FROM community_members m
                WHERE m.community_id = communities.community_id
            )
            """
        )

    @staticmethod
    def _delete_graph_scope(
        conn: Any, *, database: str, collection: str | None
    ) -> None:
        """Delete every graph row of one collection or whole database."""
        if collection is None:
            scope = "database = ?"
            params: tuple[Any, ...] = (database,)
        else:
            scope = "database = ? AND collection = ?"
            params = (database, collection)
        conn.execute(
            f"""
            DELETE FROM edge_mentions WHERE edge_id IN (
                SELECT edge_id FROM edges WHERE {scope}
            )
            """,
            params,
        )
        conn.execute(
            f"""
            DELETE FROM entity_mentions WHERE entity_id IN (
                SELECT entity_id FROM entities WHERE {scope}
            )
            """,
            params,
        )
        conn.execute(
            f"""
            DELETE FROM community_members WHERE community_id IN (
                SELECT community_id FROM communities WHERE {scope}
            )
            """,
            params,
        )
        conn.execute(f"DELETE FROM claims WHERE {scope}", params)
        conn.execute(f"DELETE FROM edges WHERE {scope}", params)
        conn.execute(f"DELETE FROM entities WHERE {scope}", params)
        conn.execute(f"DELETE FROM communities WHERE {scope}", params)

    def delete_graph(self, database: str, collection: str) -> None:
        """Delete every graph row of one collection."""
        with self._txn() as conn:
            self._delete_graph_scope(
                conn, database=database, collection=collection
            )

    def replace_graph(
        self,
        database: str,
        collection: str,
        *,
        entities: Any,
        entity_mentions: Any,
        edges: Any,
        edge_mentions: Any,
        claims: Any,
        communities: Any,
        community_members: Any,
    ) -> None:
        """Replace the whole derived graph of one collection in one txn.

        Inputs are iterables of duck-typed records (the graph package's
        build output); a failed insert rolls the previous graph back.
        """
        with self._txn() as conn:
            self._delete_graph_scope(conn, database=database, collection=collection)
            conn.executemany(
                """
                INSERT INTO entities
                    (entity_id, database, collection, name, entity_type,
                     description, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.entity_id, database, collection, e.name,
                        e.entity_type, e.description, e.created_ts, e.updated_ts,
                    )
                    for e in entities
                ],
            )
            conn.executemany(
                "INSERT INTO entity_mentions (entity_id, chunk_id) VALUES (?, ?)",
                [(m[0], m[1]) for m in entity_mentions],
            )
            conn.executemany(
                """
                INSERT INTO edges
                    (edge_id, database, collection, source_id, target_id,
                     description, weight, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        x.edge_id, database, collection, x.source_id,
                        x.target_id, x.description, x.weight,
                        x.created_ts, x.updated_ts,
                    )
                    for x in edges
                ],
            )
            conn.executemany(
                "INSERT INTO edge_mentions (edge_id, chunk_id) VALUES (?, ?)",
                [(m[0], m[1]) for m in edge_mentions],
            )
            conn.executemany(
                """
                INSERT INTO claims
                    (claim_id, database, collection, subject_id, object_id,
                     chunk_id, claim_type, status, statement, created_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.claim_id, database, collection, c.subject_id,
                        c.object_id, c.chunk_id, c.claim_type, c.status,
                        c.statement, c.created_ts,
                    )
                    for c in claims
                ],
            )
            conn.executemany(
                """
                INSERT INTO communities
                    (community_id, database, collection, community_index,
                     summary, created_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        m.community_id, database, collection, m.community_index,
                        m.summary, m.created_ts,
                    )
                    for m in communities
                ],
            )
            conn.executemany(
                """
                INSERT INTO community_members (community_id, entity_id)
                VALUES (?, ?)
                """,
                [(m[0], m[1]) for m in community_members],
            )

    def graph_stats(self, database: str, collection: str) -> dict[str, Any]:
        """Counts plus the community list for the graph status route."""
        with self._txn() as conn:
            counts = {
                name: conn.execute(
                    f"SELECT COUNT(*) FROM {name} "
                    "WHERE database = ? AND collection = ?",
                    (database, collection),
                ).fetchone()[0]
                for name in ("entities", "edges", "claims", "communities")
            }
            communities = conn.execute(
                """
                SELECT cm.community_id, cm.community_index, cm.summary,
                       COUNT(m.entity_id) AS size
                FROM communities cm
                LEFT JOIN community_members m ON m.community_id = cm.community_id
                WHERE cm.database = ? AND cm.collection = ?
                GROUP BY cm.community_id
                ORDER BY cm.community_index
                """,
                (database, collection),
            ).fetchall()
        return {
            **{f"{k}_count": int(v) for k, v in counts.items()},
            "communities": [dict(row) for row in communities],
        }

    def hydrate_entities(
        self, entity_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Return name/type/description for entities, keyed by entity id."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        with self._txn() as conn:
            rows = conn.execute(
                f"""
                SELECT entity_id, name, entity_type, description
                FROM entities WHERE entity_id IN ({placeholders})
                """,
                entity_ids,
            ).fetchall()
        return {
            row["entity_id"]: {
                "name": row["name"],
                "entity_type": row["entity_type"],
                "description": row["description"],
            }
            for row in rows
        }

    def get_entity_subgraph(
        self, entity_ids: list[str], *, depth: int = 1
    ) -> dict[str, Any]:
        """BFS subgraph around ``entity_ids`` through the edges table.

        ``depth=1`` includes direct neighbors. Returns
        ``{"nodes": [ids, seeds first in given order], "edges": [edge
        rows]}``; traversal is capped at :data:`_SUBGRAPH_NODE_CAP` nodes
        so a hub entity cannot pull the entire graph.
        """
        if not entity_ids:
            return {"nodes": [], "edges": []}
        nodes: list[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            if entity_id not in seen:
                seen.add(entity_id)
                nodes.append(entity_id)
        frontier = list(nodes)
        edges_by_id: dict[str, dict[str, Any]] = {}
        for _ in range(depth):
            if not frontier or len(seen) >= _SUBGRAPH_NODE_CAP:
                break
            placeholders = ",".join("?" for _ in frontier)
            with self._txn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT edge_id, source_id, target_id, description, weight
                    FROM edges
                    WHERE source_id IN ({placeholders})
                          OR target_id IN ({placeholders})
                    """,
                    (*frontier, *frontier),
                ).fetchall()
            next_frontier: list[str] = []
            for row in rows:
                edges_by_id.setdefault(row["edge_id"], dict(row))
                for endpoint in (row["source_id"], row["target_id"]):
                    if endpoint not in seen:
                        seen.add(endpoint)
                        nodes.append(endpoint)
                        next_frontier.append(endpoint)
                        if len(seen) >= _SUBGRAPH_NODE_CAP:
                            break
            frontier = next_frontier
        return {"nodes": nodes, "edges": list(edges_by_id.values())}

    def hydrate_communities(
        self, community_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Return index/summary/member entity ids for communities."""
        if not community_ids:
            return {}
        placeholders = ",".join("?" for _ in community_ids)
        with self._txn() as conn:
            rows = conn.execute(
                f"""
                SELECT community_id, community_index, summary
                FROM communities WHERE community_id IN ({placeholders})
                """,
                community_ids,
            ).fetchall()
            members = conn.execute(
                f"""
                SELECT community_id, entity_id
                FROM community_members WHERE community_id IN ({placeholders})
                """,
                community_ids,
            ).fetchall()
        member_ids: dict[str, list[str]] = {}
        for row in members:
            member_ids.setdefault(row["community_id"], []).append(row["entity_id"])
        return {
            row["community_id"]: {
                "community_index": row["community_index"],
                "summary": row["summary"],
                "entities": member_ids.get(row["community_id"], []),
            }
            for row in rows
        }

    def claims_for_entities(
        self, entity_ids: list[str], *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Claims whose subject/object is one of the entities."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        with self._txn() as conn:
            rows = conn.execute(
                f"""
                SELECT claim_id, subject_id, object_id, chunk_id, claim_type,
                       status, statement, created_ts
                FROM claims
                WHERE subject_id IN ({placeholders})
                      OR object_id IN ({placeholders})
                ORDER BY created_ts, claim_id
                LIMIT ?
                """,
                (*entity_ids, *entity_ids, limit),
            ).fetchall()
        return [dict(row) for row in rows]


_SUBGRAPH_NODE_CAP = 500


# Projection for the chunks⋈documents reads: chunk-sourced columns from
# ``c``, document-level columns (title / author / page_count / filename)
# from ``d``. Output names cover every key in CONTENT_FIELDS plus chunk_id.
_JOIN_PROJECTION = ", ".join(  # noqa: FLY002 — joining literals, no f-string applicable
    (
        "c.chunk_id",
        "c.doc_id",
        "c.chunk_index",
        "c.text",
        "c.section_header",
        "c.page_number",
        "c.char_start",
        "c.char_end",
        "c.token_count",
        "c.summary",
        "d.title AS title",
        "d.author AS author",
        "d.page_count AS page_count",
        "d.filename AS filename",
    )
)

# Ancestor-chain projection: the join fields plus the links the walk
# needs (parent_id / level / char spans).
_ANCESTOR_PROJECTION = ", ".join(  # noqa: FLY002 — joining literals, no f-string applicable
    (
        _JOIN_PROJECTION,
        "c.parent_id AS parent_id",
        "c.level AS level",
        "c.char_start AS char_start",
        "c.char_end AS char_end",
    )
)
