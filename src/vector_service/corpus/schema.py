"""SQLite DDL for the corpus layer.

Schema versioning follows the project's no-backward-compat policy: tables
are created idempotently and ``PRAGMA user_version`` records the schema
generation. A future breaking change recreates or migrates destructively
rather than maintaining compatibility branches.
"""

from __future__ import annotations

SCHEMA_VERSION = 9

# One ordered tuple so ``initialize`` can execute every statement without
# the caller knowing how many tables exist.
DDL: tuple[str, ...] = (
    # ---- documents ----
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id       TEXT PRIMARY KEY,
        database     TEXT NOT NULL,
        collection   TEXT NOT NULL,
        filename     TEXT,
        mime         TEXT,
        content_hash TEXT NOT NULL,
        title        TEXT,
        author       TEXT,
        page_count   INTEGER,
        status       TEXT NOT NULL DEFAULT 'active',
        created_ts   REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents(database, collection)",
    "CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents(content_hash)",
    # ---- chunks (content + hierarchy scaffold) ----
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id       TEXT PRIMARY KEY,
        doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        database       TEXT NOT NULL,
        collection     TEXT NOT NULL,
        chunk_index    INTEGER NOT NULL,
        text           TEXT NOT NULL,
        section_header TEXT NOT NULL DEFAULT '',
        page_number    INTEGER,
        token_count    INTEGER NOT NULL DEFAULT 0,
        context        TEXT,
        summary        TEXT,
        parent_id      TEXT,
        level          TEXT NOT NULL DEFAULT 'chunk',
        char_start     INTEGER,
        char_end       INTEGER,
        created_ts     REAL NOT NULL,
        UNIQUE(doc_id, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(doc_id)",
    "CREATE INDEX IF NOT EXISTS chunks_corpus_order_idx ON chunks(database, collection, doc_id, chunk_index)",
    "CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_id)",
    # ---- registry of derived indexes per chunk ----
    """
    CREATE TABLE IF NOT EXISTS chunk_indexes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id   TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        doc_id     TEXT NOT NULL,
        database   TEXT NOT NULL,
        collection TEXT NOT NULL,
        index_kind TEXT NOT NULL,
        model      TEXT NOT NULL,
        index_ref  TEXT NOT NULL,
        created_ts REAL NOT NULL,
        -- One current index per (chunk, kind) within a collection. A
        -- model switch rebuilds that kind and replaces the row; running
        -- two models side by side means separate collections instead.
        UNIQUE(chunk_id, index_kind)
    )
    """,
    "CREATE INDEX IF NOT EXISTS chunk_indexes_doc_idx ON chunk_indexes(doc_id)",
    "CREATE INDEX IF NOT EXISTS chunk_indexes_corpus_idx ON chunk_indexes(database, collection)",
    # ---- active / canary physical index per logical collection ----
    # A rebuild populates a fresh physical collection and parks it here
    # as the canary; promote moves it to active and rewrites
    # chunk_indexes. With no binding row the logical collection name is
    # itself the physical index.
    """
    CREATE TABLE IF NOT EXISTS index_bindings (
        database       TEXT NOT NULL,
        collection     TEXT NOT NULL,
        active_ref     TEXT NOT NULL,
        canary_ref     TEXT,
        canary_percent INTEGER NOT NULL DEFAULT 0,
        model          TEXT NOT NULL,
        created_ts     REAL NOT NULL,
        updated_ts     REAL NOT NULL,
        PRIMARY KEY (database, collection)
    )
    """,
    # ---- ingest jobs ----
    # v3 widened the table for async submission + background worker:
    # doc_id/params are known at submit time, spool_path owns the raw
    # upload, attempts/cancel/progress drive the worker.
    # v4 added not_before_ts (retry backoff gate). v5 added job_type
    # ('ingest' / 'rebuild') so the same worker owns index rebuilds.
    # Each bumped with a destructive rebuild (see db._BREAKING_CHANGES).
    """
    CREATE TABLE IF NOT EXISTS ingest_jobs (
        job_id          TEXT PRIMARY KEY,
        job_type        TEXT NOT NULL DEFAULT 'ingest',
        doc_id          TEXT,
        database        TEXT NOT NULL,
        collection      TEXT NOT NULL,
        filename        TEXT,
        mime            TEXT,
        status          TEXT NOT NULL,
        params_json     TEXT NOT NULL DEFAULT '{}',
        spool_path      TEXT,
        attempts        INTEGER NOT NULL DEFAULT 0,
        max_attempts    INTEGER NOT NULL DEFAULT 3,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        progress_current INTEGER,
        progress_total  INTEGER,
        chunk_count     INTEGER NOT NULL DEFAULT 0,
        page_count      INTEGER,
        tokens_used     INTEGER NOT NULL DEFAULT 0,
        error_code      TEXT,
        error_message   TEXT,
        not_before_ts   REAL,
        created_ts      REAL NOT NULL,
        updated_ts      REAL NOT NULL,
        finished_ts     REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx ON ingest_jobs(status)",
    "CREATE INDEX IF NOT EXISTS ingest_jobs_spool_idx ON ingest_jobs(status, created_ts)",
    # ---- GraphRAG knowledge graph ----
    # All graph rows are derived from the leaf chunks and replaced wholesale
    # by a graph build; entities are merged by name within one logical
    # collection, mentions tables carry chunk provenance, communities are
    # the detected clusters whose LLM summaries get their own ANN index.
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id    TEXT PRIMARY KEY,
        database     TEXT NOT NULL,
        collection   TEXT NOT NULL,
        name         TEXT NOT NULL,
        entity_type  TEXT NOT NULL DEFAULT '',
        description  TEXT NOT NULL DEFAULT '',
        created_ts   REAL NOT NULL,
        updated_ts   REAL NOT NULL,
        UNIQUE(database, collection, name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS entities_corpus_idx ON entities(database, collection)",
    """
    CREATE TABLE IF NOT EXISTS entity_mentions (
        entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        chunk_id  TEXT NOT NULL,
        PRIMARY KEY (entity_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS entity_mentions_chunk_idx ON entity_mentions(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        edge_id      TEXT PRIMARY KEY,
        database     TEXT NOT NULL,
        collection   TEXT NOT NULL,
        source_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        target_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        description  TEXT NOT NULL DEFAULT '',
        weight       INTEGER NOT NULL DEFAULT 1,
        created_ts   REAL NOT NULL,
        updated_ts   REAL NOT NULL,
        UNIQUE(source_id, target_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS edges_corpus_idx ON edges(database, collection)",
    "CREATE INDEX IF NOT EXISTS edges_target_idx ON edges(target_id)",
    """
    CREATE TABLE IF NOT EXISTS edge_mentions (
        edge_id  TEXT NOT NULL REFERENCES edges(edge_id) ON DELETE CASCADE,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (edge_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS edge_mentions_chunk_idx ON edge_mentions(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS claims (
        claim_id    TEXT PRIMARY KEY,
        database    TEXT NOT NULL,
        collection  TEXT NOT NULL,
        subject_id  TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        object_id   TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
        chunk_id    TEXT NOT NULL,
        claim_type  TEXT NOT NULL DEFAULT '',
        status      TEXT NOT NULL DEFAULT '',
        statement   TEXT NOT NULL,
        created_ts  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS claims_corpus_idx ON claims(database, collection)",
    "CREATE INDEX IF NOT EXISTS claims_subject_idx ON claims(subject_id)",
    """
    CREATE TABLE IF NOT EXISTS communities (
        community_id    TEXT PRIMARY KEY,
        database        TEXT NOT NULL,
        collection      TEXT NOT NULL,
        community_index INTEGER NOT NULL,
        summary         TEXT NOT NULL DEFAULT '',
        created_ts      REAL NOT NULL,
        UNIQUE(database, collection, community_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS community_members (
        community_id TEXT NOT NULL REFERENCES communities(community_id) ON DELETE CASCADE,
        entity_id    TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        PRIMARY KEY (community_id, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS community_members_entity_idx ON community_members(entity_id)",
    # ---- retrieval feedback (v7) ----
    """
    CREATE TABLE IF NOT EXISTS feedback (
        feedback_id   TEXT PRIMARY KEY,
        database      TEXT NOT NULL,
        collection    TEXT NOT NULL,
        query         TEXT NOT NULL,
        chunk_id      TEXT,
        kind          TEXT NOT NULL,
        pipeline_json TEXT NOT NULL DEFAULT '{}',
        comment       TEXT,
        created_ts    REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS feedback_corpus_idx ON feedback(database, collection)",
    "CREATE INDEX IF NOT EXISTS feedback_kind_idx ON feedback(kind)",
    # ---- evaluation sets / runs (v8) ----
    # Like feedback, evaluation is corpus metadata and never enters
    # Milvus. A set is scoped to one (database, collection); its
    # questions carry expected leaf chunks / docs / answer. Runs batch
    # the shared RetrievalPipeline over every question; per-question
    # ranked ids, basic metrics and optional generated answers land as
    # run results. All additive tables — cascade from eval_sets.
    """
    CREATE TABLE IF NOT EXISTS eval_sets (
        set_id      TEXT PRIMARY KEY,
        database    TEXT NOT NULL,
        collection  TEXT NOT NULL,
        name        TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        created_ts  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_sets_corpus_idx ON eval_sets(database, collection)",
    """
    CREATE TABLE IF NOT EXISTS eval_questions (
        question_id        TEXT PRIMARY KEY,
        set_id             TEXT NOT NULL REFERENCES eval_sets(set_id) ON DELETE CASCADE,
        question           TEXT NOT NULL,
        expected_chunk_ids TEXT NOT NULL DEFAULT '[]',
        expected_doc_ids   TEXT NOT NULL DEFAULT '[]',
        expected_answer    TEXT,
        created_ts         REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_questions_set_idx ON eval_questions(set_id)",
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        run_id      TEXT PRIMARY KEY,
        set_id      TEXT NOT NULL REFERENCES eval_sets(set_id) ON DELETE CASCADE,
        params_json TEXT NOT NULL DEFAULT '{}',
        created_ts  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_runs_set_idx ON eval_runs(set_id)",
    """
    CREATE TABLE IF NOT EXISTS eval_results (
        result_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
        question_id  TEXT NOT NULL,
        chunk_ids    TEXT NOT NULL DEFAULT '[]',
        metrics_json TEXT NOT NULL DEFAULT '{}',
        answer       TEXT,
        answer_error TEXT,
        created_ts   REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results(run_id)",
    # ---- frozen dataset versions + regression gates (v9) ----
    # A version freezes a set's questions and expectations into an
    # immutable snapshot. A gate binds one (database, collection) to a
    # version + thresholds; a gate check runs the questions against the
    # parked canary and must pass before promotion.
    """
    CREATE TABLE IF NOT EXISTS eval_set_versions (
        version_id     TEXT PRIMARY KEY,
        set_id         TEXT NOT NULL REFERENCES eval_sets(set_id) ON DELETE CASCADE,
        tag            TEXT NOT NULL,
        question_count INTEGER NOT NULL,
        snapshot_json  TEXT NOT NULL,
        created_ts     REAL NOT NULL,
        UNIQUE(set_id, tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS eval_set_versions_set_idx ON eval_set_versions(set_id)",
    """
    CREATE TABLE IF NOT EXISTS regression_gates (
        gate_id         TEXT PRIMARY KEY,
        database        TEXT NOT NULL,
        collection      TEXT NOT NULL,
        set_id          TEXT NOT NULL REFERENCES eval_sets(set_id) ON DELETE CASCADE,
        version_id      TEXT NOT NULL REFERENCES eval_set_versions(version_id) ON DELETE CASCADE,
        template_json   TEXT NOT NULL DEFAULT '{}',
        min_recall      REAL,
        min_mrr         REAL,
        min_ndcg        REAL,
        max_recall_drop REAL,
        max_mrr_drop    REAL,
        max_ndcg_drop   REAL,
        baseline_run_id TEXT,
        created_ts      REAL NOT NULL,
        updated_ts      REAL NOT NULL,
        UNIQUE(database, collection)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gate_checks (
        check_id      TEXT PRIMARY KEY,
        gate_id       TEXT NOT NULL REFERENCES regression_gates(gate_id) ON DELETE CASCADE,
        run_id        TEXT,
        candidate_ref TEXT NOT NULL,
        status        TEXT NOT NULL,
        report_json   TEXT NOT NULL DEFAULT '{}',
        created_ts    REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS gate_checks_gate_idx ON gate_checks(gate_id)",
)
