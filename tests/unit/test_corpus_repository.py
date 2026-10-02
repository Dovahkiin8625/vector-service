"""Unit tests for ``CorpusRepository`` against a temporary SQLite file.

Pins the system-of-record surface: job lifecycle, document/chunk writes,
content hydration, filename resolution, derived-index registry and the
cascade delete that ingest rollback relies on.
"""

from __future__ import annotations

import sqlite3

import pytest

from vector_service.corpus import (
    JOB_DONE,
    JOB_FAILED,
    JOB_QUEUED,
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
    IndexEntry,
)


@pytest.fixture
def repo(tmp_path):
    r = CorpusRepository(tmp_path / "corpus.db")
    r.initialize()
    # initialize must be idempotent (lifespan restarts on the same file).
    r.initialize()
    return r


def _document(
    doc_id="d1", *, database="default", collection="ingest", filename="年报.pdf"
):
    return DocumentRecord(
        doc_id=doc_id,
        database=database,
        collection=collection,
        filename=filename,
        mime="application/pdf",
        content_hash="a" * 64,
        title="年度报告",
        author="张三",
        page_count=10,
    )


def _chunks(doc_id="d1", *, database="default", collection="ingest"):
    return [
        ChunkRecord(
            chunk_id=f"{doc_id}_0",
            doc_id=doc_id,
            database=database,
            collection=collection,
            chunk_index=0,
            text="第一段内容",
            section_header="引言",
            page_number=1,
            char_start=0,
            char_end=100,
            token_count=5,
        ),
        ChunkRecord(
            chunk_id=f"{doc_id}_1",
            doc_id=doc_id,
            database=database,
            collection=collection,
            chunk_index=1,
            text="第二段内容",
            section_header="正文",
            page_number=2,
            char_start=100,
            char_end=200,
            token_count=5,
            summary="摘要文本",
        ),
    ]


# ---- ingest jobs -------------------------------------------------------


def test_job_lifecycle_queued_to_done(repo):
    repo.create_job(
        "job1",
        database="default",
        collection="ingest",
        filename="年报.pdf",
        mime="application/pdf",
        now=1.0,
    )
    job = repo.get_job("job1")
    assert job["status"] == JOB_QUEUED
    assert job["database"] == "default"

    repo.mark_job("job1", "parsing", now=2.0)
    assert repo.get_job("job1")["status"] == "parsing"

    repo.finish_job("job1", chunk_count=2, page_count=10, tokens_used=42, now=3.0)
    job = repo.get_job("job1")
    assert job["status"] == JOB_DONE
    assert job["chunk_count"] == 2
    assert job["page_count"] == 10
    assert job["tokens_used"] == 42
    assert job["finished_ts"] == 3.0


def test_fail_job_records_error(repo):
    repo.create_job(
        "job2", database="default", collection="ingest", filename=None, mime=None
    )
    repo.fail_job("job2", error_code="embedder_unavailable", error_message="CUDA OOM")
    job = repo.get_job("job2")
    assert job["status"] == JOB_FAILED
    assert job["error_code"] == "embedder_unavailable"
    assert job["error_message"] == "CUDA OOM"
    assert job["finished_ts"]


def test_get_missing_job_is_none(repo):
    assert repo.get_job("nope") is None


def _create_async_job(
    repo,
    job_id,
    *,
    now=1.0,
    spool="/tmp/upload",
    status=JOB_QUEUED,
    max_attempts=3,
):
    repo.create_job(
        job_id,
        doc_id=f"doc-{job_id}",
        database="default",
        collection="ingest",
        filename="f.txt",
        mime="text/plain",
        params_json='{"strategy": "recursive"}',
        spool_path=spool,
        max_attempts=max_attempts,
        now=now,
    )
    if status != JOB_QUEUED:
        repo.mark_job(job_id, status, now=now)


def test_create_async_job_persists_submission_fields(repo):
    _create_async_job(repo, "j1")
    job = repo.get_job("j1")
    assert job["doc_id"] == "doc-j1"
    assert job["params_json"] == '{"strategy": "recursive"}'
    assert job["spool_path"] == "/tmp/upload"
    assert job["attempts"] == 0
    assert job["max_attempts"] == 3
    assert job["cancel_requested"] == 0
    assert job["progress_current"] is None
    assert job["progress_total"] is None


def test_set_job_doc_id_and_progress(repo):
    _create_async_job(repo, "j1")
    repo.set_job_doc_id("j1", "doc-new")
    repo.set_job_progress("j1", 3, 12)
    job = repo.get_job("j1")
    assert job["doc_id"] == "doc-new"
    assert job["progress_current"] == 3
    assert job["progress_total"] == 12


def test_request_cancel_active_vs_terminal(repo):
    _create_async_job(repo, "j1", status="parsing")
    _create_async_job(repo, "j2", status="done", now=2.0)
    assert repo.request_cancel("j1") is True
    assert repo.get_job("j1")["cancel_requested"] == 1
    assert repo.request_cancel("j2") is False
    assert repo.get_job("j2")["cancel_requested"] == 0
    assert repo.request_cancel("missing") is False


def test_claim_next_queued_orders_by_created_ts_regardless_of_spool(repo):
    # Rebuild rows carry no spool; the oldest queued row is claimed
    # whether or not one exists.
    repo.create_job(
        "rebuild",
        job_type="rebuild",
        database="default",
        collection="ingest",
        filename=None,
        mime=None,
        now=0.5,
    )
    _create_async_job(repo, "later", now=2.0)
    _create_async_job(repo, "earlier", now=1.0)

    claim = repo.claim_next_queued()
    assert claim["job_id"] == "rebuild"
    assert claim["job_type"] == "rebuild"

    # Marking it out of queued makes the next claim the earlier one.
    repo.mark_job("rebuild", "running")
    claim = repo.claim_next_queued()
    assert claim["job_id"] == "earlier"
    assert claim["job_type"] == "ingest"

    repo.mark_job("earlier", "parsing")
    assert repo.claim_next_queued()["job_id"] == "later"

    repo.mark_job("later", "done")
    assert repo.claim_next_queued() is None


def test_requeue_clears_execution_state(repo):
    _create_async_job(repo, "j1", status="failed")
    repo.set_job_progress("j1", 9, 10)
    repo.fail_job("j1", error_code="store_unavailable", error_message="down")
    repo.request_cancel("j1")

    repo.requeue_job("j1", attempts=2)
    job = repo.get_job("j1")
    assert job["status"] == JOB_QUEUED
    assert job["attempts"] == 2
    assert job["cancel_requested"] == 0
    assert job["progress_current"] is None
    assert job["progress_total"] is None
    assert job["error_code"] is None
    assert job["error_message"] is None


def test_mark_cancelled_is_terminal(repo):
    _create_async_job(repo, "j1", status="parsing")
    repo.mark_cancelled("j1")
    job = repo.get_job("j1")
    assert job["status"] == "cancelled"
    assert job["finished_ts"] is not None


def test_list_jobs_filter_and_order(repo):
    _create_async_job(repo, "j1", now=1.0)
    _create_async_job(repo, "j2", now=2.0, status="done")

    rows, total = repo.list_jobs()
    assert total == 2
    assert [r["job_id"] for r in rows] == ["j2", "j1"]  # newest first

    rows, total = repo.list_jobs(status="done")
    assert total == 1
    assert rows[0]["job_id"] == "j2"

    rows, total = repo.list_jobs(limit=1, offset=1)
    assert total == 2
    assert [r["job_id"] for r in rows] == ["j1"]


# ---- documents + chunks -----------------------------------------------


def test_store_and_get_document(repo):
    repo.store_document(_document(), _chunks())
    doc = repo.get_document("d1")
    assert doc["filename"] == "年报.pdf"
    assert doc["content_hash"] == "a" * 64
    assert doc["title"] == "年度报告"
    assert doc["author"] == "张三"
    assert doc["page_count"] == 10
    assert doc["status"] == "active"


def test_store_document_atomic_on_bad_chunk(repo):
    document = _document()
    chunks = _chunks()
    # Duplicate the first chunk id inside one batch: the executemany
    # must fail and the whole document write roll back.
    chunks.append(
        ChunkRecord(
            chunk_id="d1_0",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=2,
            text="重复主键",
        )
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.store_document(document, chunks)
    assert repo.get_document("d1") is None
    assert repo.hydrate(["d1_0"]) == {}


def test_hydrate_assembles_content_from_join(repo):
    repo.store_document(_document(), _chunks())
    hydrated = repo.hydrate(["d1_1", "d1_0", "missing"])
    assert set(hydrated) == {"d1_0", "d1_1"}

    first = hydrated["d1_0"]
    assert first["text"] == "第一段内容"
    assert first["section_header"] == "引言"
    assert first["page_number"] == 1
    assert first["char_start"] == 0
    assert first["char_end"] == 100
    assert first["token_count"] == 5
    assert first["doc_id"] == "d1"
    assert first["chunk_index"] == 0
    # Document-level columns ride the join.
    assert first["title"] == "年度报告"
    assert first["author"] == "张三"
    assert first["page_count"] == 10
    assert first["filename"] == "年报.pdf"

    second = hydrated["d1_1"]
    assert second["summary"] == "摘要文本"


def test_hydrate_empty_input(repo):
    assert repo.hydrate([]) == {}


def test_leaf_chunk_texts_in_order_scoped_and_level_filtered(repo):
    leaves = _chunks()
    # Parent rows (section/document) share the same table but must never
    # feed BM25 stats — only ANN leaves do.
    parent = [
        ChunkRecord(
            chunk_id="d1_p0",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=len(leaves),
            text="父级章节内容",
            level="section",
            parent_id="d1_doc",
        ),
    ]
    repo.store_document(_document(), [*leaves, *parent])
    repo.store_document(
        _document("d2", collection="other"),
        _chunks("d2", collection="other"),
    )
    texts = repo.leaf_chunk_texts("default", "ingest")
    assert texts == ["第一段内容", "第二段内容"]
    assert repo.leaf_chunk_texts("default", "other") == [
        "第一段内容",
        "第二段内容",
    ]
    assert repo.leaf_chunk_texts("default", "ghost") == []


# ---- ancestor walks (small-to-large) -----------------------------------


def _hierarchy_chunks(doc_id="d1", *, database="default", collection="ingest"):
    def row(cid, index, text, level, parent):
        return ChunkRecord(
            chunk_id=f"{doc_id}_{cid}",
            doc_id=doc_id,
            database=database,
            collection=collection,
            chunk_index=index,
            text=text,
            level=level,
            parent_id=f"{doc_id}_{parent}" if parent else None,
        )

    return [
        row("doc", 0, "整篇文档", "document", None),
        row("sec", 1, "章节全文", "section", "doc"),
        row("l0", 2, "叶子一", "chunk", "sec"),
        row("l1", 3, "叶子二", "chunk", "sec"),
    ]


def test_ancestor_rows_walk_to_section_and_document(repo):
    repo.store_document(_document(), _hierarchy_chunks())

    section = repo.get_chunk_ancestor_rows(
        ["d1_l0", "d1_l1"], "section"
    )
    assert set(section) == {"d1_l0", "d1_l1"}
    assert {row["chunk_id"] for row in section.values()} == {"d1_sec"}
    assert all(row["level"] == "section" for row in section.values())
    assert section["d1_l0"]["text"] == "章节全文"
    # Document-level columns ride the join.
    assert section["d1_l0"]["filename"] == "年报.pdf"

    document = repo.get_chunk_ancestor_rows(
        ["d1_l0", "d1_l1"], "document"
    )
    assert {row["chunk_id"] for row in document.values()} == {"d1_doc"}
    assert all(row["level"] == "document" for row in document.values())


def test_ancestor_rows_at_chunk_level_is_identity(repo):
    repo.store_document(_document(), _hierarchy_chunks())
    rows = repo.get_chunk_ancestor_rows(["d1_l0", "d1_l1"], "chunk")
    assert rows["d1_l0"]["chunk_id"] == "d1_l0"
    assert rows["d1_l1"]["chunk_id"] == "d1_l1"
    assert all(row["level"] == "chunk" for row in rows.values())


def test_ancestor_rows_unknown_leaf_omitted(repo):
    repo.store_document(_document(), _hierarchy_chunks())
    rows = repo.get_chunk_ancestor_rows(["d1_l0", "ghost"], "section")
    assert set(rows) == {"d1_l0"}
    assert repo.get_chunk_ancestor_rows([], "section") == {}


def test_ancestor_rows_broken_chain_falls_back_to_deepest(repo):
    chunks = _hierarchy_chunks()
    chunks.append(
        ChunkRecord(
            chunk_id="d1_lost",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=4,
            text="失联叶子",
            level="chunk",
            parent_id="d1_missing_section",
        )
    )
    repo.store_document(_document(), chunks)
    rows = repo.get_chunk_ancestor_rows(["d1_lost"], "section")
    # The missing section link cannot be followed: the leaf row itself
    # is the deepest row that exists.
    assert rows["d1_lost"]["chunk_id"] == "d1_lost"
    assert rows["d1_lost"]["level"] == "chunk"


# ---- filename resolution ----------------------------------------------


def test_doc_ids_for_filename_like(repo):
    repo.store_document(_document("d1"), _chunks("d1"))
    repo.store_document(
        _document("d2", filename="季度报告.pdf"),
        _chunks("d2"),
    )
    assert repo.doc_ids_for_filename("default", "ingest", "年报") == ["d1"]
    assert repo.doc_ids_for_filename("default", "ingest", "报告") == ["d2"]
    # 报 is the one substring shared by 年报.pdf and 季度报告.pdf.
    assert set(repo.doc_ids_for_filename("default", "ingest", "报")) == {
        "d1",
        "d2",
    }
    assert repo.doc_ids_for_filename("default", "ingest", "zzz") == []


def test_is_corpus_collection(repo):
    assert repo.is_corpus_collection("default", "ingest") is False
    repo.store_document(_document(), _chunks())
    assert repo.is_corpus_collection("default", "ingest") is True
    assert repo.is_corpus_collection("default", "other") is False


# ---- derived-index registry -------------------------------------------


def test_record_indexes_replace_not_duplicate(repo):
    repo.store_document(_document(), _chunks())
    entries = [
        IndexEntry(
            chunk_id="d1_0",
            doc_id="d1",
            database="default",
            collection="ingest",
            index_kind="dense",
            model="bge-m3",
            index_ref="ingest:vector",
        ),
    ]
    repo.record_indexes(entries)
    # Re-record with a different model (model switch re-run): replaces.
    entries[0].model = "bge-m3-v2"
    repo.record_indexes(entries)

    with repo._txn() as conn:
        rows = conn.execute(
            "SELECT model FROM chunk_indexes WHERE chunk_id = ? AND index_kind = ?",
            ("d1_0", "dense"),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "bge-m3-v2"


def test_record_indexes_empty_is_noop(repo):
    repo.record_indexes([])  # no raise


# ---- delete ------------------------------------------------------------


def test_delete_document_cascades_and_keeps_jobs(repo):
    repo.create_job(
        "job1",
        database="default",
        collection="ingest",
        filename="年报.pdf",
        mime="application/pdf",
    )
    repo.store_document(_document(), _chunks())
    repo.record_indexes(
        [
            IndexEntry(
                chunk_id="d1_0",
                doc_id="d1",
                database="default",
                collection="ingest",
                index_kind="dense",
                model="bge-m3",
                index_ref="ingest:vector",
            ),
        ]
    )

    repo.delete_document("d1")

    assert repo.get_document("d1") is None
    assert repo.hydrate(["d1_0", "d1_1"]) == {}
    with repo._txn() as conn:
        idx_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM chunk_indexes WHERE doc_id = 'd1'"
        ).fetchone()["n"]
    assert idx_rows == 0
    # Job history (including the failure story) is intentionally retained.
    assert repo.get_job("job1") is not None


def test_delete_missing_document_is_noop(repo):
    repo.delete_document("ghost")  # no raise


# ---- browse ------------------------------------------------------------


def test_browse_returns_items_and_total(repo):
    repo.store_document(_document(), _chunks())
    items, total = repo.browse("default", "ingest", limit=1, offset=0)
    assert total == 2
    assert len(items) == 1
    assert items[0]["id"] == "d1_0"
    assert items[0]["fields"]["text"] == "第一段内容"
    assert items[0]["fields"]["filename"] == "年报.pdf"

    items, total = repo.browse("default", "ingest", limit=10, offset=1)
    assert total == 2
    assert [it["id"] for it in items] == ["d1_1"]


def test_browse_empty_collection(repo):
    items, total = repo.browse("default", "ghost", limit=10, offset=0)
    assert items == []
    assert total == 0


def test_browse_level_filters_to_one_hierarchy_level(repo):
    # The browse list defaults to leaves because retrieval only searches
    # them; the parent levels stay reachable through an explicit filter.
    repo.store_document(_document(), _hierarchy_chunks())

    leaves, total = repo.browse("default", "ingest", limit=10, offset=0, level="chunk")
    assert total == 2
    assert [it["id"] for it in leaves] == ["d1_l0", "d1_l1"]
    assert all(it["fields"]["level"] == "chunk" for it in leaves)

    sections, total = repo.browse(
        "default", "ingest", limit=10, offset=0, level="section"
    )
    assert total == 1
    assert [it["id"] for it in sections] == ["d1_sec"]

    documents, total = repo.browse(
        "default", "ingest", limit=10, offset=0, level="document"
    )
    assert total == 1
    assert [it["id"] for it in documents] == ["d1_doc"]

    # No filter: every content row, each still carrying its level so the
    # UI can badge leaves against parents on a mixed page.
    everything, total = repo.browse("default", "ingest", limit=10, offset=0)
    assert total == 4
    assert [it["fields"]["level"] for it in everything] == [
        "document",
        "section",
        "chunk",
        "chunk",
    ]


def test_browse_level_paging_scopes_only_that_level(repo):
    repo.store_document(_document(), _hierarchy_chunks())
    items, total = repo.browse("default", "ingest", limit=1, offset=1, level="chunk")
    assert total == 2  # the count excludes the parent rows
    assert [it["id"] for it in items] == ["d1_l1"]


def test_get_chunk_detail_walks_parent_chain_and_indexes(repo):
    chunks = _hierarchy_chunks()
    for c in chunks:
        c.context = "上下文前缀" if c.level == "chunk" else None
    repo.store_document(_document(), chunks)
    repo.record_indexes(
        [
            IndexEntry(
                chunk_id="d1_l0",
                doc_id="d1",
                database="default",
                collection="ingest",
                index_kind="sparse",
                model="bm25-jieba",
                index_ref="ingest:text",
            ),
            IndexEntry(
                chunk_id="d1_l0",
                doc_id="d1",
                database="default",
                collection="ingest",
                index_kind="dense",
                model="bge-m3",
                index_ref="ingest:vector",
            ),
        ]
    )

    detail = repo.get_chunk_detail("default", "ingest", "d1_l0")
    assert detail["chunk_id"] == "d1_l0"
    assert detail["fields"]["text"] == "叶子一"
    assert detail["fields"]["level"] == "chunk"
    assert detail["fields"]["parent_id"] == "d1_sec"
    assert detail["fields"]["context"] == "上下文前缀"
    # Ancestors are ordered parent → root: section, then document.
    assert [a["chunk_id"] for a in detail["ancestors"]] == ["d1_sec", "d1_doc"]
    assert [a["level"] for a in detail["ancestors"]] == ["section", "document"]
    assert detail["ancestors"][0]["text"] == "章节全文"
    assert detail["ancestors"][1]["text"] == "整篇文档"
    # Index registry rows come back sorted by kind with their model tag.
    assert [ix["index_kind"] for ix in detail["indexes"]] == ["dense", "sparse"]
    assert detail["indexes"][0]["model"] == "bge-m3"
    assert detail["indexes"][0]["index_ref"] == "ingest:vector"


def test_get_chunk_detail_root_has_no_ancestors(repo):
    repo.store_document(_document(), _hierarchy_chunks())
    detail = repo.get_chunk_detail("default", "ingest", "d1_doc")
    assert detail["ancestors"] == []
    assert detail["indexes"] == []
    assert detail["fields"]["parent_id"] is None


def test_get_chunk_detail_missing_or_foreign_is_none(repo):
    repo.store_document(_document(), _hierarchy_chunks())
    assert repo.get_chunk_detail("default", "ingest", "ghost") is None
    # Same chunk_id in a different logical collection is not this chunk.
    assert repo.get_chunk_detail("default", "other", "d1_l0") is None
    assert repo.get_chunk_detail("otherdb", "ingest", "d1_l0") is None


def test_get_chunk_detail_broken_chain_stops_at_last_existing_row(repo):
    chunks = _hierarchy_chunks()
    chunks.append(
        ChunkRecord(
            chunk_id="d1_lost",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=4,
            text="失联叶子",
            level="chunk",
            parent_id="d1_missing_section",
        )
    )
    repo.store_document(_document(), chunks)
    detail = repo.get_chunk_detail("default", "ingest", "d1_lost")
    assert detail["ancestors"] == []


def test_get_chunk_detail_cycle_guard_terminates(repo):
    chunks = _hierarchy_chunks()
    # Mutually-linked parents: the walk must stop revisiting rows
    # instead of looping forever.
    chunks.append(
        ChunkRecord(
            chunk_id="d1_a",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=5,
            text="环甲",
            level="section",
            parent_id="d1_b",
        )
    )
    chunks.append(
        ChunkRecord(
            chunk_id="d1_b",
            doc_id="d1",
            database="default",
            collection="ingest",
            chunk_index=6,
            text="环乙",
            level="section",
            parent_id="d1_a",
        )
    )
    repo.store_document(_document(), chunks)
    detail = repo.get_chunk_detail("default", "ingest", "d1_a")
    assert [a["chunk_id"] for a in detail["ancestors"]] == ["d1_b"]


# ---- index bindings (blue/green rebuild) ------------------------------


def _registry_rows(ref="ingest", *, dense_model="bge-m3"):
    return [
        IndexEntry(
            chunk_id="d1_0", doc_id="d1", database="default",
            collection="ingest", index_kind="dense", model=dense_model,
            index_ref=ref,
        ),
        IndexEntry(
            chunk_id="d1_0", doc_id="d1", database="default",
            collection="ingest", index_kind="sparse", model="bm25-jieba",
            index_ref=ref,
        ),
    ]


def test_set_canary_first_binding_takes_logical_active_ref(repo):
    repo.set_canary(
        "default", "ingest",
        canary_ref="ingest__rebuild_abc", canary_percent=10,
        model="bge-m3-next",
    )
    binding = repo.get_binding("default", "ingest")
    assert binding["active_ref"] == "ingest"
    assert binding["canary_ref"] == "ingest__rebuild_abc"
    assert binding["canary_percent"] == 10
    assert binding["model"] == "bge-m3-next"
    rows = repo.list_bindings("default")
    assert [r["collection"] for r in rows] == ["ingest"]

    # Re-parking another rebuild replaces only the canary slot; the
    # active ref is untouched.
    repo.set_canary(
        "default", "ingest",
        canary_ref="ingest__rebuild_def", canary_percent=25,
        model="bge-m3-other",
    )
    binding = repo.get_binding("default", "ingest")
    assert binding["active_ref"] == "ingest"
    assert binding["canary_ref"] == "ingest__rebuild_def"
    assert binding["canary_percent"] == 25
    assert len(repo.list_bindings("default")) == 1


def test_promote_index_swaps_refs_and_rewrites_registry(repo):
    repo.store_document(_document(), _chunks())
    repo.record_indexes(_registry_rows("ingest"))
    repo.set_canary(
        "default", "ingest",
        canary_ref="ingest__rebuild_abc", canary_percent=100,
        model="bge-m3-next",
    )

    promoted = repo.promote_index("default", "ingest")
    assert promoted == {
        "old_ref": "ingest",
        "new_ref": "ingest__rebuild_abc",
        "model": "bge-m3-next",
    }

    binding = repo.get_binding("default", "ingest")
    assert binding["active_ref"] == "ingest__rebuild_abc"
    assert binding["canary_ref"] is None
    assert binding["canary_percent"] == 0

    with repo._txn() as conn:
        dense = conn.execute(
            """
            SELECT index_ref, model FROM chunk_indexes
            WHERE chunk_id = 'd1_0' AND index_kind = 'dense'
            """
        ).fetchone()
        sparse = conn.execute(
            """
            SELECT index_ref, model FROM chunk_indexes
            WHERE chunk_id = 'd1_0' AND index_kind = 'sparse'
            """
        ).fetchone()
    assert tuple(dense) == ("ingest__rebuild_abc", "bge-m3-next")
    # Sparse rows move pointer but keep their analyzer model tag.
    assert tuple(sparse) == ("ingest__rebuild_abc", "bm25-jieba")


def test_promote_index_without_canary_raises_lookup_error(repo):
    with pytest.raises(LookupError):
        repo.promote_index("default", "ingest")


def test_cascade_delete_removes_bindings(repo):
    repo.store_document(_document(), _chunks())
    repo.set_canary(
        "default", "ingest",
        canary_ref="ingest__rebuild_abc", canary_percent=10, model="m2",
    )
    repo.store_document(
        _document("d2", database="otherdb", collection="ingest"),
        _chunks("d2", database="otherdb"),
    )
    repo.set_canary(
        "otherdb", "ingest",
        canary_ref="ingest__rebuild_xyz", canary_percent=0, model="m2",
    )

    repo.delete_for_collection("default", "ingest")
    assert repo.get_binding("default", "ingest") is None
    # The other database's binding survives a collection cascade.
    assert repo.get_binding("otherdb", "ingest") is not None

    repo.delete_for_database("otherdb")
    assert repo.get_binding("otherdb", "ingest") is None
    assert repo.list_bindings("otherdb") == []
