"""Unit tests for the client-side ``SparseBM25`` encoder.

Pins the derived-state contract:

- ``fit_for_ingest`` fits stats over the whole corpus and persists them;
- document sparse vectors come back as non-empty ``{int: float}`` dicts;
- a query on an un-fitted collection rebuilds stats from the corpus;
- a persisted state file is reused by a fresh encoder (corpus untouched);
- an empty corpus cannot produce stats (RuntimeError).
"""
from __future__ import annotations

import pytest

from vector_service.corpus import (
    ChunkRecord,
    CorpusRepository,
    DocumentRecord,
)
from vector_service.corpus.bm25 import SparseBM25


@pytest.fixture
def repo(tmp_path):
    r = CorpusRepository(tmp_path / "corpus.db")
    r.initialize()
    r.store_document(
        DocumentRecord(
            doc_id="d1", database="default", collection="ingest",
            filename="年报.pdf", mime="application/pdf", content_hash="a" * 64,
        ),
        [
            ChunkRecord(
                chunk_id="d1_0", doc_id="d1", database="default",
                collection="ingest", chunk_index=0,
                text="向量数据库支持高维相似度检索",
            ),
            ChunkRecord(
                chunk_id="d1_1", doc_id="d1", database="default",
                collection="ingest", chunk_index=1,
                text="全文检索依靠中文分词与倒排索引",
            ),
            ChunkRecord(
                chunk_id="d1_2", doc_id="d1", database="default",
                collection="ingest", chunk_index=2,
                text="混合检索融合多路召回结果并重新排序",
            ),
        ],
    )
    return r


def test_fit_then_encode_documents(repo, tmp_path):
    bm25 = SparseBM25(tmp_path / "bm25")
    bm25.fit_for_ingest(repo, "default", "ingest")
    vectors = bm25.encode_documents(
        "default", "ingest",
        ["向量数据库支持高维相似度检索", "全文检索依靠分词"],
    )
    assert len(vectors) == 2
    for sv in vectors:
        assert isinstance(sv, dict) and sv
        assert all(isinstance(term, int) and not isinstance(term, bool)
                   for term in sv)
        assert all(isinstance(weight, float) for weight in sv.values())

    # Stats file is persisted derived state.
    assert list((tmp_path / "bm25").glob("*.bm25.json"))


def test_encode_query_rebuilds_when_never_fitted(repo, tmp_path):
    bm25 = SparseBM25(tmp_path / "bm25")
    sv = bm25.encode_query(
        repo, "default", "ingest", "ingest", "向量检索"
    )
    assert isinstance(sv, dict)
    assert sv  # query terms exist in the corpus


def test_persisted_state_reused_without_corpus(repo, tmp_path):
    state_dir = tmp_path / "bm25"
    SparseBM25(state_dir).fit_for_ingest(repo, "default", "ingest")

    # Fresh encoder (cache cold) loads the state file. A repo that
    # explodes proves the corpus rebuild path was not taken.
    class _BoomRepo:
        def leaf_chunk_texts(self, *a):
            raise AssertionError("should rebuild from the state file")

    bm25 = SparseBM25(state_dir)
    sv = bm25.encode_query(
        _BoomRepo(), "default", "ingest", "ingest", "混合检索"
    )
    assert sv


def test_fit_for_rebuild_isolates_physical_stats(repo, tmp_path):
    state_dir = tmp_path / "bm25"
    bm25 = SparseBM25(state_dir)
    bm25.fit_for_rebuild(
        repo, "default", "ingest", "ingest__rebuild_abc"
    )
    # Stats live under the physical name; the logical name stays cold.
    assert (state_dir / "default__ingest__rebuild_abc.bm25.json").is_file()
    sv = bm25.encode_documents(
        "default", "ingest__rebuild_abc",
        ["向量数据库支持高维相似度检索"],
    )
    assert sv and sv[0]


def test_fit_for_ingest_includes_newly_stored_chunks(repo, tmp_path):
    bm25 = SparseBM25(tmp_path / "bm25")
    # Stats must be fittable straight from the corpus; every chunk's
    # vocabulary is observable afterwards.
    bm25.fit_for_ingest(repo, "default", "ingest")
    vectors = bm25.encode_documents(
        "default", "ingest", repo.leaf_chunk_texts("default", "ingest"),
    )
    assert len(vectors) == 3
    assert all(vectors)


def test_encode_query_empty_corpus_raises(repo, tmp_path):
    bm25 = SparseBM25(tmp_path / "bm25")
    with pytest.raises(RuntimeError, match="no chunks"):
        bm25.encode_query(
            repo, "default", "ghost", "ghost", "任意查询"
        )
