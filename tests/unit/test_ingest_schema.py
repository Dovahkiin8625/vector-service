"""Ingest schema field/index shapes and collection bootstrap.

Covers the ingest-side unit surface that the stream route tests only
reach indirectly:

- ``_ingest_scalar_fields / _ingest_indexes`` shapes;
- ``_ensure_collection`` creation, auto-database and the
  ``CollectionAlreadyExists`` race.
"""
from __future__ import annotations

from vector_service.api import ingest as ing
from vector_service.core.errors import CollectionAlreadyExists
from vector_service.stores.base import FieldSpec

# ---- field/index shapes -----------------------------------------------


def test_scalar_fields_shapes():
    by_name = {f.name: f for f in ing._ingest_scalar_fields()}
    assert by_name["id"].is_primary is True
    assert by_name["id"].max_length == 64

    # Thin derived index: no text, no analyzer — content lives in
    # SQLite, sparse vectors are written as data.
    assert by_name["doc_id"].dtype == "varchar"
    assert by_name["chunk_index"].dtype == "int64"
    assert "text" not in by_name

    assert by_name["sparse"].dtype == "sparse_float_vector"


def test_indexes_cover_vector_summary_and_sparse():
    indexes = ing._ingest_indexes()
    by_target = {i.field_name: i for i in indexes}
    assert set(by_target) == {"vector", "summary_vector", "sparse"}
    # Sparse inverted index ranks with inner product (client-side BM25).
    sparse_idx = by_target["sparse"]
    assert sparse_idx.metric_type == "ip"
    assert sparse_idx.index_type == "SPARSE_INVERTED_INDEX"
    dense_idx = by_target["vector"]
    assert dense_idx.metric_type == "cosine"
    assert dense_idx.index_type == "HNSW"
    assert dense_idx.params == {"M": 16, "efConstruction": 200}
    # Summary embeddings get their own HNSW with the same shape.
    summary_idx = by_target["summary_vector"]
    assert summary_idx.metric_type == "cosine"
    assert summary_idx.index_type == "HNSW"
    assert summary_idx.params == {"M": 16, "efConstruction": 200}


# ---- _ensure_collection ------------------------------------------------


class _RecordingStore:
    def __init__(self, colls=None, create_raises=None):
        self.dbs = ["default"]
        self.colls = colls or {"default": []}
        self.created_dbs: list[str] = []
        self.creates: list[dict] = []
        self._create_raises = create_raises

    def list_databases(self):
        return list(self.dbs)

    def create_database(self, name, **_opts):
        self.dbs.append(name)
        self.colls.setdefault(name, [])
        self.created_dbs.append(name)

    def list_collections(self, db):
        return list(self.colls.get(db, []))

    def create_collection(self, **kwargs):
        if self._create_raises is not None:
            raise self._create_raises
        self.creates.append(kwargs)
        self.colls.setdefault(kwargs["database"], []).append(kwargs["name"])


def test_ensure_collection_creates_full_schema():
    store = _RecordingStore()
    ing._ensure_collection(store, "default", "ingest", 1024)
    assert len(store.creates) == 1
    kwargs = store.creates[0]
    assert kwargs["vector_field"] == FieldSpec(
        name="vector", dtype="float_vector", dim=1024
    )
    assert {f.name for f in kwargs["scalar_fields"]} == {
        "id", "doc_id", "chunk_index", "sparse",
    }
    assert kwargs["extra_vector_fields"] == [
        FieldSpec(name="summary_vector", dtype="float_vector", dim=1024)
    ]
    assert {i.field_name for i in kwargs["indexes"]} == {
        "vector", "summary_vector", "sparse",
    }


def test_ensure_collection_auto_creates_database():
    store = _RecordingStore(colls={})
    ing._ensure_collection(store, "kb", "ingest", 4)
    assert store.created_dbs == ["kb"]
    assert store.creates[0]["database"] == "kb"


def test_ensure_collection_already_exists_is_swallowed():
    store = _RecordingStore(create_raises=CollectionAlreadyExists("ingest"))
    # Must not raise: another worker won the creation race.
    ing._ensure_collection(store, "default", "ingest", 4)
    assert store.creates == []


def test_ensure_collection_existing_matching_dim_is_noop():
    class _Info:
        dim = 4

    store = _RecordingStore(colls={"default": ["ingest"]})
    store.collection_info = lambda db, coll: _Info()
    ing._ensure_collection(store, "default", "ingest", 4)
    assert store.creates == []
