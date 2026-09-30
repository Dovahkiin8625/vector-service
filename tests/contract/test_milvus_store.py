"""Contract tests for ``MilvusStore`` against a real Milvus server.

These tests are marked ``contract`` — they require a reachable Milvus.
Default pytest run skips them; opt in with::

    pytest -m contract

Override the target with ``VS_MILVUS_URI``. Each test uses a unique,
randomly-named database and tears it down on exit so the runs are
independent and side-effect-free.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vector_service.core.config import Settings
from vector_service.core.errors import DimensionMismatch, StoreError
from vector_service.stores.base import FieldSpec, IndexSpec
from vector_service.stores.milvus import MilvusStore


pytestmark = pytest.mark.contract


def _store() -> MilvusStore:
    uri = os.environ.get("VS_MILVUS_URI", "http://localhost:19530")
    s = Settings(milvus_uri=uri, milvus_timeout=10.0)
    return MilvusStore(settings=s)


@pytest.fixture
def store():
    s = _store()
    db_name = f"vs_test_{secrets.token_hex(6)}"
    s._ensure_connected()  # type: ignore[attr-defined]
    try:
        yield s, db_name
    finally:
        try:
            if db_name in s.list_databases():
                s.drop_database(db_name)
        except Exception:
            pass
        s.close()


def _id_scalar() -> FieldSpec:
    return FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64)


def _category_scalar() -> FieldSpec:
    return FieldSpec(name="category", dtype="varchar", max_length=32)


def _vec_field(dim: int = 4) -> FieldSpec:
    return FieldSpec(name="vector", dtype="float_vector", dim=dim)  # metric_type lives on IndexSpec, not FieldSpec


def _default_index(metric: str = "cosine") -> IndexSpec:
    return IndexSpec(
        field_name="vector", metric_type=metric, index_type="HNSW",
        params={"M": 16, "efConstruction": 200},
    )


# ---- database ----

def test_create_and_drop_database(store):
    s, db_name = store
    assert db_name not in s.list_databases()
    info = s.create_database(db_name)
    assert info.name == db_name
    assert db_name in s.list_databases()
    s.drop_database(db_name)
    assert db_name not in s.list_databases()


def test_create_database_twice_raises(store):
    s, db_name = store
    s.create_database(db_name)
    with pytest.raises(Exception) as ei:
        s.create_database(db_name)
    assert "already exists" in str(ei.value).lower() or "exists" in str(ei.value).lower()
    s.drop_database(db_name)


def test_drop_unknown_database_raises(store):
    s, _ = store
    with pytest.raises(Exception):
        s.drop_database("vs_does_not_exist_xyz")


def test_drop_non_empty_database_drops_collections_first(store):
    """Regression: Milvus rejects `drop_database` for a non-empty db
    (error 1100, "must drop all collections before drop database").
    The adapter must pre-clean every collection so this succeeds."""
    s, db_name = store
    s.create_database(db_name)
    s.create_collection(
        db_name, "c_a",
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar()],
        indexes=[_default_index()],
    )
    s.create_collection(
        db_name, "c_b",
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar()],
        indexes=[_default_index()],
    )
    # Sanity: the database now has both collections.
    assert set(s.list_collections(db_name)) >= {"c_a", "c_b"}

    s.drop_database(db_name)
    assert db_name not in s.list_databases()


# ---- collections ----

def test_create_collection_with_extra_scalar(store):
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    info = s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar(), _category_scalar()],
        indexes=[_default_index()],
    )
    assert info.dim == 4
    assert info.metric == "cosine"
    assert info.primary_field == "id"
    assert info.vector_field == "vector"
    assert coll in s.list_collections(db_name)
    s.drop_collection(db_name, coll)
    s.drop_database(db_name)


def test_create_collection_with_ivf_index(store):
    """Custom index type / params must round-trip."""
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    ivf = IndexSpec(field_name="vector", metric_type="l2",
                    index_type="IVF_FLAT", params={"nlist": 64})
    info = s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=8),
        scalar_fields=[_id_scalar()],
        indexes=[ivf],
    )
    assert info.metric == "l2"
    s.drop_collection(db_name, coll)
    s.drop_database(db_name)


def test_create_collection_invalid_index_raises(store):
    s, db_name = store
    s.create_database(db_name)
    with pytest.raises(StoreError):
        s.create_collection(
            db_name, "x",
            primary_field="id",
            vector_field=_vec_field(),
            scalar_fields=[_id_scalar()],
            indexes=[IndexSpec(field_name="vector", metric_type="hamming",
                               index_type="HNSW", params={})],
        )
    s.drop_database(db_name)


def test_create_collection_no_indexes_raises(store):
    s, db_name = store
    s.create_database(db_name)
    with pytest.raises(StoreError):
        s.create_collection(
            db_name, "x",
            primary_field="id",
            vector_field=_vec_field(),
            scalar_fields=[_id_scalar()],
            indexes=[],
        )
    s.drop_database(db_name)


# ---- vectors ----

def test_upsert_get_search_roundtrip(store):
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar(), _category_scalar()],
        indexes=[_default_index()],
    )

    ids = ["a", "b", "c"]
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ]
    fields = [{"category": "x"}, {"category": "y"}, {"category": "x"}]
    s.upsert(db_name, coll, "id", "vector", ids, vecs, fields)

    # Give Milvus a moment to build the index for fresh data — otherwise
    # the HNSW search can transiently return rows outside the filter window.
    import time as _t
    _t.sleep(2.0)

    # get by id with output_fields
    items = s.get(db_name, coll, "id", ids, output_fields=["category"])
    assert {it["id"] for it in items} == set(ids)
    cats = {it["id"]: it["fields"].get("category") for it in items}
    assert cats["a"] == "x" and cats["b"] == "y" and cats["c"] == "x"

    # search with filter_expr
    hits = s.search(
        db_name, coll, "vector",
        vecs[0], top_k=2,
        filter_expr="category == 'x'",
        output_fields=["category"],
    )
    assert len(hits) >= 1
    assert all(h.fields.get("category") == "x" for h in hits)
    assert hits[0].id == "a"

    # delete
    s.delete(db_name, coll, "id", [ids[0]])
    after = s.get(db_name, coll, "id", [ids[0]])
    assert after == []

    s.drop_collection(db_name, coll)
    s.drop_database(db_name)


def test_upsert_dimension_mismatch_is_caught(store):
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar()],
        indexes=[_default_index()],
    )
    with pytest.raises(DimensionMismatch):
        s.upsert(db_name, coll, "id", "vector", ["x"], [[1.0, 2.0]])  # 2-dim vs 4-dim
    s.drop_collection(db_name, coll)
    s.drop_database(db_name)


def test_search_dimension_mismatch(store):
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=4),
        scalar_fields=[_id_scalar()],
        indexes=[_default_index()],
    )
    with pytest.raises(DimensionMismatch):
        s.search(db_name, coll, "vector", [1.0, 2.0])  # 2-dim query
    s.drop_collection(db_name, coll)
    s.drop_database(db_name)


# ---- multi-vector + BM25 -----------------------------------------------

# Dim here is tiny (4) so vectors can be hand-written; production uses
# embedder.dim (BGE-M3 = 1024) for both vector fields.
_SCHEMA_DIM = 4


def _text_scalar() -> FieldSpec:
    return FieldSpec(name="text", dtype="varchar", max_length=2048)


def _sparse_scalar() -> FieldSpec:
    return FieldSpec(name="sparse", dtype="sparse_float_vector")


def _summary_scalar() -> FieldSpec:
    return FieldSpec(name="summary", dtype="varchar", max_length=2048,
                     nullable=True)


def _summary_vec_field(dim: int = _SCHEMA_DIM) -> FieldSpec:
    return FieldSpec(name="summary_vector", dtype="float_vector", dim=dim)


def _ingest_indexes() -> list[IndexSpec]:
    return [
        IndexSpec(field_name="vector", metric_type="cosine", index_type="HNSW",
                  params={"M": 16, "efConstruction": 200}),
        IndexSpec(field_name="sparse", metric_type="ip",
                  index_type="SPARSE_INVERTED_INDEX", params={}),
        IndexSpec(field_name="summary_vector", metric_type="cosine",
                  index_type="HNSW",
                  params={"M": 16, "efConstruction": 200}),
    ]


def test_multi_vector_and_bm25_roundtrip(store):
    """The multi-vector schema against real Milvus:

    - two FLOAT_VECTOR fields round-trip through create/upsert/search;
    - client-encoded sparse dicts are written as data (no server-side
      analyzer / Function) and ranked via IP;
    - regression: sparse search must NOT request the sparse field back —
      the server rejects "not allowed to retrieve raw data of field sparse".
    """
    s, db_name = store
    s.create_database(db_name)
    coll = f"c_{secrets.token_hex(4)}"
    info = s.create_collection(
        db_name, coll,
        primary_field="id",
        vector_field=_vec_field(dim=_SCHEMA_DIM),
        scalar_fields=[
            _id_scalar(), _text_scalar(), _sparse_scalar(), _summary_scalar(),
        ],
        indexes=_ingest_indexes(),
        extra_vector_fields=[_summary_vec_field()],
    )
    by_name = {f["name"]: f for f in info.fields}
    assert by_name["vector"]["dim"] == _SCHEMA_DIM
    assert by_name["summary_vector"]["dim"] == _SCHEMA_DIM
    assert {i.field_name for i in _ingest_indexes()} == {
        "vector", "sparse", "summary_vector",
    }

    ids = ["a", "b", "c"]
    vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    # Summary vectors deliberately point in different directions than the
    # chunk-text vectors so the two ANN legs are independently testable.
    summary_vecs = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    fields = [
        {"text": "向量数据库支持高维相似度检索", "summary": "向量库做相似检索"},
        {"text": "全文检索依靠分词与倒排索引", "summary": "全文检索用倒排索引"},
        {"text": "混合检索融合多路召回结果", "summary": "混合检索融合多路召回"},
    ]
    # Sparse vectors come precomputed: each row gets a distinct term so
    # the IP-ranked sparse leg is independently testable.
    sparse_rows = [{1: 1.0}, {2: 1.0}, {3: 1.0}]
    s.upsert(
        db_name, coll, "id", "vector", ids, vecs, fields,
        extra_vectors={"summary_vector": summary_vecs},
        sparse_vectors={"sparse": sparse_rows},
    )

    import time as _t
    _t.sleep(2.0)

    projection = ["text", "summary"]

    # Dense ANN over the chunk-text vector.
    hits = s.search(db_name, coll, "vector", vecs[0], top_k=2,
                    output_fields=projection)
    assert hits and hits[0].id == "a"
    assert hits[0].fields.get("summary") == "向量库做相似检索"
    assert "sparse" not in hits[0].fields

    # Dense ANN over the summary vector — closest row differs from the
    # chunk-text leg by construction.
    sum_hits = s.search(db_name, coll, "summary_vector", summary_vecs[0],
                        top_k=2, output_fields=projection)
    assert sum_hits and sum_hits[0].id == "a"
    assert sum_hits[0].fields.get("text")

    # Sparse full-text leg (client-encoded query, IP index). This is the
    # regression: the adapter must never put ``sparse`` in output_fields —
    # Milvus raises code=1100 "not allowed to retrieve raw data of field
    # sparse". Term 1 is unique to row "a".
    bm25_hits = s.search_sparse(db_name, coll, "sparse", {1: 1.0},
                                top_k=3, output_fields=projection)
    assert bm25_hits, "sparse leg returned no matches"
    assert bm25_hits[0].id == "a"
    for h in bm25_hits:
        assert "sparse" not in h.fields
        assert h.fields.get("text")

    # Browse default projection must also drop the non-retrievable
    # sparse field (and both vector fields).
    browsed = s.browse(db_name, coll, "id", limit=10)
    assert {r["id"] for r in browsed} == set(ids)
    for r in browsed:
        assert "sparse" not in r["fields"]
        assert "vector" not in r["fields"]
        assert "summary_vector" not in r["fields"]

    # Query-vector dim validation is per target field.
    with pytest.raises(DimensionMismatch):
        s.search(db_name, coll, "summary_vector", [1.0, 2.0])

    s.drop_collection(db_name, coll)
    s.drop_database(db_name)
