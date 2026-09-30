"""Sparse-vector support in the store layer (client-side BM25 data path).

The derived index stores no text and runs no analyzer: sparse vectors
are encoded client-side and written as ``{term_id: weight}`` dicts.
These tests pin:

- ``MilvusStore`` schema/index validation rules for the sparse field;
- the adapter building the sparse field + SPARSE_INVERTED_INDEX (IP)
  without any analyzer/Function machinery;
- upsert validation/merge of sparse row dicts;
- ``search_sparse`` sending the encoded query with metric IP.

The pymilvus client is an in-memory fake; no live Milvus required.
"""
from __future__ import annotations

import pytest

from vector_service.core.errors import StoreError
from vector_service.stores import _milvus_adapter as ma_mod
from vector_service.stores._milvus_adapter import MilvusAdapter
from vector_service.stores.base import FieldSpec, IndexSpec
from vector_service.stores.milvus import (
    MilvusStore,
    _validate_indexes,
    _validate_schema,
)


# ---- MilvusStore validation ----

def _v1_scalars():
    return [
        FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name="doc_id", dtype="varchar", max_length=64),
    ]


def test_sparse_scalar_field_accepted():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    scalars = [
        *_v1_scalars(),
        FieldSpec(name="sparse", dtype="sparse_float_vector"),
    ]
    _validate_schema("id", vf, scalars)  # no raise


def test_ip_index_allowed_on_sparse_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    _validate_indexes(vf, [
        IndexSpec(field_name="vector", metric_type="cosine"),
        IndexSpec(field_name="sparse", metric_type="ip",
                  index_type="SPARSE_INVERTED_INDEX"),
    ])  # no raise


def test_bm25_metric_rejected_on_sparse_field():
    # "bm25" is no longer a metric: sparse vectors are precomputed,
    # the index ranks them with plain inner product.
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    with pytest.raises(Exception, match="metric_type"):
        _validate_indexes(vf, [
            IndexSpec(field_name="vector", metric_type="cosine"),
            IndexSpec(field_name="sparse", metric_type="bm25",
                      index_type="SPARSE_INVERTED_INDEX"),
        ])


def test_dense_metric_rejected_on_sparse_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    with pytest.raises(Exception, match="metric_type"):
        _validate_indexes(vf, [
            IndexSpec(field_name="vector", metric_type="cosine"),
            IndexSpec(field_name="sparse", metric_type="cosine"),
        ])


# ---- adapter fake client ----

class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append({"name": name, "dtype": dtype, **kwargs})


class FakeMilvusClient:
    def __init__(self):
        self.schema = None
        self.index_params = None
        self.created = None
        self.search_calls = []
        self.upsert_calls = []

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        self.index_params = FakeIndexParams()
        return self.index_params

    def create_collection(self, collection_name, schema, index_params):
        self.created = collection_name

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return [[{
            "id": "c1",
            "distance": 0.5,
            "entity": {"doc_id": "d1"},
        }]]

    def upsert(self, collection, data):
        self.upsert_calls.append({"collection": collection, "data": data})


@pytest.fixture
def adapter(monkeypatch):
    ad = MilvusAdapter(uri="http://localhost:19530", db_name="default")
    fake = FakeMilvusClient()
    ad._client = fake
    monkeypatch.setattr(ad, "_ensure_connected", lambda: None)
    monkeypatch.setattr(ad, "_using_db", lambda db: None)
    monkeypatch.setattr(ad, "has_collection", lambda db, n: True)
    monkeypatch.setattr(ad, "_invalidate_collection", lambda db, n: None)
    return ad, fake


_V2_SCALAR_DICTS = [
    {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
    {"name": "doc_id", "dtype": "varchar", "max_length": 64},
    {"name": "chunk_index", "dtype": "int64"},
    {"name": "sparse", "dtype": "sparse_float_vector"},
]
_V2_INDEX_DICTS = [
    {"field_name": "vector", "metric_type": "cosine",
     "index_type": "HNSW", "params": {}},
    {"field_name": "sparse", "metric_type": "ip",
     "index_type": "SPARSE_INVERTED_INDEX", "params": {}},
]


def test_adapter_builds_thin_schema_no_analyzer_no_function(adapter):
    ad, fake = adapter
    ad.create_collection("default", "ingest", "id", "vector", 4, "cosine",
                         _V2_SCALAR_DICTS, _V2_INDEX_DICTS)
    by_name = {f["name"]: f for f in fake.schema.fields}
    assert "sparse" in by_name
    # No analyzer kwargs on any field, and schemas carry no functions.
    for field in fake.schema.fields:
        assert "enable_analyzer" not in field
        assert "analyzer_params" not in field
    assert not hasattr(fake.schema, "functions")
    sparse_index = fake.index_params.indexes[1]
    assert sparse_index["field_name"] == "sparse"
    assert sparse_index["metric_type"] == "IP"
    assert sparse_index["index_type"] == "SPARSE_INVERTED_INDEX"
    assert fake.created == "ingest"


# ---- upsert sparse validation/merge ----

_SPARSE_SCHEMA = {
    "fields": [
        {"name": "id", "dtype": "varchar", "is_primary": True,
         "max_length": 64},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
        {"name": "doc_id", "dtype": "varchar", "max_length": 64},
        {"name": "chunk_index", "dtype": "int64"},
        {"name": "sparse", "dtype": "sparse_float_vector"},
    ],
    "dim": 4,
}


def _upsert_adapter(monkeypatch):
    ad = MilvusAdapter(uri="http://localhost:19530", db_name="default")
    fake = FakeMilvusClient()
    ad._client = fake
    monkeypatch.setattr(ad, "_ensure_connected", lambda: None)
    monkeypatch.setattr(ad, "_using_db", lambda db: None)
    monkeypatch.setattr(ad, "has_collection", lambda db, n: True)
    monkeypatch.setattr(ad, "describe_collection",
                        lambda db, c: dict(_SPARSE_SCHEMA))
    return ad, fake


def test_adapter_upsert_merges_sparse_rows(monkeypatch):
    ad, fake = _upsert_adapter(monkeypatch)
    ad.upsert(
        "default", "ingest", "id", "vector",
        ids=["r0", "r1"],
        vectors=[[0.1] * 4, [0.2] * 4],
        fields=[{"doc_id": "d1", "chunk_index": 0},
                {"doc_id": "d1", "chunk_index": 1}],
        sparse_vectors={"sparse": [{1: 0.5, 3: 1.2}, {2: 0.9}]},
    )
    rows = fake.upsert_calls[0]["data"]
    assert rows[0]["sparse"] == {1: 0.5, 3: 1.2}
    assert rows[1]["sparse"] == {2: 0.9}


def test_adapter_upsert_unknown_sparse_field_raises(monkeypatch):
    ad, fake = _upsert_adapter(monkeypatch)
    with pytest.raises(StoreError, match="sparse vector field"):
        ad.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            sparse_vectors={"ghost": [{1: 0.5}]},
        )
    assert fake.upsert_calls == []


def test_adapter_upsert_sparse_row_count_mismatch_raises(monkeypatch):
    ad, fake = _upsert_adapter(monkeypatch)
    with pytest.raises(StoreError, match="rows"):
        ad.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0", "r1"], vectors=[[0.1] * 4, [0.1] * 4], fields=None,
            sparse_vectors={"sparse": [{1: 0.5}]},
        )
    assert fake.upsert_calls == []


@pytest.mark.parametrize("bad", [
    None, {}, [1, 2],
])
def test_adapter_upsert_sparse_bad_row_shape(monkeypatch, bad):
    ad, fake = _upsert_adapter(monkeypatch)
    with pytest.raises(StoreError):
        ad.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            sparse_vectors={"sparse": [bad]},
        )
    assert fake.upsert_calls == []


def test_adapter_upsert_sparse_bad_term_id_raises(monkeypatch):
    ad, fake = _upsert_adapter(monkeypatch)
    with pytest.raises(StoreError, match="term ids"):
        ad.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            sparse_vectors={"sparse": [{"1": 0.5}]},
        )


def test_adapter_upsert_sparse_bad_weight_raises(monkeypatch):
    ad, fake = _upsert_adapter(monkeypatch)
    with pytest.raises(StoreError, match="weights"):
        ad.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            sparse_vectors={"sparse": [{1: "0.5"}]},
        )


# ---- search_sparse ----


def _search_adapter(monkeypatch):
    return _upsert_adapter(monkeypatch)


def test_adapter_search_sparse_sends_encoded_dict(monkeypatch):
    ad, fake = _search_adapter(monkeypatch)
    monkeypatch.setattr(ad, "_ensure_loaded", lambda c: None)
    hits = ad.search_sparse("default", "ingest", "sparse",
                            {1: 0.5, 3: 1.2}, top_k=5)
    call = fake.search_calls[0]
    assert call["data"] == [{1: 0.5, 3: 1.2}]
    assert call["anns_field"] == "sparse"
    assert call["limit"] == 5
    assert call["search_params"] == {"metric_type": "IP"}
    assert hits == [{"id": "c1", "score": 0.5, "fields": {"doc_id": "d1"}}]


def test_adapter_search_sparse_empty_query_skips_call(monkeypatch):
    ad, fake = _search_adapter(monkeypatch)
    assert ad.search_sparse("default", "ingest", "sparse", {}, top_k=5) == []
    assert fake.search_calls == []


def test_adapter_search_sparse_unknown_field_raises(monkeypatch):
    """Preflight symmetry with dense search: an undeclared sparse field
    must raise StoreError (422 invalid_request), never reach Milvus as a
    backend failure (503)."""
    ad, fake = _search_adapter(monkeypatch)
    with pytest.raises(StoreError, match="sparse_float_vector"):
        ad.search_sparse("default", "ingest", "missing", {1: 0.5}, top_k=5)
    assert fake.search_calls == []
