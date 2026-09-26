"""Sparse-vector / BM25 support in the store layer.

Covers (a) MilvusStore-side schema/index validation rules, and (b) the
adapter translating our schema dicts into pymilvus schema + BM25
Function + BM25 search calls. The pymilvus client is an in-memory fake;
no live Milvus required.
"""
from __future__ import annotations

import pytest

from vector_service.core.errors import StoreError
from vector_service.stores import _milvus_adapter as ma_mod
from vector_service.stores._milvus_adapter import MilvusAdapter
from vector_service.stores.base import FieldSpec, Hit, IndexSpec
from vector_service.stores.milvus import (
    MilvusStore,
    _validate_indexes,
    _validate_schema,
)


# ---- MilvusStore validation ----

def _v1_scalars():
    return [
        FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name="text", dtype="varchar", max_length=8192),
    ]


def test_sparse_scalar_field_accepted():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    scalars = [
        *_v1_scalars(),
        FieldSpec(name="sparse", dtype="sparse_float_vector"),
    ]
    _validate_schema("id", vf, scalars)  # no raise


def test_enable_analyzer_only_on_varchar():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    scalars = [
        FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name="text", dtype="int64", enable_analyzer=True),
    ]
    with pytest.raises(Exception, match="analyzer"):
        _validate_schema("id", vf, scalars)


def test_bm25_index_allowed_on_sparse_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    _validate_indexes(vf, [
        IndexSpec(field_name="vector", metric_type="cosine"),
        IndexSpec(field_name="sparse", metric_type="bm25",
                  index_type="SPARSE_INVERTED_INDEX"),
    ])  # no raise


def test_bm25_metric_rejected_on_dense_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    with pytest.raises(Exception, match="metric_type"):
        _validate_indexes(vf, [IndexSpec(field_name="vector", metric_type="bm25")])


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
        self.functions = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append({"name": name, "dtype": dtype, **kwargs})

    def add_function(self, function):
        self.functions.append(function)


class FakeMilvusClient:
    def __init__(self):
        self.schema = None
        self.index_params = None
        self.created = None
        self.search_calls = []

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        self.index_params = FakeIndexParams()
        return self.index_params

    def create_collection(self, collection_name, schema, index_params):
        self.created = collection_name

    def describe_collection(self, collection_name):
        # Raw MilvusClient shape (type codes + params); the adapter's
        # describe_collection() normalizes it. Type 21 = varchar,
        # 104 = sparse_float_vector.
        return {
            "fields": [
                {"name": "id", "type": 21, "is_primary": True,
                 "params": {"max_length": 64}},
                {"name": "text", "type": 21,
                 "params": {"max_length": 8192}},
                {"name": "sparse", "type": 104, "params": {}},
            ]
        }

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return [[{
            "id": "c1",
            "distance": 0.5,
            "entity": {"text": "hello"},
        }]]


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
    {"name": "text", "dtype": "varchar", "max_length": 8192,
     "enable_analyzer": True, "analyzer": {"type": "chinese"}},
    {"name": "sparse", "dtype": "sparse_float_vector"},
]
_V2_INDEX_DICTS = [
    {"field_name": "vector", "metric_type": "cosine",
     "index_type": "HNSW", "params": {}},
    {"field_name": "sparse", "metric_type": "bm25",
     "index_type": "SPARSE_INVERTED_INDEX", "params": {}},
]


def test_adapter_builds_v2_schema_with_bm25_function(adapter):
    ad, fake = adapter
    ad.create_collection("default", "ingest", "id", "vector", 4, "cosine",
                         _V2_SCALAR_DICTS, _V2_INDEX_DICTS)
    by_name = {f["name"]: f for f in fake.schema.fields}
    assert "sparse" in by_name
    text_field = by_name["text"]
    assert text_field["enable_analyzer"] is True
    assert text_field["analyzer_params"] == {"type": "chinese"}
    assert len(fake.schema.functions) == 1
    fn = fake.schema.functions[0]
    assert fn.input_field_names == ["text"]
    assert fn.output_field_names == ["sparse"]
    sparse_index = fake.index_params.indexes[1]
    assert sparse_index["field_name"] == "sparse"
    assert sparse_index["metric_type"] == "BM25"
    assert sparse_index["index_type"] == "SPARSE_INVERTED_INDEX"
    assert fake.created == "ingest"


def test_adapter_search_text_sends_raw_text(adapter):
    ad, fake = adapter
    hits = ad.search_text("default", "ingest", "sparse", "季度营收", top_k=5)
    call = fake.search_calls[0]
    assert call["data"] == ["季度营收"]
    assert call["anns_field"] == "sparse"
    assert call["limit"] == 5
    assert call["search_params"] == {"metric_type": "BM25"}
    assert hits == [{"id": "c1", "score": 0.5, "fields": {"text": "hello"}}]


def test_adapter_rejects_analyzer_on_non_varchar_field(adapter):
    """Adapter-level defence-in-depth: enable_analyzer on an int64
    field must fail while building the schema (store-level validation
    normally catches this first; callers invoking the adapter directly
    must still be guarded)."""
    ad, fake = adapter
    scalars = [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "year", "dtype": "int64", "enable_analyzer": True},
    ]
    indexes = [
        {"field_name": "vector", "metric_type": "cosine",
         "index_type": "HNSW", "params": {}},
    ]
    with pytest.raises(Exception, match="analyzer"):
        ad.create_collection("default", "ingest", "id", "vector", 4, "cosine",
                             scalars, indexes)
    assert fake.created is None


def test_search_text_unknown_sparse_field_raises_store_error(adapter):
    """Preflight symmetry with dense search: an undeclared sparse_field
    must raise StoreError (mapped to 422 invalid_request), never reach
    Milvus as a backend failure (503)."""
    ad, fake = adapter
    with pytest.raises(StoreError, match="sparse_field"):
        ad.search_text("default", "ingest", "missing", "季度营收", top_k=5)
    assert fake.search_calls == []
