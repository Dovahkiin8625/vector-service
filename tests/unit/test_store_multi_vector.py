"""Multi-vector support: store validation + adapter create/describe/upsert.

The ingest schema carries a second ``FLOAT_VECTOR`` field
(``summary_vector``). These tests pin:

- ``MilvusStore`` schema/index validation rules for extra vector fields;
- the adapter threading extra fields through create_collection;
- describe_collection's first-wins canonical vector (the old
  last-float-vector-wins bug would flip the collection-level field);
- upsert extra-vector validation/merge;
- per-field search dim checks and browse's all-vector exclusion.

Client is a ``MagicMock``; connection hooks are stubbed on the instance.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vector_service.core.errors import DimensionMismatch, StoreError
from vector_service.stores import _milvus_adapter as adapter_mod
from vector_service.stores._milvus_adapter import MilvusAdapter
from vector_service.stores.base import FieldSpec
from vector_service.stores.milvus import _validate_indexes, _validate_schema


def _id_field():
    return FieldSpec(name="id", dtype="varchar", is_primary=True,
                     max_length=64)


def _vec(name="vector", dim=4):
    return FieldSpec(name=name, dtype="float_vector", dim=dim)


# ---- _validate_schema with extras --------------------------------------


def test_validate_schema_extras_happy():
    extras = [FieldSpec(name="summary_vector", dtype="float_vector", dim=4)]
    # Must not raise.
    _validate_schema("id", _vec(), [_id_field()], extras)


def test_validate_schema_extra_must_be_float_vector():
    bad = [FieldSpec(name="summary_vector", dtype="varchar", max_length=64)]
    with pytest.raises(StoreError, match="float_vector"):
        _validate_schema("id", _vec(), [_id_field()], bad)


@pytest.mark.parametrize("dim", [None, 0, -1])
def test_validate_schema_extra_requires_positive_dim(dim):
    bad = [FieldSpec(name="summary_vector", dtype="float_vector", dim=dim)]
    with pytest.raises(StoreError, match="dim"):
        _validate_schema("id", _vec(), [_id_field()], bad)


def test_validate_schema_extra_names_must_be_unique():
    bad = [
        FieldSpec(name="summary_vector", dtype="float_vector", dim=4),
        FieldSpec(name="summary_vector", dtype="float_vector", dim=4),
    ]
    with pytest.raises(StoreError, match="unique|duplicate"):
        _validate_schema("id", _vec(), [_id_field()], bad)


def test_validate_schema_extra_cannot_alias_main_vector():
    bad = [FieldSpec(name="vector", dtype="float_vector", dim=4)]
    with pytest.raises(StoreError, match="vector"):
        _validate_schema("id", _vec(), [_id_field()], bad)


def test_validate_schema_extra_cannot_alias_scalar():
    bad = [FieldSpec(name="id", dtype="float_vector", dim=4)]
    with pytest.raises(StoreError, match="id"):
        _validate_schema("id", _vec(), [_id_field()], bad)


# ---- _validate_indexes with extras -------------------------------------


def test_validate_indexes_dense_metrics_ok_on_extra():
    extras = [FieldSpec(name="summary_vector", dtype="float_vector", dim=4)]
    from vector_service.stores.base import IndexSpec

    for metric in ("cosine", "ip", "l2"):
        _validate_indexes(
            _vec(),
            [IndexSpec(field_name="summary_vector", metric_type=metric)],
            extras,
        )


def test_validate_indexes_bm25_rejected_on_extra():
    from vector_service.stores.base import IndexSpec

    extras = [FieldSpec(name="summary_vector", dtype="float_vector", dim=4)]
    with pytest.raises(StoreError, match="metric"):
        _validate_indexes(
            _vec(),
            [IndexSpec(field_name="summary_vector", metric_type="bm25")],
            extras,
        )


# ---- adapter create_collection ----------------------------------------


def _adapter() -> MilvusAdapter:
    a = MilvusAdapter(uri="http://example.invalid:65535")
    a._client = MagicMock()
    a._ensure_connected = lambda: None
    a._using_db = lambda db: None
    return a


def _scalar_dicts():
    return [
        {"name": "id", "dtype": "int64", "is_primary": True},
        {"name": "text", "dtype": "varchar", "max_length": 2048},
        {"name": "sparse", "dtype": "sparse_float_vector"},
    ]


def _index_dicts():
    return [
        {"field_name": "vector", "metric_type": "cosine",
         "index_type": "HNSW", "params": {"M": 16}},
        {"field_name": "sparse", "metric_type": "ip",
         "index_type": "SPARSE_INVERTED_INDEX", "params": {}},
        {"field_name": "summary_vector", "metric_type": "cosine",
         "index_type": "HNSW", "params": {"M": 16}},
    ]


def test_adapter_create_threads_extra_vector_fields():
    a = _adapter()
    schema = MagicMock()
    a._client.create_schema.return_value = schema

    a.create_collection(
        database="default", name="ingest", primary_field="id",
        vector_field_name="vector", vector_dim=4, vector_metric="cosine",
        scalar_fields=_scalar_dicts(), indexes=_index_dicts(),
        extra_vector_fields=[{"name": "summary_vector", "dim": 4}],
    )

    add_fields = [c.args[:2] for c in schema.add_field.call_args_list]
    assert (
        "vector", adapter_mod.DataType.FLOAT_VECTOR
    ) in add_fields
    assert (
        "summary_vector", adapter_mod.DataType.FLOAT_VECTOR
    ) in add_fields
    # Dim kwargs on each vector add_field.
    vec_kw = {
        c.args[0]: c.kwargs["dim"]
        for c in schema.add_field.call_args_list
        if len(c.args) >= 2 and c.args[1] == adapter_mod.DataType.FLOAT_VECTOR
    }
    assert vec_kw == {"vector": 4, "summary_vector": 4}

    # Thin index: no server-side analyzer / Function machinery.
    assert schema.add_function.call_count == 0


def test_adapter_create_bm25_metric_on_extra_raises():
    a = _adapter()
    a._client.create_schema.return_value = MagicMock()
    indexes = [
        {"field_name": "vector", "metric_type": "cosine",
         "index_type": "HNSW", "params": {}},
        {"field_name": "summary_vector", "metric_type": "bm25",
         "index_type": "HNSW", "params": {}},
    ]
    with pytest.raises(StoreError, match="metric"):
        a.create_collection(
            database="default", name="ingest", primary_field="id",
            vector_field_name="vector", vector_dim=4, vector_metric="cosine",
            scalar_fields=_scalar_dicts(), indexes=indexes,
            extra_vector_fields=[{"name": "summary_vector", "dim": 4}],
        )


# ---- adapter describe_collection: first-wins ---------------------------


def test_adapter_describe_first_vector_wins_canonical_field():
    a = _adapter()
    a._client.describe_collection.return_value = {
        "fields": [
            {"name": "vector", "type": adapter_mod.DataType.FLOAT_VECTOR,
             "params": {"dim": 4}},
            {"name": "summary_vector", "type": adapter_mod.DataType.FLOAT_VECTOR,
             "params": {"dim": 8}},
        ],
    }
    a._client.list_indexes.return_value = ["vector", "summary_vector"]
    a._client.describe_index.return_value = {
        "metric_type": "COSINE", "index_type": "HNSW", "params": {},
    }
    a._client.get_collection_stats.return_value = {"row_count": 3}

    result = a.describe_collection("default", "ingest")
    assert result["vector_field"] == "vector"
    assert result["dim"] == 4
    assert result["metric"] == "cosine"
    assert result["count"] == 3
    by_name = {f["name"]: f for f in result["fields"]}
    assert by_name["vector"]["dim"] == 4
    assert by_name["summary_vector"]["dim"] == 8
    assert {i["field_name"] for i in result["indexes"]} == {
        "vector", "summary_vector",
    }


# ---- adapter upsert extra vectors --------------------------------------


def _v3_schema_dict():
    return {
        "fields": [
            {"name": "id", "dtype": "int64", "is_primary": True},
            {"name": "vector", "dtype": "float_vector", "dim": 4},
            {"name": "text", "dtype": "varchar", "max_length": 2048},
            {"name": "sparse", "dtype": "sparse_float_vector"},
            {"name": "summary", "dtype": "varchar", "max_length": 2048},
            {"name": "summary_vector", "dtype": "float_vector", "dim": 4},
        ],
        "dim": 4,
    }


def _upsert_adapter():
    a = _adapter()
    a.has_collection = lambda db, c: True
    a.describe_collection = lambda db, c: _v3_schema_dict()
    return a


def test_adapter_upsert_merges_extra_vectors_and_scalars():
    a = _upsert_adapter()
    a.upsert(
        "default", "ingest", "id", "vector",
        ids=["r0", "r1"],
        vectors=[[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]],
        fields=[
            {"summary": "sum 0"},
            {"summary": "sum 1"},
        ],
        extra_vectors={
            "summary_vector": [
                [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4],
            ],
        },
    )
    rows = a._client.upsert.call_args.kwargs["data"]
    assert [r["id"] for r in rows] == ["r0", "r1"]
    assert rows[0]["vector"] == [0.1, 0.1, 0.1, 0.1]
    assert rows[0]["summary_vector"] == [0.3, 0.3, 0.3, 0.3]
    assert rows[0]["summary"] == "sum 0"
    assert rows[1]["summary_vector"] == [0.4, 0.4, 0.4, 0.4]


def test_adapter_upsert_unknown_extra_field_raises():
    a = _upsert_adapter()
    with pytest.raises(StoreError, match="ghost"):
        a.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            extra_vectors={"ghost": [[0.1] * 4]},
        )


def test_adapter_upsert_extra_row_count_mismatch_raises():
    a = _upsert_adapter()
    with pytest.raises(StoreError, match="rows"):
        a.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0", "r1"], vectors=[[0.1] * 4, [0.1] * 4], fields=None,
            extra_vectors={"summary_vector": [[0.1] * 4]},
        )


def test_adapter_upsert_extra_dim_mismatch_raises():
    a = _upsert_adapter()
    with pytest.raises(DimensionMismatch):
        a.upsert(
            "default", "ingest", "id", "vector",
            ids=["r0"], vectors=[[0.1] * 4], fields=None,
            extra_vectors={"summary_vector": [[0.1] * 3]},
        )


# ---- adapter search per-field dim --------------------------------------


def _search_adapter():
    a = _upsert_adapter()
    a._ensure_loaded = lambda c: None
    a._client.search.return_value = []
    return a


def test_adapter_search_targets_summary_vector():
    a = _search_adapter()
    a.search(
        "default", "ingest", "summary_vector", [0.1] * 4,
        top_k=5, output_fields=["summary"],
    )
    kwargs = a._client.search.call_args.kwargs
    assert kwargs["anns_field"] == "summary_vector"
    assert kwargs["limit"] == 5


def test_adapter_search_non_vector_field_raises():
    a = _search_adapter()
    with pytest.raises(StoreError, match="float_vector"):
        a.search("default", "ingest", "summary", [0.1] * 4)


def test_adapter_search_dim_mismatch_uses_target_field_dim():
    a = _search_adapter()
    with pytest.raises(DimensionMismatch):
        a.search("default", "ingest", "summary_vector", [0.1] * 3)


# ---- adapter browse excludes all vector fields -------------------------


def _browse_adapter():
    a = _upsert_adapter()
    a._ensure_loaded = lambda c: None
    a._client.query.return_value = []
    return a


def test_browse_default_projection_excludes_all_vectors():
    a = _browse_adapter()
    a.browse("default", "ingest", "id")
    output = a._client.query.call_args.kwargs["output_fields"]
    assert "vector" not in output
    assert "summary_vector" not in output
    # The BM25 sparse field is never retrievable, even by default.
    assert "sparse" not in output
    assert "summary" in output


def test_browse_explicit_vector_fields_are_dropped():
    a = _browse_adapter()
    a.browse("default", "ingest", "id",
             output_fields=["summary", "vector", "summary_vector", "sparse"])
    output = a._client.query.call_args.kwargs["output_fields"]
    assert "vector" not in output
    assert "summary_vector" not in output
    assert "sparse" not in output
    assert "summary" in output


def test_browse_include_vectors_keeps_dense_vectors_drops_sparse():
    a = _browse_adapter()
    a.browse("default", "ingest", "id", include_vectors=True)
    output = a._client.query.call_args.kwargs["output_fields"]
    assert "vector" in output
    assert "summary_vector" in output
    assert "sparse" not in output

