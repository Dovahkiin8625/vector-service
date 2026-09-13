"""Unit tests for ``MilvusStore`` that don't need a live Milvus server.

We exercise the synchronous validation / construction paths only. The
actual gRPC handshake is deferred to the first call
(:meth:`MilvusStore._ensure_connected`), so these tests can run in any
environment, including CI without Milvus.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_service.core.config import Settings
from vector_service.core.errors import StoreError
from vector_service.stores.base import FieldSpec, IndexSpec
from vector_service.stores.milvus import (
    MilvusStore,
    _validate_indexes,
    _validate_name,
    _validate_schema,
)


def _id_field() -> FieldSpec:
    return FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64)


def _vec() -> FieldSpec:
    return FieldSpec(name="vector", dtype="float_vector", dim=4)


# ---- name validation ----

def test_validate_name_accepts_canonical():
    _validate_name("alpha", "db")
    _validate_name("tenant_a", "coll")
    _validate_name("a1_b2_c3", "db")


@pytest.mark.parametrize("name", [
    "",
    "1starts_with_digit",
    "has-dash",
    "has space",
    "x" * 65,
    "with/slash",
])
def test_validate_name_rejects_bad(name):
    with pytest.raises(StoreError):
        _validate_name(name, "thing")


# ---- schema validation ----

def test_validate_schema_minimal_ok():
    _validate_schema("id", _vec(), [_id_field()])


def test_validate_schema_extra_scalar_ok():
    fields = [_id_field(), FieldSpec(name="category", dtype="varchar", max_length=64)]
    _validate_schema("id", _vec(), fields)


def test_validate_schema_no_primary_raises():
    fields = [FieldSpec(name="category", dtype="varchar", max_length=64)]
    with pytest.raises(StoreError, match="primary"):
        _validate_schema("id", _vec(), fields)


def test_validate_schema_two_primaries_raises():
    fields = [
        _id_field(),
        FieldSpec(name="id2", dtype="varchar", is_primary=True, max_length=64),
    ]
    with pytest.raises(StoreError, match="one"):
        _validate_schema("id", _vec(), fields)


def test_validate_schema_primary_must_be_varchar():
    bad = FieldSpec(name="id", dtype="int64", is_primary=True)
    with pytest.raises(StoreError, match="varchar"):
        _validate_schema("id", _vec(), [bad])


def test_validate_schema_primary_must_have_max_length():
    bad = FieldSpec(name="id", dtype="varchar", is_primary=True)  # no max_length
    with pytest.raises(StoreError, match="max_length"):
        _validate_schema("id", _vec(), [bad])


def test_validate_schema_vector_field_name_collision():
    """Vector field name collides with an existing scalar field."""
    bad_vec = FieldSpec(name="id", dtype="float_vector", dim=4)
    with pytest.raises(StoreError, match="collides"):
        _validate_schema("id", bad_vec, [_id_field()])


def test_validate_schema_unsupported_dtype():
    bad = FieldSpec(name="foo", dtype="banana", max_length=4)
    with pytest.raises(StoreError, match="banana"):
        _validate_schema("id", _vec(), [_id_field(), bad])


def test_validate_schema_duplicate_scalar_names():
    fields = [_id_field(), FieldSpec(name="id", dtype="varchar", max_length=64)]
    with pytest.raises(StoreError, match="unique"):
        _validate_schema("id", _vec(), fields)


def test_validate_schema_varchar_requires_max_length():
    bad = FieldSpec(name="title", dtype="varchar")  # no max_length
    with pytest.raises(StoreError, match="max_length"):
        _validate_schema("id", _vec(), [_id_field(), bad])


def test_validate_schema_primary_name_mismatch():
    """primary_field kwarg must equal the is_primary field's name."""
    with pytest.raises(StoreError, match="primary_field"):
        _validate_schema("not_id", _vec(), [_id_field()])


def test_validate_schema_vector_must_be_float_vector():
    bad = FieldSpec(name="v", dtype="varchar", dim=4)
    with pytest.raises(StoreError, match="float_vector"):
        _validate_schema("id", bad, [_id_field()])


def test_validate_schema_vector_requires_dim():
    bad = FieldSpec(name="v", dtype="float_vector")  # no dim
    with pytest.raises(StoreError, match="dim"):
        _validate_schema("id", bad, [_id_field()])


# ---- index validation ----

def test_validate_indexes_requires_at_least_one():
    with pytest.raises(StoreError, match="at least one"):
        _validate_indexes(_vec(), [])


def test_validate_indexes_unknown_field_raises():
    bad = IndexSpec(field_name="other")
    with pytest.raises(StoreError, match="not the vector field"):
        _validate_indexes(_vec(), [bad])


def test_validate_indexes_unknown_metric_raises():
    bad = IndexSpec(field_name="vector", metric_type="hamming")
    with pytest.raises(StoreError, match="metric_type"):
        _validate_indexes(_vec(), [bad])


def test_validate_indexes_ok():
    ip = IndexSpec(field_name="vector", metric_type="cosine", index_type="HNSW", params={"M": 16})
    _validate_indexes(_vec(), [ip])


# ---- construction / lazy connect ----

def test_construct_does_not_connect(monkeypatch):
    """``__init__`` only builds an adapter — no client is instantiated."""
    from vector_service.stores import _milvus_adapter as adapter_mod
    from vector_service.stores.milvus import MilvusStore as MS

    called = {"n": 0}

    def _fake_client(**kwargs):
        called["n"] += 1
        # Return a stub that exposes ``close`` so MilvusStore.close() doesn't blow up.
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(adapter_mod, "MilvusClient", _fake_client)

    s = Settings(milvus_uri="http://does-not-matter.invalid:65535",
                 milvus_user="u", milvus_password="p", milvus_timeout=5.0)
    store = MS(settings=s)
    try:
        assert called["n"] == 0, "MilvusClient should not be constructed eagerly"
        assert store.backend_name == "milvus"
        assert store.uri == "http://does-not-matter.invalid:65535"
        a = store._adapter
        assert a._uri == "http://does-not-matter.invalid:65535"
        assert a._user == "u"
        assert a._password == "p"
        assert a._timeout == 5.0
    finally:
        store.close()


def test_construct_explicit_args_override_settings():
    s = Settings(milvus_uri="http://settings.invalid:19530")
    store = MilvusStore(
        uri="http://override.invalid:19530",
        user="u2",
        token="tok",
        settings=s,
    )
    try:
        assert store.uri == "http://override.invalid:19530"
        assert store._adapter._user == "u2"
        assert store._adapter._token == "tok"
    finally:
        store.close()


def test_close_is_idempotent_without_connect():
    store = MilvusStore(uri="http://noop.invalid:65535")
    store.close()
    store.close()
