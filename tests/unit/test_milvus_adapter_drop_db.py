"""Unit tests for MilvusAdapter.drop_database() pre-clean behaviour.

Milvus rejects ``drop_database`` for a non-empty database (error 1100,
"not empty, must drop all collections"). The adapter must list every
collection in the target database and drop them first, then drop the
database. These tests stub ``_client`` and the ``milvus_db`` module so
we can verify the call sequence without standing up a real Milvus.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_service.core.errors import (
    BackendError,
    CollectionNotFound,
    DatabaseNotFound,
)
from vector_service.stores._milvus_adapter import (
    MilvusAdapter,
    _truncate_utf8_bytes,
    _varchar_byte_caps,
)


def _adapter() -> MilvusAdapter:
    a = MilvusAdapter(uri="http://example.invalid:65535")
    # Skip real gRPC connect; attach a mock client directly.
    a._client = MagicMock()
    a._orm_connected = True
    return a


def _fake_db_module(list_names, drop_calls):
    """Replace ``milvus_db`` with a MagicMock-shaped stand-in that
    accepts the kwargs our adapter uses."""
    fake = MagicMock()
    fake.list_database.return_value = list_names
    fake.drop_database.side_effect = lambda name, using: drop_calls.append((name, using))
    return fake


def test_drop_database_drops_collections_then_db(monkeypatch):
    """When the database has collections, the adapter must list them
    and drop each before calling milvus_db.drop_database."""
    a = _adapter()
    a._client.list_collections.return_value = ["c1", "c2", "c3"]

    dropped: list[str] = []
    a._client.drop_collection.side_effect = lambda name: dropped.append(name)

    db_calls: list[tuple] = []
    fake_db = _fake_db_module(["tgt", "other"], db_calls)
    monkeypatch.setattr("vector_service.stores._milvus_adapter.milvus_db", fake_db)

    a.drop_database("tgt")

    assert sorted(dropped) == ["c1", "c2", "c3"]
    assert db_calls == [("tgt", "default")]
    # The adapter must have bound itself to the target db before listing.
    assert a._client.using_database.call_args.args == ("tgt",)


def test_drop_database_skips_collection_drop_when_empty(monkeypatch):
    """If the database has no collections, don't call drop_collection."""
    a = _adapter()
    a._client.list_collections.return_value = []

    db_calls: list[tuple] = []
    fake_db = _fake_db_module(["tgt"], db_calls)
    monkeypatch.setattr("vector_service.stores._milvus_adapter.milvus_db", fake_db)

    a.drop_database("tgt")

    a._client.drop_collection.assert_not_called()
    assert db_calls == [("tgt", "default")]


def test_drop_unknown_database_raises_not_found(monkeypatch):
    a = _adapter()
    fake_db = _fake_db_module(["alpha", "beta"], [])
    monkeypatch.setattr("vector_service.stores._milvus_adapter.milvus_db", fake_db)

    with pytest.raises(DatabaseNotFound):
        a.drop_database("ghost")
    # Nothing should be deleted.
    a._client.drop_collection.assert_not_called()
    fake_db.drop_database.assert_not_called()


def test_drop_database_collection_drop_failure_aborts(monkeypatch):
    """If dropping one collection fails, the whole op must surface that
    error and NOT call drop_database (db stays inspectable for retry)."""
    a = _adapter()
    a._client.list_collections.return_value = ["ok", "bad"]
    a._client.drop_collection.side_effect = lambda name: (
        None if name == "ok" else (_ for _ in ()).throw(
            BackendError(f"failed to drop {name}")
        )
    )

    db_calls: list[tuple] = []
    fake_db = _fake_db_module(["tgt"], db_calls)
    monkeypatch.setattr("vector_service.stores._milvus_adapter.milvus_db", fake_db)

    with pytest.raises(BackendError):
        a.drop_database("tgt")

    # The good collection was dropped; the bad one failed; db drop skipped.
    fake_db.drop_database.assert_not_called()


def test_drop_database_list_collections_failure_is_backend_error(monkeypatch):
    """A failure while listing collections must propagate as BackendError,
    not as a confusing 5xx from Milvus."""
    a = _adapter()
    a._client.list_collections.side_effect = RuntimeError("rpc broken")

    db_calls: list[tuple] = []
    fake_db = _fake_db_module(["tgt"], db_calls)
    monkeypatch.setattr("vector_service.stores._milvus_adapter.milvus_db", fake_db)

    with pytest.raises(BackendError):
        a.drop_database("tgt")


# ---- batch 2A regressions -----------------------------------------------


def test_has_collection_caches_result(monkeypatch):
    """has_collection must not call the underlying client on every call —
    pin batch-2A fix #6 (Milvus N+1 RPC). Two has_collection calls for
    the same (db, name) must result in a single underlying RPC."""
    a = _adapter()
    a._client.has_collection.return_value = True

    assert a.has_collection("db1", "c1") is True
    assert a.has_collection("db1", "c1") is True

    assert a._client.has_collection.call_count == 1


def test_has_collection_cache_invalidated_on_create(monkeypatch):
    """create_collection must invalidate the has_collection cache for
    that (db, name) so subsequent calls re-query the backend."""
    a = _adapter()
    a._client.has_collection.return_value = False  # pre-create: not present

    # Pre-populate the cache.
    assert a.has_collection("db1", "c1") is False

    # Now simulate create_collection succeeding. The adapter calls
    # has_collection internally; we want the cache invalidated on
    # success so the *next* external call re-queries.
    a._client.create_collection.return_value = None

    # Bypass the create_collection body for the cache-invalidation test
    # by calling the invalidation hook directly (the integration of
    # create + invalidate is exercised in the management-routes tests).
    a._invalidate_collection("db1", "c1")
    assert a.has_collection("db1", "c1") is False
    # Two underlying RPCs total: the pre-populate call + the
    # post-invalidation call. If the cache weren't invalidated,
    # we'd see only one.
    assert a._client.has_collection.call_count == 2


def test_describe_collection_caches_result():
    """describe_collection must cache the normalised schema dict and
    skip both has_collection and describe_collection on the hot path."""
    a = _adapter()
    schema = {
        "fields": [
            {"name": "id", "type": 21, "is_primary": True,
             "params": {"max_length": 64}},
            {"name": "vector", "type": 101,
             "params": {"dim": 4}},
        ]
    }
    a._client.has_collection.return_value = True
    a._client.describe_collection.return_value = schema
    a._client.list_indexes.return_value = ["vector"]
    a._client.describe_index.return_value = {
        "metric_type": "L2", "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    a._client.get_collection_stats.return_value = {"row_count": 7}

    first = a.describe_collection("db1", "c1")
    second = a.describe_collection("db1", "c1")

    assert first is second  # identity check — same cached object
    # describe_collection called once despite two describe_collection calls
    assert a._client.describe_collection.call_count == 1
    # has_collection is called once (cache miss); the second describe
    # hits the schema cache before reaching has_collection.
    assert a._client.has_collection.call_count == 1
    # New: indexes are surfaced alongside fields.
    assert "indexes" in first
    assert first["indexes"], "describe_collection must enumerate indexes"
    entry = first["indexes"][0]
    assert entry["field_name"] == "vector"
    assert entry["index_type"] == "HNSW"
    assert entry["metric_type"] == "L2"
    assert entry["params"]["M"] == 16
    # L2 → "l2" on the normalised metric
    assert first["metric"] == "l2"


def test_delete_does_not_refresh_load():
    """delete must NOT call ``refresh_load`` — that was pinning p99
    latency on large collections. Pin batch-2A fix #7."""
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.get_load_state.return_value = {"state": "Loaded"}
    a._client.delete.return_value = {"delete_count": 3}
    # describe_collection now goes through the schema cache. We have
    # to prime it so the test can exercise the delete path without
    # hitting the real backend. Use the adapter's private invalidation
    # hook to drop any cached entry, then let the test bypass via
    # the lower-level upsert/delete calls — actually we need a schema
    # whose "id" field is the primary_field used by delete().
    # Easiest: skip the cache by calling _schema_cache directly.
    a._schema_cache[("db1", "c1")] = (
        9e9,
        {
            "primary_field": "id",
            "vector_field": "vector",
            "dim": 4,
            "metric": "cosine",
            "count": 0,
            "fields": [{"name": "id", "dtype": "varchar", "is_primary": True}],
        },
    )

    n = a.delete("db1", "c1", "id", ["a", "b", "c"])
    assert n == 3
    a._client.refresh_load.assert_not_called()


# ---- post-delete freshness: cache invalidation + live count -------------


def test_delete_invalidates_schema_and_has_caches():
    """A successful delete must drop both the cached describe payload
    (which carries the stale ``count``) and the has_collection entry, so
    the next read re-queries the backend instead of serving pre-delete
    data for up to the schema-cache TTL."""
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.get_load_state.return_value = {"state": "Loaded"}
    a._client.delete.return_value = {"delete_count": 2}
    a._schema_cache[("db1", "c1")] = (
        9e9,
        {
            "primary_field": "id",
            "vector_field": "vector",
            "dim": 4,
            "metric": "cosine",
            "count": 99,
            "fields": [{"name": "id", "dtype": "varchar", "is_primary": True}],
        },
    )
    a._has_cache[("db1", "c1")] = (9e9, True)

    assert a.delete("db1", "c1", "id", ["a", "b"]) == 2
    assert ("db1", "c1") not in a._schema_cache
    assert ("db1", "c1") not in a._has_cache


def test_count_runs_strong_count_star_query():
    """count() is a ``count(*)`` query at Strong consistency so it
    reflects tombstones immediately, unlike get_collection_stats."""
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.query.return_value = [{"count(*)": 5}]

    assert a.count("db1", "c1") == 5

    kwargs = a._client.query.call_args.kwargs
    assert kwargs["output_fields"] == ["count(*)"]
    assert kwargs["consistency_level"] == "Strong"
    assert kwargs["filter"] == ""


def test_count_passes_filter_expr_for_filtered_total():
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.query.return_value = [{"count(*)": 0}]

    assert a.count("db1", "c1", filter_expr="category == 'x'") == 0
    assert a._client.query.call_args.kwargs["filter"] == "category == 'x'"


def test_count_empty_result_is_zero():
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.query.return_value = []
    assert a.count("db1", "c1") == 0


def test_count_missing_collection_raises_not_found():
    a = _adapter()
    a._client.has_collection.return_value = False
    with pytest.raises(CollectionNotFound):
        a.count("db1", "ghost")
    a._client.query.assert_not_called()


def test_count_query_failure_is_backend_error():
    a = _adapter()
    a._client.has_collection.return_value = True
    a._client.query.side_effect = RuntimeError("rpc broken")
    with pytest.raises(BackendError):
        a.count("db1", "c1")


def test_browse_reads_under_strong_consistency():
    """browse() must query at Strong so a delete issued by another worker
    is visible on the next page load instead of a stale page."""
    a = _adapter()
    a._client.has_collection.return_value = True
    _prime_schema(a, [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
    ])
    a._client.query.return_value = [{"id": "r1"}]

    rows = a.browse("db1", "c1", "id", limit=20, offset=0)
    assert rows == [{"id": "r1", "fields": {}}]
    assert a._client.query.call_args.kwargs["consistency_level"] == "Strong"


# ---- upsert varchar truncation ------------------------------------------

def _prime_schema(a, fields, dim=4, vector_field="vector", primary="id"):
    """Push a synthetic describe_collection result into the schema cache.

    upsert() pulls the schema through describe_collection, which would
    hit the real backend in this test environment. Priming the cache
    lets the truncation path run without a live Milvus.
    """
    a._schema_cache[("db1", "c1")] = (
        9e9,
        {
            "primary_field": primary,
            "vector_field": vector_field,
            "dim": dim,
            "metric": "cosine",
            "count": 0,
            "fields": fields,
            "indexes": [],
        },
    )


def test_upsert_truncates_overlong_varchar(monkeypatch):
    """Pin the regression: a value over the field's max_length must be
    truncated before reaching pymilvus so error 1100 never surfaces."""
    a = _adapter()
    a._client.has_collection.return_value = True
    _prime_schema(a, [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "section_header", "dtype": "varchar", "max_length": 512},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
    ])

    # Spy on the adapter's structlog logger so we can assert the truncation
    # is loud without depending on the project's logging config / caplog.
    warnings: list[str] = []
    from vector_service.stores import _milvus_adapter as adapter_mod
    monkeypatch.setattr(
        adapter_mod.log, "warning",
        lambda msg, *a, **kw: warnings.append(msg % a if a else msg),
    )

    long_header = "x" * 600  # 600 ASCII bytes > 512 cap
    a.upsert(
        "db1", "c1", "id", "vector",
        ids=["r1"], vectors=[[0.1, 0.2, 0.3, 0.4]],
        fields=[{"section_header": long_header}],
    )

    sent = a._client.upsert.call_args.kwargs["data"]
    assert sent[0]["section_header"] == "x" * 512
    assert len(sent[0]["section_header"].encode("utf-8")) == 512
    # The truncation must be loud — operators should be able to spot it.
    assert any("section_header" in w and "truncated" in w for w in warnings)


def test_upsert_leaves_under_cap_varchar_untouched():
    a = _adapter()
    a._client.has_collection.return_value = True
    _prime_schema(a, [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "section_header", "dtype": "varchar", "max_length": 512},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
    ])

    short_header = "1. Intro > 1.1 Background"  # 28 bytes
    a.upsert(
        "db1", "c1", "id", "vector",
        ids=["r1"], vectors=[[0.1, 0.2, 0.3, 0.4]],
        fields=[{"section_header": short_header}],
    )
    sent = a._client.upsert.call_args.kwargs["data"]
    assert sent[0]["section_header"] == short_header


def test_upsert_truncation_respects_utf8_boundaries():
    """Multi-byte characters must not be split. A 512-byte cap on a
    string of 3-byte Chinese characters yields 170 chars + possibly a
    partial char that must be dropped, never an invalid sequence."""
    a = _adapter()
    a._client.has_collection.return_value = True
    _prime_schema(a, [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "section_header", "dtype": "varchar", "max_length": 4},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
    ])

    # 4 Chinese characters = 12 bytes — over the 4-byte cap. Truncating
    # in the middle of a 3-byte sequence must yield valid UTF-8, so the
    # result is either "" or exactly 1 character (the first 3 bytes).
    a.upsert(
        "db1", "c1", "id", "vector",
        ids=["r1"], vectors=[[0.1, 0.2, 0.3, 0.4]],
        fields=[{"section_header": "汉字测试"}],
    )
    sent = a._client.upsert.call_args.kwargs["data"]
    val = sent[0]["section_header"]
    # Always valid UTF-8 (round-trips through encode/decode).
    assert val.encode("utf-8").decode("utf-8") == val
    assert len(val.encode("utf-8")) <= 4
    # We expect exactly one complete 3-byte char (the cap lands inside
    # the second char's byte sequence, which is dropped).
    assert val == "汉"


def test_upsert_passes_non_varchar_fields_through():
    """INT64 / FLOAT / JSON / nullable scalars must not be touched."""
    a = _adapter()
    a._client.has_collection.return_value = True
    _prime_schema(a, [
        {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
        {"name": "year", "dtype": "int32"},
        {"name": "vector", "dtype": "float_vector", "dim": 4},
    ])

    a.upsert(
        "db1", "c1", "id", "vector",
        ids=["r1"], vectors=[[0.1, 0.2, 0.3, 0.4]],
        fields=[{"year": 2030}],
    )
    sent = a._client.upsert.call_args.kwargs["data"]
    assert sent[0]["year"] == 2030


# ---- helper unit tests --------------------------------------------------

def test_varchar_byte_caps_filters_non_varchar_and_invalid():
    caps = _varchar_byte_caps([
        {"name": "id", "dtype": "varchar", "max_length": 64},
        {"name": "section_header", "dtype": "varchar", "max_length": 512},
        {"name": "year", "dtype": "int32"},
        {"name": "broken", "dtype": "varchar"},  # no max_length
        {"name": "zero", "dtype": "varchar", "max_length": 0},  # invalid
    ])
    assert caps == {"id": 64, "section_header": 512}


def test_truncate_utf8_bytes_handles_ascii_and_cjk():
    assert _truncate_utf8_bytes("hello", 100) == "hello"
    assert _truncate_utf8_bytes("x" * 600, 512) == "x" * 512
    # 3-byte CJK: 4 chars = 12 bytes; cap at 4 bytes keeps one full char.
    assert _truncate_utf8_bytes("汉字测试", 4) == "汉"
    # Cap at 0 → empty string (never an empty bytes object).
    assert _truncate_utf8_bytes("anything", 0) == ""
    # Cap equal to current length: untouched.
    assert _truncate_utf8_bytes("hello", 5) == "hello"


