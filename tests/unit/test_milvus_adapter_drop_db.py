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
from vector_service.stores._milvus_adapter import MilvusAdapter


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
    a._client.describe_index.return_value = {"metric_type": "L2"}
    a._client.get_collection_stats.return_value = {"row_count": 7}

    first = a.describe_collection("db1", "c1")
    second = a.describe_collection("db1", "c1")

    assert first is second  # identity check — same cached object
    # describe_collection called once despite two describe_collection calls
    assert a._client.describe_collection.call_count == 1
    # has_collection is called once (cache miss); the second describe
    # hits the schema cache before reaching has_collection.
    assert a._client.has_collection.call_count == 1


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


