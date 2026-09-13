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

