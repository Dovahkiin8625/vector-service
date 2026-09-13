"""Unit tests for the typed error hierarchy."""
from __future__ import annotations

from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    StoreError,
    VectorServiceError,
)


def test_database_not_found_carries_name():
    e = DatabaseNotFound("missing", name="alpha")
    assert e.name == "alpha"
    assert isinstance(e, StoreError)
    assert isinstance(e, VectorServiceError)


def test_database_already_exists_carries_name():
    e = DatabaseAlreadyExists("dup", name="alpha")
    assert e.name == "alpha"
    assert isinstance(e, StoreError)


def test_collection_not_found():
    assert isinstance(CollectionNotFound("x"), StoreError)


def test_collection_already_exists():
    assert isinstance(CollectionAlreadyExists("x"), StoreError)


def test_dimension_mismatch_dims():
    e = DimensionMismatch("bad", expected=4, got=2)
    assert e.expected == 4
    assert e.got == 2


def test_backend_error_is_store_error():
    assert isinstance(BackendError("x"), StoreError)
