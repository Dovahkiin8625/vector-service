import pytest

from vector_service.core.errors import (
    VectorServiceError,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
    CollectionNotFound,
    CollectionAlreadyExists,
    DimensionMismatch,
    BackendError,
)


def test_hierarchy():
    assert issubclass(EmbedderError, VectorServiceError)
    assert issubclass(StoreError, VectorServiceError)
    assert issubclass(CollectionNotFound, StoreError)
    assert issubclass(CollectionAlreadyExists, StoreError)
    assert issubclass(ModelNotLoaded, EmbedderError)
    assert issubclass(BackendError, StoreError)


def test_dimension_mismatch_carries_attrs():
    e = DimensionMismatch("dim wrong", expected=4, got=3)
    assert e.expected == 4
    assert e.got == 3
    assert "dim wrong" in str(e)


def test_dimension_mismatch_defaults():
    e = DimensionMismatch("x")
    assert e.expected is None
    assert e.got is None


def test_can_be_caught_as_base():
    with pytest.raises(VectorServiceError):
        raise CollectionNotFound("missing")
