"""Unit tests for Settings + store construction."""
from __future__ import annotations

import pytest

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import StoreError
from vector_service.stores.milvus import MilvusStore
from vector_service.stores.registry import build_store


def test_settings_milvus_uri_default():
    s = Settings()
    assert s.milvus_uri.startswith("http")


def test_settings_milvus_timeout_bounds():
    # Below the lower bound must be rejected by pydantic.
    with pytest.raises(Exception):
        Settings(milvus_timeout=0.0)


def test_build_store_returns_milvus_store():
    s = Settings(milvus_uri="http://example.invalid:65535")
    store = build_store(s)
    try:
        assert isinstance(store, MilvusStore)
        assert store.backend_name == "milvus"
        assert store.uri == "http://example.invalid:65535"
    finally:
        store.close()


def test_build_store_unknown_backend():
    s = Settings(vector_store_backend="not-a-real-backend")
    with pytest.raises(StoreError):
        build_store(s)


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
