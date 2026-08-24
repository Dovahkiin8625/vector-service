import os
import sys
import uuid

import pytest

pymilvus = pytest.importorskip("pymilvus")
# milvus-lite ships only Linux/macOS wheels; on Windows this entire module
# cannot run. The implementation is exercised in CI on Linux / WSL / Docker.
if sys.platform == "win32":
    pytest.skip("milvus-lite has no Windows wheels", allow_module_level=True)

from vector_service.stores.milvus_lite import MilvusLiteStore


@pytest.fixture
def store(tmp_path):
    uri = str(tmp_path / f"milvus_{uuid.uuid4().hex[:8]}.db")
    s = MilvusLiteStore(uri=uri)
    yield s
    s.close()


def test_collection_info_includes_count(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]])
    info = store.collection_info("c")
    assert info["count"] == 2


def test_get_returns_vector_and_metadata(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]], metadatas=[{"k": "v"}])
    got = store.get("c", ids=["a"])
    assert got[0]["metadata"] == {"k": "v"}


def test_filter_search(store):
    store.create_collection("c", dim=4)
    store.upsert(
        "c",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadatas=[{"src": "x"}, {"src": "y"}],
    )
    hits = store.search("c", query_vector=[1, 0, 0, 0], top_k=10, filter={"src": "x"})
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_upsert_updates_existing(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]], metadatas=[{"v": 1}])
    store.upsert("c", ids=["a"], vectors=[[0, 1, 0, 0]], metadatas=[{"v": 2}])
    got = store.get("c", ids=["a"])
    assert got[0]["vector"] == [0, 1, 0, 0]
    assert got[0]["metadata"] == {"v": 2}


def test_delete_idempotent(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]])
    store.delete("c", ids=["a", "nonexistent"])  # 不应抛


def test_backend_property_returns_native(store):
    assert store.backend is not None
    # 应当是 pymilvus 的 Connection 之类
    assert hasattr(store.backend, "list_collections") or hasattr(store.backend, "describe_collection")