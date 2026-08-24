import pytest

from vector_service.stores.base import VectorStore, Hit
from vector_service.core.errors import (
    CollectionNotFound,
    CollectionAlreadyExists,
    DimensionMismatch,
)


@pytest.fixture
def store():
    from vector_service.testing.fake_store import FakeStore
    return FakeStore()


def test_hit_is_dataclass():
    h = Hit(id="a", score=0.1, metadata={"k": "v"})
    assert h.id == "a"
    assert h.score == 0.1
    assert h.metadata == {"k": "v"}


def test_create_and_drop(store):
    store.create_collection("c1", dim=4)
    assert "c1" in store.list_collections()
    store.drop_collection("c1")
    assert "c1" not in store.list_collections()


def test_create_twice_raises(store):
    store.create_collection("c2", dim=4)
    with pytest.raises(CollectionAlreadyExists):
        store.create_collection("c2", dim=4)


def test_drop_missing_raises(store):
    with pytest.raises(CollectionNotFound):
        store.drop_collection("nope")


def test_list_collections_empty(store):
    assert store.list_collections() == []


def test_upsert_and_get(store):
    store.create_collection("c3", dim=4)
    store.upsert(
        "c3",
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadatas=[{"k": "v"}, {}],
    )
    got = store.get("c3", ids=["a", "b"])
    assert len(got) == 2
    assert got[0]["id"] == "a"
    assert got[0]["metadata"] == {"k": "v"}


def test_search_returns_top_k_ordered(store):
    store.create_collection("c4", dim=4)
    store.upsert(
        "c4",
        ids=["a", "b", "c"],
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
    )
    hits = store.search("c4", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0].id in ("a", "c")  # 余弦最近
    assert hits[1].id in ("a", "c")


def test_search_on_missing_collection_raises(store):
    with pytest.raises(CollectionNotFound):
        store.search("missing", query_vector=[0.0] * 4)


def test_upsert_wrong_dim_raises(store):
    store.create_collection("c5", dim=4)
    with pytest.raises(DimensionMismatch):
        store.upsert("c5", ids=["x"], vectors=[[1.0, 0.0, 0.0]])


def test_delete(store):
    store.create_collection("c6", dim=4)
    store.upsert("c6", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]])
    store.delete("c6", ids=["a"])
    assert {h["id"] for h in store.get("c6", ids=["a", "b"])} == {"b"}


def test_collection_info(store):
    store.create_collection("c7", dim=4, metric="cosine")
    info = store.collection_info("c7")
    assert info["dim"] == 4
    assert info["metric"] == "cosine"


def test_filter_is_accepted(store):
    store.create_collection("c8", dim=4)
    store.upsert(
        "c8",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadatas=[{"src": "doc1"}, {"src": "doc2"}],
    )
    hits = store.search("c8", query_vector=[1, 0, 0, 0], top_k=10, filter={"src": "doc1"})
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_backend_property_present(store):
    # 抽象属性应该有具体值
    assert isinstance(store.backend_name, str)