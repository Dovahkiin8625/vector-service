"""ingest collection: schema v2 helpers and v1/v2 capability probe."""
from __future__ import annotations

from vector_service.api.ingest import (
    _ensure_collection,
    _ingest_indexes_v2,
    _ingest_scalar_fields,
    _ingest_scalar_fields_v2,
    schema_version,
)
from vector_service.stores.base import CollectionInfo, FieldSpec


class FakeStore:
    def __init__(self):
        self.dbs = ["default"]
        self.colls = {}
        self.created = None

    def list_databases(self):
        return list(self.dbs)

    def create_database(self, name, **_opts):
        self.dbs.append(name)

    def list_collections(self, database):
        return list(self.colls.get(database, {}))

    def collection_info(self, database, name):
        spec = self.colls[database][name]
        return CollectionInfo(database=database, name=name, dim=spec["dim"],
                              metric="cosine", count=0,
                              fields=spec["fields"])

    def create_collection(self, database, name, primary_field,
                          vector_field, scalar_fields, indexes=None):
        self.colls.setdefault(database, {})[name] = {
            "dim": vector_field.dim,
            "fields": [{"name": f.name} for f in scalar_fields],
        }
        self.created = {
            "scalar": scalar_fields, "indexes": indexes,
        }


def test_v2_scalar_fields_add_analyzed_text_and_sparse():
    by_name = {f.name: f for f in _ingest_scalar_fields_v2()}
    assert "sparse" in by_name
    assert by_name["sparse"].dtype == "sparse_float_vector"
    text = by_name["text"]
    assert text.enable_analyzer is True
    assert text.analyzer == {"type": "chinese"}
    # Every v1 field is still present.
    v1_names = {f.name for f in _ingest_scalar_fields()}
    assert v1_names <= set(by_name)


def test_v2_indexes_cover_dense_and_sparse():
    by_field = {ip.field_name: ip for ip in _ingest_indexes_v2()}
    assert by_field["vector"].metric_type == "cosine"
    assert by_field["sparse"].metric_type == "bm25"
    assert by_field["sparse"].index_type == "SPARSE_INVERTED_INDEX"


def test_ensure_collection_creates_v2():
    store = FakeStore()
    _ensure_collection(store, "default", "ingest", 4)
    names = {f.name for f in store.created["scalar"]}
    assert "sparse" in names
    index_fields = {ip.field_name for ip in store.created["indexes"]}
    assert {"vector", "sparse"} <= index_fields


def test_ensure_collection_existing_is_left_alone():
    store = FakeStore()
    _ensure_collection(store, "default", "ingest", 4)
    store.created = None
    _ensure_collection(store, "default", "ingest", 4)
    assert store.created is None  # no recreation


def _info(field_names):
    return CollectionInfo(
        database="default", name="ingest", dim=4, metric="cosine", count=0,
        fields=[{"name": n} for n in field_names],
    )


def test_schema_version_probe():
    assert schema_version(_info(["id", "text", "vector"])) == 1
    assert schema_version(_info(["id", "text", "sparse", "vector"])) == 2
