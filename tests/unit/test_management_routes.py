"""Unit tests for the management router's HTTP layer.

We don't stand up a real store; instead we patch ``request.app.state.store``
with a fake client. Tests cover route plumbing + error envelope + status codes
against the new caller-defined schema interface.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.management import router as management_router
from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DatabaseAlreadyExists,
    DatabaseNotFound,
    DimensionMismatch,
    ModelNotLoaded,
    StoreError,
)
from vector_service.core.logging import request_id_var
from vector_service.core.middleware import RequestIDMiddleware
from vector_service.stores.base import CollectionInfo, DatabaseInfo, Hit


class _Settings:
    inference_timeout_seconds = 60.0


class FakeStore:
    backend_name = "fake"

    def __init__(self):
        self.calls: list[tuple] = []
        # Live count consulted by POST .../rows — Milvus's
        # authoritative, tombstone-aware count(*). Defaults to the same
        # number collection_info reports; tests overwrite it to mimic a
        # post-delete divergence.
        self.count_rows_result: int = 3

    # database
    def list_databases(self):
        self.calls.append(("list_databases",))
        return ["alpha", "beta"]

    def create_database(self, name, **opts):
        self.calls.append(("create_database", name, opts))
        return DatabaseInfo(name=name, metadata={})

    def drop_database(self, name):
        self.calls.append(("drop_database", name))

    def database_info(self, name):
        self.calls.append(("database_info", name))
        if name == "missing":
            raise DatabaseNotFound("missing", name=name)
        return DatabaseInfo(name=name, metadata={"k": 1})

    # collections
    def list_collections(self, database):
        self.calls.append(("list_collections", database))
        if database == "missing":
            raise DatabaseNotFound("missing", name=database)
        return ["c1", "c2"]

    def create_collection(self, database, name, primary_field, vector_field, scalar_fields, indexes=None):
        self.calls.append(("create_collection", database, name,
                           primary_field, vector_field, scalar_fields, indexes))
        if name == "dup":
            raise CollectionAlreadyExists("dup")
        # Mirror MilvusStore: build a fields payload from the inputs the
        # caller just gave us, and fabricate indexes from the IndexSpec list.
        fields_payload = [
            {
                "name": f.name,
                "dtype": f.dtype,
                "is_primary": bool(f.is_primary),
                "dim": f.dim,
                "max_length": f.max_length,
                "nullable": bool(f.nullable),
                "default_value": f.default_value,
            }
            for f in scalar_fields
        ]
        fields_payload.append({
            "name": vector_field.name,
            "dtype": "float_vector",
            "is_primary": False,
            "dim": int(vector_field.dim or 0),
        })
        indexes_payload = [
            {
                "field_name": ip.field_name,
                "metric_type": ip.metric_type,
                "index_type": ip.index_type,
                "params": dict(ip.params or {}),
            }
            for ip in (indexes or [])
        ]
        return CollectionInfo(
            database=database, name=name, dim=vector_field.dim,
            metric=(indexes[0].metric_type if indexes else vector_field.metric_type),
            count=0, primary_field=primary_field, vector_field=vector_field.name,
            metadata={},
            fields=fields_payload,
            indexes=indexes_payload,
        )

    def drop_collection(self, database, name):
        self.calls.append(("drop_collection", database, name))
        if name == "missing":
            raise CollectionNotFound("missing")

    def collection_info(self, database, name):
        self.calls.append(("collection_info", database, name))
        return CollectionInfo(
            database=database, name=name, dim=4, metric="cosine", count=3,
            primary_field="id", vector_field="vector",
            fields=[
                {"name": "id", "dtype": "varchar", "is_primary": True,
                 "max_length": 64, "nullable": False},
                {"name": "category", "dtype": "varchar", "is_primary": False,
                 "max_length": 64, "nullable": True,
                 "default_value": "unknown"},
                {"name": "vector", "dtype": "float_vector",
                 "is_primary": False, "dim": 4, "nullable": False},
            ],
            indexes=[
                {"field_name": "vector", "metric_type": "cosine",
                 "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
            ],
        )

    # vectors
    def upsert(self, database, collection, primary_field, vector_field, ids, vectors, fields=None):
        self.calls.append(("upsert", database, collection, primary_field, vector_field, ids, vectors, fields))
        if vectors and len(vectors[0]) != 4:
            raise DimensionMismatch("bad", expected=4, got=len(vectors[0]))

    def delete(self, database, collection, primary_field, ids=None, *, filter_expr=None):
        self.calls.append((
            "delete", database, collection, primary_field, ids, filter_expr,
        ))
        if collection == "missing":
            raise CollectionNotFound("missing")
        # Mirror the production adapter's "return the count" contract.
        # ids-mode returns len(ids); filter-mode returns a small
        # canned number so tests can assert on the body shape.
        if ids is not None:
            return len(ids)
        return 2

    def get(self, database, collection, primary_field, ids, output_fields=None):
        self.calls.append(("get", database, collection, primary_field, ids, output_fields))
        return [{"id": ids[0], "vector": None, "fields": {"x": 1}}]

    def create_index(self, database, collection, *, field_name,
                     metric_type="cosine", index_type="HNSW", params=None):
        self.calls.append((
            "create_index", database, collection, field_name,
            metric_type, index_type, params or {},
        ))
        if collection == "missing":
            raise CollectionNotFound("missing")

    def drop_index(self, database, collection, *, field_name):
        self.calls.append(("drop_index", database, collection, field_name))
        if collection == "missing":
            raise CollectionNotFound("missing")

    def search(self, database, collection, vector_field, query_vector, top_k=10, filter_expr=None, output_fields=None):
        self.calls.append(("search", database, collection, vector_field, top_k, filter_expr, output_fields))
        return [Hit(id="a", score=0.9, fields={"x": 1})]

    def browse(
        self, database, collection, primary_field,
        *, limit=20, offset=0, filter_expr=None, output_fields=None,
    ):
        self.calls.append((
            "browse", database, collection, primary_field,
            limit, offset, filter_expr, output_fields,
        ))
        if collection == "missing":
            raise CollectionNotFound("missing")
        # Two-row default response. Tests that care about exact row
        # counts can reach into ``self.calls`` to verify the args.
        return [
            {"id": f"k-{offset}", "fields": {"category": "mouse", "price": 9.9}},
            {"id": f"k-{offset + 1}", "fields": {"category": "keyboard", "price": 99.0}},
        ]

    def count_rows(self, database, collection, *, filter_expr=None):
        self.calls.append(("count_rows", database, collection, filter_expr))
        if collection == "count-boom":
            raise BackendError("count rpc broken")
        return self.count_rows_result

    def close(self):
        pass


class FakeRepo:
    def __init__(self):
        self.calls: list[tuple] = []
        #: Hashes the delete methods will report as orphaned; tests set
        #: this to verify content-addressed blobs are dropped.
        self.orphan_hashes: list[str] = []

    def is_corpus_collection(self, database, collection):
        # These unit tests exercise the Milvus-backed browse path.
        return False

    def corpus_collections(self, database):
        self.calls.append(("corpus_collections", database))
        return []

    def list_bindings(self, database):
        return []

    def get_binding(self, database, collection):
        # Default: no binding — physical name == logical name.
        return None

    def delete_for_database(self, database):
        self.calls.append(("delete_for_database", database))
        return list(self.orphan_hashes)

    def delete_for_collection(self, database, collection):
        self.calls.append(("delete_for_collection", database, collection))
        return list(self.orphan_hashes)

    def delete_chunks(self, chunk_ids):
        self.calls.append(("delete_chunks", tuple(chunk_ids)))
        return list(self.orphan_hashes)


class FakeBlobStore:
    def __init__(self):
        self.deleted: list[str] = []

    def delete(self, digest):
        self.deleted.append(digest)
        return True


class FakeBM25:
    def __init__(self):
        self.discarded: list[tuple] = []

    def discard(self, database, collection):
        self.discarded.append((database, collection))


class FakeEmbedder:
    model_name = "fake-model"
    dim = 4

    def load(self):
        # No-op: tests bypass the real BGE-M3 loader.
        return None

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def _http_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail and isinstance(detail["error"], dict):
        inner = detail["error"]
        extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {
                "code": inner.get("code", "error"),
                "message": inner.get("message", str(detail)),
                "request_id": request_id_var.get(),
                "extra": extras,
            }},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "error", "message": str(detail),
                            "request_id": request_id_var.get(), "extra": {}}},
    )


def _validation_handler(request: Request, exc: RequestValidationError):
    """Mirror ``main._validation_handler`` for tests."""
    safe_errors = []
    first_msg = ""
    for e in exc.errors():
        safe = {k: v for k, v in e.items() if k != "ctx"}
        ctx = e.get("ctx")
        if isinstance(ctx, dict):
            safe["ctx"] = {k: str(v) for k, v in ctx.items()}
        safe_errors.append(safe)
        if not first_msg:
            loc = ".".join(str(p) for p in e.get("loc", []) if p != "body")
            err_type = e.get("type", "invalid")
            raw_msg = e.get("msg", "")
            first_msg = (
                f"{loc}: {raw_msg}" if loc else f"{err_type}: {raw_msg}"
            )
    return JSONResponse(
        status_code=422,
        content={"error": {
            "code": "invalid_request",
            "message": first_msg or "validation error",
            "request_id": request_id_var.get(),
            "extra": {"errors": safe_errors},
        }},
    )


@pytest.fixture
def app():
    a = FastAPI()
    a.add_middleware(RequestIDMiddleware)
    a.include_router(management_router)
    a.add_exception_handler(HTTPException, _http_error_handler)
    a.add_exception_handler(RequestValidationError, _validation_handler)
    a.state.store = FakeStore()
    a.state.corpus = FakeRepo()
    a.state.blob_store = FakeBlobStore()
    a.state.bm25 = FakeBM25()
    a.state.embedder = FakeEmbedder()
    a.state.settings = _Settings()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- databases ----

def test_list_databases(client):
    r = client.get("/v1/databases")
    assert r.status_code == 200
    assert r.json() == {"databases": ["alpha", "beta"]}


def test_create_database(client):
    r = client.post("/v1/databases", json={"name": "gamma"})
    assert r.status_code == 201
    assert r.json()["name"] == "gamma"


def test_create_database_conflict(client):
    fake = client.app.state.store
    fake.create_database = lambda name, **opts: (_ for _ in ()).throw(DatabaseAlreadyExists("dup", name=name))
    r = client.post("/v1/databases", json={"name": "gamma"})
    assert r.status_code == 409
    body = r.json()["error"]
    assert body["code"] == "database_exists"
    assert body["extra"]["name"] == "gamma"


def test_drop_database(client):
    r = client.delete("/v1/databases/alpha")
    assert r.status_code == 200
    assert r.json() == {"deleted": "alpha"}


def test_drop_database_corpus_backed_cleans_derived_state(client):
    repo = client.app.state.corpus
    bm25 = client.app.state.bm25
    repo.corpus_collections = lambda name: ["c1", "c2"]
    repo.orphan_hashes = ["h1", "h2"]
    r = client.delete("/v1/databases/alpha")
    assert r.status_code == 200
    assert ("delete_for_database", "alpha") in repo.calls
    assert set(bm25.discarded) == {("alpha", "c1"), ("alpha", "c2")}
    # Orphaned originals dropped from the content-addressed store.
    assert client.app.state.blob_store.deleted == ["h1", "h2"]


def test_database_info_not_found(client):
    r = client.get("/v1/databases/missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "database_not_found"


# ---- collections ----

def test_list_collections(client):
    r = client.get("/v1/databases/alpha/collections")
    assert r.status_code == 200
    assert r.json() == {"collections": ["c1", "c2"]}


def test_list_collections_db_not_found(client):
    r = client.get("/v1/databases/missing/collections")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "database_not_found"


def _coll_body(name="c3", dim=4, metric="cosine", primary="id"):
    return {
        "name": name,
        "primary_field": primary,
        "scalar_fields": [
            {"name": primary, "dtype": "varchar", "is_primary": True, "max_length": 64},
            {"name": "category", "dtype": "varchar", "max_length": 64},
        ],
        "vector_field": {"name": "vector", "dim": dim, "metric_type": metric},
        "index_params": [
            {"field_name": "vector", "metric_type": metric, "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
        ],
    }


def test_create_collection(client):
    r = client.post("/v1/databases/alpha/collections", json=_coll_body())
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "c3"
    assert body["dim"] == 4
    assert body["primary_field"] == "id"
    assert body["vector_field"] == "vector"
    assert any(f["name"] == "vector" and f["dim"] == 4 for f in body["fields"])
    # Indexes are now surfaced through the POST response too, so the operator
    # immediately sees which index the create call chose.
    assert body["indexes"], "create response should expose at least one index"
    assert body["indexes"][0]["field_name"] == "vector"
    assert body["indexes"][0]["index_type"] == "HNSW"
    assert body["indexes"][0]["metric_type"] == "cosine"
    assert body["indexes"][0]["params"] == {"M": 16, "efConstruction": 200}


def test_create_collection_uses_index_metric(client):
    r = client.post("/v1/databases/alpha/collections", json=_coll_body(metric="l2"))
    assert r.status_code == 201
    assert r.json()["metric"] == "l2"


def test_create_collection_conflict(client):
    r = client.post("/v1/databases/alpha/collections", json=_coll_body(name="dup"))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "collection_exists"


def test_create_collection_schema_validation_missing_primary(client):
    body = _coll_body()
    body["scalar_fields"] = [
        {"name": "id", "dtype": "varchar", "max_length": 64},
        {"name": "category", "dtype": "varchar", "max_length": 64},
    ]
    r = client.post("/v1/databases/alpha/collections", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_create_collection_schema_validation_no_index(client):
    body = _coll_body()
    body["index_params"] = [{"field_name": "vector", "metric_type": "cosine", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}]
    r = client.post("/v1/databases/alpha/collections", json=body)
    assert r.status_code == 201


def test_create_collection_schema_validation_invalid_index_target(client):
    body = _coll_body()
    body["index_params"] = [{"field_name": "category", "metric_type": "cosine", "index_type": "HNSW", "params": {}}]
    r = client.post("/v1/databases/alpha/collections", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_collection_info(client):
    r = client.get("/v1/databases/alpha/collections/c1")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "c1"
    assert body["database"] == "alpha"
    # The detail endpoint now exposes fields + indexes so the dashboard can
    # render the collection's schema without a follow-up RPC.
    assert body["fields"], "detail response should include fields"
    assert any(f["name"] == "id" and f["is_primary"] for f in body["fields"])
    assert any(f["name"] == "vector" and f["dtype"] == "float_vector" for f in body["fields"])
    assert body["indexes"], "detail response should include indexes"
    assert body["indexes"][0]["field_name"] == "vector"
    assert body["indexes"][0]["index_type"] == "HNSW"
    assert body["indexes"][0]["params"] == {"M": 16, "efConstruction": 200}
    # Field-level attributes (max_length / nullable / default_value) must
    # reach the wire so the dashboard can render the full schema picture.
    primary = next(f for f in body["fields"] if f["is_primary"])
    assert primary["max_length"] == 64
    assert primary["nullable"] is False
    secondary = next(f for f in body["fields"] if f["name"] == "category")
    assert secondary["max_length"] == 64
    assert secondary["nullable"] is True
    assert secondary["default_value"] == "unknown"


def test_collection_info_exposes_fields_and_indexes(client):
    """Focused regression: GET detail must surface fields + indexes payload.

    Without this, the dashboard collection row has nothing to expand into —
    every card would render empty.
    """
    r = client.get("/v1/databases/alpha/collections/c1")
    assert r.status_code == 200
    body = r.json()
    # Three fields: scalar primary + scalar secondary + vector field.
    assert len(body["fields"]) == 3
    primary = next(f for f in body["fields"] if f["is_primary"])
    assert primary["name"] == "id" and primary["dtype"] == "varchar"
    assert primary["max_length"] == 64
    assert primary["nullable"] is False
    vector_field = next(f for f in body["fields"] if f["dtype"] == "float_vector")
    assert vector_field["dim"] == 4
    assert vector_field["nullable"] is False
    # Single index on the vector field.
    assert len(body["indexes"]) == 1
    ix = body["indexes"][0]
    assert ix["field_name"] == "vector"
    assert ix["index_type"] == "HNSW"
    assert ix["metric_type"] == "cosine"
    assert ix["params"]["M"] == 16
    assert ix["params"]["efConstruction"] == 200


def test_collection_info_surfaces_field_attributes(client):
    """Pin dashboard regression: the field card must show varchar length
    and nullability, not just dtype."""
    r = client.get("/v1/databases/alpha/collections/c1")
    body = r.json()
    by_name = {f["name"]: f for f in body["fields"]}
    assert by_name["id"]["max_length"] == 64
    assert by_name["id"]["nullable"] is False
    assert by_name["category"]["max_length"] == 64
    assert by_name["category"]["nullable"] is True
    assert by_name["category"]["default_value"] == "unknown"
    assert by_name["vector"]["dim"] == 4


def test_drop_collection_not_found(client):
    r = client.delete("/v1/databases/alpha/collections/missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


def test_drop_collection_corpus_backed_cleans_corpus_and_bm25(client):
    repo = client.app.state.corpus
    bm25 = client.app.state.bm25
    repo.is_corpus_collection = lambda db, coll: True
    repo.orphan_hashes = ["h3"]
    r = client.delete("/v1/databases/alpha/collections/c1")
    assert r.status_code == 200
    assert ("delete_for_collection", "alpha", "c1") in repo.calls
    assert bm25.discarded == [("alpha", "c1")]
    assert client.app.state.blob_store.deleted == ["h3"]


# ---- vectors / search ----

def test_upsert_vectors(client):
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0, 0, 0]],
            "fields": [{"category": "x"}],
        },
    )
    assert r.status_code == 200
    assert r.json() == {"upserted": 1}


def test_upsert_dimension_mismatch(client):
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0]],
        },
    )
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "dimension_mismatch"
    assert body["extra"]["expected"] == 4


def test_upsert_store_error_returns_422(client):
    """StoreError → invalid_request (422), not 503."""
    fake = client.app.state.store
    fake.upsert = lambda *a, **kw: (_ for _ in ()).throw(StoreError("unknown scalar field 'foo'"))
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0, 0, 0]],
            "fields": [{"foo": "bar"}],
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_search(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_vector": [1, 0, 0, 0],
            "top_k": 5,
        },
    )
    assert r.status_code == 200
    assert r.json()["hits"][0]["id"] == "a"


def test_search_with_filter_expr(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_vector": [1, 0, 0, 0],
            "top_k": 5,
            "filter_expr": "category == 'mouse'",
        },
    )
    assert r.status_code == 200


def test_search_requires_one_of(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/search",
        json={"primary_field": "id", "vector_field": "vector", "top_k": 5},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---- browse (POST .../rows) ----

def test_browse_happy_path(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "limit": 20, "offset": 0},
    )
    assert r.status_code == 200
    body = r.json()
    # FakeStore.collection_info reports count=3; the response should
    # echo total/limit/offset/returned and flag has_more correctly.
    assert body["total"] == 3
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert body["returned"] == len(body["items"]) == 2
    # has_more == offset + returned < total  →  0 + 2 < 3
    assert body["has_more"] is True
    # Each item is shaped like a GetVectorItem: id, vector (None), fields.
    for it in body["items"]:
        assert "id" in it
        assert "fields" in it
    # The fake store echoes the offset into the first id, so we can
    # also assert the call was forwarded with the right paging args.
    last_call = client.app.state.store.calls[-1]
    assert last_call[0] == "browse"
    assert last_call[1:] == ("alpha", "c1", "id", 20, 0, None, None)


def test_browse_with_filter_and_paging(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={
            "primary_field": "id",
            "limit": 50,
            "offset": 20,
            "filter_expr": "category == 'mouse'",
            "output_fields": ["id", "category"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 50
    assert body["offset"] == 20
    # With offset=20 + returned=2 < total=3 the flag is still True;
    # the contract is purely arithmetic, not backend-driven.
    assert body["has_more"] is False
    last_call = client.app.state.store.calls[-1]
    assert last_call[0] == "browse"
    assert last_call[5] == 20          # offset
    assert last_call[6] == "category == 'mouse'"  # filter_expr
    assert last_call[7] == ["id", "category"]      # output_fields


def test_browse_last_page_no_more(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "limit": 20, "offset": 0},
    )
    body = r.json()
    # Pinned: total=3, returned=2, offset=0 → has_more=True.
    assert body["has_more"] is True


def test_browse_collection_not_found(client):
    r = client.post(
        "/v1/databases/alpha/collections/missing/rows",
        json={"primary_field": "id"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


def test_browse_limit_above_cap_rejected(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "limit": 500},
    )
    # Pydantic ge/le enforcement → 422 with the standard envelope.
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_browse_limit_below_one_rejected(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "limit": 0},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_browse_negative_offset_rejected(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "offset": -1},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_browse_backend_error_returns_503(client):
    """BackendError → store_unavailable (503), not 422."""
    fake = client.app.state.store
    fake.browse = lambda *a, **kw: (_ for _ in ()).throw(BackendError("milvus down"))
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id"},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "store_unavailable"


def test_browse_store_error_returns_422(client):
    """A StoreError from the adapter (e.g. unknown output_fields) maps
    to invalid_request, same envelope shape as upsert/search."""
    fake = client.app.state.store
    fake.browse = lambda *a, **kw: (
        (_ for _ in ()).throw(StoreError("unknown output_fields ['foo']"))
    )
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "output_fields": ["foo"]},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_browse_filter_expr_too_long_rejected(client):
    # Cap is 4096 chars; a 600-char token × 10 = 6000 chars busts it.
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "filter_expr": "abcdefghij" * 600},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---- browse total: live count_rows ----

def test_browse_total_comes_from_live_count_rows(client):
    """total is the live count_rows(*) result, not the metadata count."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id"},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 3
    calls = client.app.state.store.calls
    assert any(c[0] == "count_rows" for c in calls)


def test_browse_total_prefers_live_count_rows(client):
    """Metadata says 3 but the live count(*) says 1 (rows deleted since
    the last compaction): total must be the live number so the pager
    reflects the delete immediately."""
    fake = client.app.state.store
    fake.count_rows_result = 1
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "limit": 20, "offset": 0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    # 0 + 2 returned < 1 is False — the pager stops on the live total.
    assert body["has_more"] is False
    count_calls = [c for c in fake.calls if c[0] == "count_rows"]
    assert count_calls[-1] == ("count_rows", "alpha", "c1", None)


def test_browse_live_count_receives_filter_and_drives_empty_total(client):
    """The post-delete-by-filter view: no row matches anymore. The
    filtered count(*) is 0 even though metadata still counts tombstones,
    and the same filter_expr is forwarded to count_rows."""
    fake = client.app.state.store
    fake.count_rows_result = 0
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "filter_expr": "category == 'ghost'"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["has_more"] is False
    count_calls = [c for c in fake.calls if c[0] == "count_rows"]
    assert count_calls[-1] == ("count_rows", "alpha", "c1", "category == 'ghost'")


def test_browse_live_count_backend_error_returns_503(client):
    """A failing count(*) takes the same 503 envelope as a browse failure."""
    r = client.post(
        "/v1/databases/alpha/collections/count-boom/rows",
        json={"primary_field": "id"},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "store_unavailable"


def test_browse_live_count_store_error_returns_422(client):
    """An invalid filter reaches count_rows first; its StoreError maps
    to invalid_request exactly like a browse StoreError."""
    fake = client.app.state.store

    def _raise(*a, **k):
        raise StoreError("invalid filter expression")

    fake.count_rows = _raise
    r = client.post(
        "/v1/databases/alpha/collections/c1/rows",
        json={"primary_field": "id", "filter_expr": "== bad"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---- delete (POST .../vectors/delete) — ids vs filter_expr ----

def test_delete_by_ids_happy_path(client):
    """Regression: the original ids-only delete path still works."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "ids": ["a", "b"]},
    )
    assert r.status_code == 200
    assert r.json() == {"deleted": 2}
    last_call = client.app.state.store.calls[-1]
    assert last_call[0] == "delete"
    assert last_call[4] == ["a", "b"]  # ids
    assert last_call[5] is None         # filter_expr


def test_delete_by_filter_expr_happy_path(client):
    """Bulk delete via Milvus filter expression is the new path."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id",
              "filter_expr": "category == 'mouse' and price < 100"},
    )
    assert r.status_code == 200
    # FakeStore.delete returns 2 in filter mode so we can assert
    # the count flows through to the response body.
    assert r.json() == {"deleted": 2}
    last_call = client.app.state.store.calls[-1]
    assert last_call[0] == "delete"
    assert last_call[4] is None                       # ids
    assert last_call[5] == "category == 'mouse' and price < 100"  # filter_expr


def test_delete_both_ids_and_filter_rejected(client):
    """The request body must provide exactly one of ids or filter_expr."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "ids": ["a"], "filter_expr": "x > 0"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_delete_neither_ids_nor_filter_rejected(client):
    """Empty body (neither ids nor filter_expr) is a 422."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_delete_empty_ids_list_rejected(client):
    """ids=[] is treated as "not provided" by the XOR check, not as
    a 0-element delete — must be rejected with 422."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "ids": []},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_delete_filter_expr_too_long_rejected(client):
    """The same 4096 char cap used by the browse filter applies here."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "filter_expr": "abcdefghij" * 600},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_delete_collection_not_found(client):
    r = client.post(
        "/v1/databases/alpha/collections/missing/vectors/delete",
        json={"primary_field": "id", "ids": ["a"]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


def test_delete_corpus_backed_ids_mirrors_into_corpus(client):
    # Flip the collection to corpus-backed; the deleted chunk ids must
    # reach the system of record after the store delete succeeds.
    client.app.state.corpus.is_corpus_collection = lambda db, coll: True
    client.app.state.corpus.orphan_hashes = ["h4"]
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "ids": ["a", "b"]},
    )
    assert r.status_code == 200
    assert ("delete_chunks", ("a", "b")) in client.app.state.corpus.calls
    assert client.app.state.blob_store.deleted == ["h4"]


def test_delete_corpus_backed_filter_resolves_then_mirrors(client):
    # Filter mode cannot be re-resolved after the delete: the route must
    # page browse FIRST, then delete the same chunk ids from the corpus.
    client.app.state.corpus.is_corpus_collection = lambda db, coll: True
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "filter_expr": "category == 'mouse'"},
    )
    assert r.status_code == 200
    store_calls = client.app.state.store.calls
    browse_idx = max(i for i, c in enumerate(store_calls) if c[0] == "browse")
    delete_idx = max(i for i, c in enumerate(store_calls) if c[0] == "delete")
    assert browse_idx < delete_idx
    # FakeStore browse returns ids k-0/k-1 on its single page.
    assert ("delete_chunks", ("k-0", "k-1")) in client.app.state.corpus.calls


def test_delete_non_corpus_collection_skips_corpus(client):
    # Default FakeRepo says "not corpus": no chunk deletions there.
    r = client.post(
        "/v1/databases/alpha/collections/c1/vectors/delete",
        json={"primary_field": "id", "ids": ["a"]},
    )
    assert r.status_code == 200
    assert not [
        c for c in client.app.state.corpus.calls if c[0] == "delete_chunks"
    ]


# ---- index management ----

def test_create_index_happy_path(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/index",
        json={
            "field_name": "vector",
            "metric_type": "cosine",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    assert r.status_code == 200
    assert r.json() == {"rebuilt": "vector"}
    last_call = client.app.state.store.calls[-1]
    assert last_call[0] == "create_index"
    assert last_call[1:] == (
        "alpha", "c1", "vector", "cosine", "HNSW",
        {"M": 16, "efConstruction": 200},
    )


def test_create_index_with_defaults(client):
    """Caller can omit metric_type / index_type / params — defaults apply."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/index",
        json={"field_name": "vector"},
    )
    assert r.status_code == 200
    last_call = client.app.state.store.calls[-1]
    assert last_call[3:] == ("vector", "cosine", "HNSW", {})


def test_create_index_bad_metric_rejected(client):
    r = client.post(
        "/v1/databases/alpha/collections/c1/index",
        json={"field_name": "vector", "metric_type": "jaccard"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_create_index_collection_not_found(client):
    r = client.post(
        "/v1/databases/alpha/collections/missing/index",
        json={"field_name": "vector"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


def test_drop_index_happy_path(client):
    r = client.delete(
        "/v1/databases/alpha/collections/c1/index",
        params={"field_name": "vector"},
    )
    assert r.status_code == 200
    assert r.json() == {"dropped": "vector"}
    last_call = client.app.state.store.calls[-1]
    assert last_call == ("drop_index", "alpha", "c1", "vector")


def test_drop_index_missing_field_name_rejected(client):
    """field_name is required; omitting it must 422 from FastAPI's
    query-param validation (not surface as a generic 500)."""
    r = client.delete("/v1/databases/alpha/collections/c1/index")
    assert r.status_code == 422


def test_drop_index_collection_not_found(client):
    r = client.delete(
        "/v1/databases/alpha/collections/missing/index",
        params={"field_name": "vector"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


def test_create_index_store_error_returns_422(client):
    """A StoreError from the adapter maps to invalid_request, not 500."""
    fake = client.app.state.store
    fake.create_index = lambda *a, **kw: (
        (_ for _ in ()).throw(StoreError("unknown field 'foo'"))
    )
    r = client.post(
        "/v1/databases/alpha/collections/c1/index",
        json={"field_name": "vector"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---- envelope invariant ----

def test_envelope_shape(client):
    """Every error response must use the canonical envelope."""
    r = client.get("/v1/databases/missing")
    body = r.json()
    assert "error" in body
    e = body["error"]
    assert set(e.keys()) >= {"code", "message", "request_id", "extra"}
    assert e["request_id"].startswith("req_")


# ---- error reason is propagated to callers ----

def test_database_not_found_message_includes_reason(client):
    r = client.get("/v1/databases/missing")
    assert r.status_code == 404
    e = r.json()["error"]
    assert e["code"] == "database_not_found"
    # The message must reflect the underlying reason, not a generic placeholder.
    assert "missing" in e["message"]
    assert e["extra"]["exception_type"] == "DatabaseNotFound"


def test_database_already_exists_message_includes_reason(client):
    fake = client.app.state.store
    fake.create_database = lambda name, **opts: (
        (_ for _ in ()).throw(DatabaseAlreadyExists("dup", name=name))
    )
    r = client.post("/v1/databases", json={"name": "gamma"})
    assert r.status_code == 409
    e = r.json()["error"]
    assert e["code"] == "database_exists"
    assert "dup" in e["message"]
    assert e["extra"]["name"] == "gamma"
    assert e["extra"]["exception_type"] == "DatabaseAlreadyExists"


def test_collection_already_exists_message_includes_reason(client):
    r = client.post("/v1/databases/alpha/collections", json=_coll_body(name="dup"))
    assert r.status_code == 409
    e = r.json()["error"]
    assert e["code"] == "collection_exists"
    assert "dup" in e["message"]
    assert e["extra"]["exception_type"] == "CollectionAlreadyExists"


def test_collection_not_found_message_includes_reason(client):
    r = client.delete("/v1/databases/alpha/collections/missing")
    assert r.status_code == 404
    e = r.json()["error"]
    assert e["code"] == "collection_not_found"
    assert "missing" in e["message"]
    assert e["extra"]["exception_type"] == "CollectionNotFound"


def test_dimension_mismatch_message_explains_actual_size(client):
    fake = client.app.state.store
    # Mirror the production adapter: include both numbers in the message.
    fake.upsert = lambda *a, **kw: (
        (_ for _ in ()).throw(
            DimensionMismatch(
                "vector[0] dim 2 != collection dim 4",
                expected=4, got=2,
            )
        )
    )
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0]],  # 2-dim, collection is 4-dim
        },
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "dimension_mismatch"
    # Message must include the actual numbers, not just "dimension mismatch".
    assert "4" in e["message"]
    assert "2" in e["message"]
    assert e["extra"]["expected"] == 4
    assert e["extra"]["got"] == 2
    assert e["extra"]["exception_type"] == "DimensionMismatch"


def test_store_error_message_includes_reason(client):
    """StoreError (e.g. unknown scalar field) must surface the actual reason."""
    fake = client.app.state.store
    fake.upsert = lambda *a, **kw: (
        (_ for _ in ()).throw(StoreError("unknown scalar field 'foo'"))
    )
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0, 0, 0]],
            "fields": [{"foo": "bar"}],
        },
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "invalid_request"
    # Caller must see the actual reason ("unknown scalar field 'foo'").
    assert "unknown scalar field" in e["message"]
    assert "foo" in e["message"]
    assert e["extra"]["exception_type"] == "StoreError"


def test_backend_error_message_includes_reason(client):
    fake = client.app.state.store
    # create_database catches BackendError internally and re-raises as 503.
    fake.create_database = lambda name, **opts: (
        (_ for _ in ()).throw(BackendError("milvus gRPC channel closed"))
    )
    r = client.post("/v1/databases", json={"name": "gamma"})
    assert r.status_code == 503
    e = r.json()["error"]
    assert e["code"] == "store_unavailable"
    assert "milvus gRPC channel closed" in e["message"]
    assert e["extra"]["exception_type"] == "BackendError"


def test_shape_mismatch_message_explains_lengths(client):
    """When texts are embedded server-side and the embedder returns a
    different number of vectors than ids, the route's shape_mismatch
    handler kicks in (Pydantic can't catch this — it can only check
    pre-submitted vectors)."""
    embedder = client.app.state.embedder

    def _wrong_count(texts):
        # Two ids, one vector: simulates an embedder that dropped a row.
        return [[0.1, 0.2, 0.3, 0.4]]

    embedder.embed_documents = _wrong_count  # type: ignore[method-assign]

    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a", "b"],
            "texts": ["hello", "world"],
        },
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "shape_mismatch"
    assert "ids has 2" in e["message"]
    assert "vectors has 1" in e["message"]
    assert e["extra"]["ids_len"] == 2
    assert e["extra"]["vectors_len"] == 1


def test_embedder_unavailable_message_includes_reason(client):
    embedder = client.app.state.embedder

    def _boom(texts):
        raise ModelNotLoaded("model weights missing on disk")

    embedder.embed_documents = _boom  # type: ignore[method-assign]
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "texts": ["hello"],
        },
    )
    assert r.status_code == 503
    e = r.json()["error"]
    assert e["code"] == "embedder_unavailable"
    assert "model weights missing on disk" in e["message"]
    assert e["extra"]["exception_type"] == "ModelNotLoaded"
    assert e["extra"]["text_count"] == 1


# ---- validation messages surface actual reason ----

def test_validation_message_mentions_field(client):
    """A missing required field must show up by name in the top-level
    ``message`` rather than the generic 'validation error' string."""
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            # primary_field missing on purpose
            "vector_field": "vector",
            "ids": ["a"],
            "vectors": [[1, 0, 0, 0]],
        },
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "invalid_request"
    # The first validation error must appear in the message so callers
    # can diagnose without parsing the structured ``errors`` array.
    assert "primary_field" in e["message"]
    # The structured ``errors`` array is still present for programmatic use.
    assert isinstance(e["extra"]["errors"], list)
    assert e["extra"]["errors"]
