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


class FakeStore:
    backend_name = "fake"

    def __init__(self):
        self.calls: list[tuple] = []

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
        return CollectionInfo(
            database=database, name=name, dim=vector_field.dim,
            metric=(indexes[0].metric_type if indexes else vector_field.metric_type),
            count=0, primary_field=primary_field, vector_field=vector_field.name,
            metadata={},
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
        )

    # vectors
    def upsert(self, database, collection, primary_field, vector_field, ids, vectors, fields=None):
        self.calls.append(("upsert", database, collection, primary_field, vector_field, ids, vectors, fields))
        if vectors and len(vectors[0]) != 4:
            raise DimensionMismatch("bad", expected=4, got=len(vectors[0]))

    def delete(self, database, collection, primary_field, ids):
        self.calls.append(("delete", database, collection, primary_field, ids))

    def get(self, database, collection, primary_field, ids, output_fields=None):
        self.calls.append(("get", database, collection, primary_field, ids, output_fields))
        return [{"id": ids[0], "vector": None, "fields": {"x": 1}}]

    def search(self, database, collection, vector_field, query_vector, top_k=10, filter_expr=None, output_fields=None):
        self.calls.append(("search", database, collection, vector_field, top_k, filter_expr, output_fields))
        return [Hit(id="a", score=0.9, fields={"x": 1})]

    def close(self):
        pass


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
    a.state.embedder = FakeEmbedder()
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
    assert r.json()["name"] == "c1"
    assert r.json()["database"] == "alpha"


def test_drop_collection_not_found(client):
    r = client.delete("/v1/databases/alpha/collections/missing")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "collection_not_found"


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
