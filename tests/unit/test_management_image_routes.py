"""Image-aware paths on the management router.

Covers the image branches of ``upsert_vectors`` and ``search``:

- happy path through ``decode_image`` -> image embedder -> store.upsert/search
- unknown image embedder id -> 404 ``model_not_found``
- new ``op="upsert_image"`` / ``op="search_image"`` metric labels for the
  per-route store counter so dashboards can split image traffic from text.

The fixtures reuse the ``FakeStore`` + ``FakeEmbedder`` pattern from
``test_management_routes.py`` so the routing layer is exercised without
spinning up a real Milvus.
"""
from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.management import router as management_router
from vector_service.core.logging import request_id_var
from vector_service.core.metrics import STORE_VECTORS_TOTAL
from vector_service.core.middleware import RequestIDMiddleware
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput
from vector_service.stores.base import Hit


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
VALID_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode("ascii")
HUGE_B64 = base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode("ascii")


class FakeStore:
    """Minimal stand-in for the production store.

    Records every call so tests can verify the route dispatched to the
    correct store method with the expected arguments. The image branches
    only use ``upsert`` and ``search``.
    """

    backend_name = "fake"

    def __init__(self):
        self.calls: list[tuple] = []

    def upsert(self, database, collection, primary_field, vector_field,
               ids, vectors, fields=None):
        self.calls.append(("upsert", database, collection, primary_field,
                            vector_field, ids, vectors, fields))

    def search(self, database, collection, vector_field, query_vector,
               top_k=10, filter_expr=None, output_fields=None):
        self.calls.append(("search", database, collection, vector_field,
                           top_k, filter_expr, output_fields, list(query_vector)))
        return [Hit(id="a", score=0.9, fields={"x": 1})]

    # The other methods exist so the router does not blow up if it
    # accidentally falls into a non-image branch.
    def list_databases(self):
        return []

    def create_database(self, name, **opts):
        pass

    def drop_database(self, name):
        pass

    def database_info(self, name):
        pass

    def list_collections(self, database):
        return []

    def create_collection(self, *args, **kwargs):
        pass

    def drop_collection(self, database, name):
        pass

    def collection_info(self, database, name):
        pass

    def delete(self, database, collection, primary_field, ids):
        pass

    def get(self, database, collection, primary_field, ids, output_fields=None):
        return []

    def close(self):
        pass


class FakeTextEmbedder:
    model_name = "fake-text"
    dim = 4

    def load(self):
        return None

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class FakeImageEmbedder(ImageEmbedder):
    """Records calls; returns deterministic vectors.

    ``dim`` matches the production OpenCLIP ViT-L/14 (768) so tests
    stay realistic; the FakeStore doesn't validate dimension, so any
    value is fine for routing tests.
    """

    dim = 4  # keep aligned with the fake store; routing does not care
    model_name = "fake-image"

    def __init__(self):
        self.calls = []

    def load(self):
        return None

    def embed_images(self, images):
        self.calls.append(("batch", [i.data for i in images]))
        return [[0.1, 0.2, 0.3, 0.4] for _ in images]

    def embed_query_image(self, image):
        self.calls.append(("query", image.data))
        return [0.5, 0.5, 0.5, 0.5]


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
    a.state.embedder = FakeTextEmbedder()
    a.state.image_embedder = FakeImageEmbedder()
    # The image branches pull decode-time limits out of
    # ``settings.image_embedding``; build a tiny stub with the two
    # fields the helper consults.
    a.state.settings = type("S", (), {
        "image_embedding": type("IE", (), {
            "max_image_bytes": 10 * 1024 * 1024,
            "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
        })(),
    })()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- upsert with images ----


def test_upsert_vectors_with_images_happy_path(client):
    """Image branch: decode -> embed_images -> store.upsert, 200, metric upsert_image >= 1."""
    before = STORE_VECTORS_TOTAL.labels(
        op="upsert_image", backend="fake", database="alpha"
    )._value.get()  # type: ignore[attr-defined]

    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a", "b"],
            "images": [VALID_PNG_B64, VALID_JPEG_B64],
            "image_mimes": ["image/png", "image/jpeg"],
            "model": "openclip-vit-l-14",
            "fields": [{"category": "x"}, {"category": "y"}],
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"upserted": 2}

    # store.upsert received the embedded vectors.
    fake_store = client.app.state.store
    upsert_calls = [c for c in fake_store.calls if c[0] == "upsert"]
    assert len(upsert_calls) == 1
    _, db, coll, primary, vec_field, ids, vectors, fields = upsert_calls[0]
    assert db == "alpha"
    assert coll == "c1"
    assert primary == "id"
    assert vec_field == "vector"
    assert ids == ["a", "b"]
    assert vectors == [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
    assert fields == [{"category": "x"}, {"category": "y"}]

    # The image embedder actually ran with the decoded bytes (NOT base64).
    img_embedder = client.app.state.image_embedder
    assert len(img_embedder.calls) == 1
    assert img_embedder.calls[0][0] == "batch"
    assert img_embedder.calls[0][1] == [
        base64.b64decode(VALID_PNG_B64),
        base64.b64decode(VALID_JPEG_B64),
    ]

    # Metric labelled with op="upsert_image" incremented by len(ids).
    after = STORE_VECTORS_TOTAL.labels(
        op="upsert_image", backend="fake", database="alpha"
    )._value.get()  # type: ignore[attr-defined]
    assert after - before == 2


def test_upsert_vectors_unknown_image_model_returns_404(client):
    """Image branch with unknown model id -> 404 model_not_found (never reaches embedder/store)."""
    r = client.put(
        "/v1/databases/alpha/collections/c1/vectors",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "ids": ["a"],
            "images": [VALID_PNG_B64],
            "image_mimes": ["image/png"],
            "model": "does-not-exist",
        },
    )
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "model_not_found"
    assert body["extra"]["model"] == "does-not-exist"

    # Embedder and store must NOT have been touched.
    assert client.app.state.image_embedder.calls == []
    assert client.app.state.store.calls == []


# ---- search with query_image ----


def test_search_with_query_image_happy_path(client):
    """Search image branch: decode -> embed_query_image -> store.search, 200, metric search_image >= 1."""
    before = STORE_VECTORS_TOTAL.labels(
        op="search_image", backend="fake", database="alpha"
    )._value.get()  # type: ignore[attr-defined]

    r = client.post(
        "/v1/databases/alpha/collections/c1/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_image": VALID_PNG_B64,
            "query_image_mime": "image/png",
            "model": "openclip-vit-l-14",
            "top_k": 3,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["hits"]) == 1
    assert body["hits"][0]["id"] == "a"

    # store.search received the embedded query vector (NOT the raw image bytes).
    fake_store = client.app.state.store
    search_calls = [c for c in fake_store.calls if c[0] == "search"]
    assert len(search_calls) == 1
    _, db, coll, vec_field, top_k, _filter, _out, qvec = search_calls[0]
    assert db == "alpha"
    assert coll == "c1"
    assert vec_field == "vector"
    assert top_k == 3
    assert qvec == [0.5, 0.5, 0.5, 0.5]

    # Image embedder actually ran with the decoded bytes.
    img_embedder = client.app.state.image_embedder
    assert len(img_embedder.calls) == 1
    assert img_embedder.calls[0][0] == "query"
    assert img_embedder.calls[0][1] == base64.b64decode(VALID_PNG_B64)

    # Metric labelled with op="search_image" incremented.
    after = STORE_VECTORS_TOTAL.labels(
        op="search_image", backend="fake", database="alpha"
    )._value.get()  # type: ignore[attr-defined]
    assert after - before == 1


def test_search_unknown_image_model_returns_404(client):
    """Search image branch with unknown model id -> 404 model_not_found."""
    r = client.post(
        "/v1/databases/alpha/collections/c1/search",
        json={
            "primary_field": "id",
            "vector_field": "vector",
            "query_image": VALID_PNG_B64,
            "query_image_mime": "image/png",
            "model": "does-not-exist",
            "top_k": 3,
        },
    )
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "model_not_found"
    assert body["extra"]["model"] == "does-not-exist"

    # Embedder and store must NOT have been touched.
    assert client.app.state.image_embedder.calls == []
    assert client.app.state.store.calls == []