"""Unit tests for the embeddings router's error envelopes.

We don't load a real BGE-M3 model; the embedder is a stub whose
``embed_documents`` / ``embed_query`` either succeed or raise. Tests
verify that the HTTP error envelope surfaces the actual reason to the
caller (top-level ``message`` and ``extra``) instead of a generic
placeholder.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.embeddings import router as embeddings_router
from vector_service.api.models import router as models_router
from vector_service.core.errors import EmbedderError, ModelNotLoaded
from vector_service.core.logging import request_id_var
from vector_service.core.middleware import RequestIDMiddleware
from vector_service.embeddings.registry import EMBEDDER_REGISTRY
from vector_service.rerankers import cross_encoder  # noqa: F401  registers "bge-reranker-v2-m3"


class _FakeSettings:
    embedding_max_texts_per_request = 3
    embedding_max_chars_per_text = 16


class FakeEmbedder:
    model_name = "fake-model"
    dim = 4

    def load(self):
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
def app(monkeypatch):
    monkeypatch.setitem(EMBEDDER_REGISTRY, "fake-model", type("E", (), {}))
    a = FastAPI()
    a.add_middleware(RequestIDMiddleware)
    a.include_router(embeddings_router)
    a.include_router(models_router)
    a.add_exception_handler(HTTPException, _http_error_handler)
    a.add_exception_handler(RequestValidationError, _validation_handler)
    a.state.settings = _FakeSettings()
    a.state.embedder = FakeEmbedder()

    # Stub image embedder so the /v1/models image branch doesn't AttributeError.
    class _StubImageEmbedder:
        model_name = "openclip-vit-l-14"
        dim = 768

        def load(self):
            pass

    a.state.image_embedder = _StubImageEmbedder()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- model_not_found ----

def test_unknown_model_returns_404_with_reason(client):
    r = client.post("/v1/embeddings", json={"model": "does-not-exist", "input": "hi"})
    assert r.status_code == 404
    e = r.json()["error"]
    assert e["code"] == "model_not_found"
    # The message must echo the unknown model id and list the registered ones.
    assert "does-not-exist" in e["message"]
    assert e["extra"]["model"] == "does-not-exist"


def test_get_unknown_model_returns_404_with_reason(client):
    r = client.get("/v1/models/does-not-exist")
    assert r.status_code == 404
    e = r.json()["error"]
    assert e["code"] == "model_not_found"
    assert "does-not-exist" in e["message"]
    assert e["extra"]["model"] == "does-not-exist"


# ---- too_many_texts ----

def test_too_many_texts_returns_422_with_reason(client):
    r = client.post(
        "/v1/embeddings",
        json={"model": "fake-model", "input": ["a", "b", "c", "d", "e"]},
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "too_many_texts"
    # Caller must see both the limit and what they sent.
    assert "5" in e["message"]   # got
    assert "3" in e["message"]   # max
    assert e["extra"]["max"] == 3
    assert e["extra"]["got"] == 5


# ---- text_too_long ----

def test_text_too_long_returns_422_with_reason(client):
    r = client.post(
        "/v1/embeddings",
        json={
            "model": "fake-model",
            "input": ["ok", "x" * 100, "fine"],
        },
    )
    assert r.status_code == 422
    e = r.json()["error"]
    assert e["code"] == "text_too_long"
    assert "100" in e["message"]  # got
    assert "16" in e["message"]   # max
    assert e["extra"]["index"] == 1
    assert e["extra"]["got"] == 100
    assert e["extra"]["max"] == 16


# ---- embedder_unavailable ----

def test_embedder_failure_returns_503_with_reason(client):
    embedder = client.app.state.embedder

    def _boom(texts):
        raise ModelNotLoaded("weights file /models/bge-m3 missing")

    embedder.embed_documents = _boom  # type: ignore[method-assign]

    r = client.post(
        "/v1/embeddings",
        json={"model": "fake-model", "input": ["hello"]},
    )
    assert r.status_code == 503
    e = r.json()["error"]
    assert e["code"] == "embedder_unavailable"
    assert "weights file /models/bge-m3 missing" in e["message"]
    assert e["extra"]["model"] == "fake-model"
    assert e["extra"]["text_count"] == 1
    assert e["extra"]["exception_type"] == "ModelNotLoaded"


def test_embedder_failure_with_embedder_error_returns_503(client):
    embedder = client.app.state.embedder

    def _boom(texts):
        raise EmbedderError("CUDA out of memory")

    embedder.embed_documents = _boom  # type: ignore[method-assign]

    r = client.post(
        "/v1/embeddings",
        json={"model": "fake-model", "input": ["hello", "world"]},
    )
    assert r.status_code == 503
    e = r.json()["error"]
    assert e["code"] == "embedder_unavailable"
    assert "CUDA out of memory" in e["message"]
    assert e["extra"]["exception_type"] == "EmbedderError"
    assert e["extra"]["text_count"] == 2


# ---- unified /v1/models surface ----------------------------------------


def test_list_models_returns_embedder_and_reranker(client):
    """GET /v1/models returns both embedder and reranker rows with
    `type` discriminating them; rerankers always have dimensions=null."""
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    by_id = {m["id"]: m for m in body["data"]}

    # The fixture registers `fake-model` as an embedder; cross_encoder
    # registers `bge-reranker-v2-m3` at module import time.
    assert "fake-model" in by_id
    assert "bge-reranker-v2-m3" in by_id

    fake = by_id["fake-model"]
    assert fake["type"] == "embedder"
    assert fake["object"] == "model"
    assert fake["dimensions"] == 4  # FakeEmbedder.dim == 4

    rerank = by_id["bge-reranker-v2-m3"]
    assert rerank["type"] == "reranker"
    assert rerank["object"] == "model"
    assert rerank["dimensions"] is None


def test_get_embedder_model_returns_type_embedder(client):
    r = client.get("/v1/models/fake-model")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "fake-model"
    assert body["type"] == "embedder"
    assert body["dimensions"] == 4


def test_get_reranker_model_returns_type_reranker(client):
    r = client.get("/v1/models/bge-reranker-v2-m3")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "bge-reranker-v2-m3"
    assert body["type"] == "reranker"
    assert body["dimensions"] is None
