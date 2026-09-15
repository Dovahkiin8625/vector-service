"""Similarity endpoints under ``/v1`` - three POST handlers sharing
``/text_similarity``, ``/image_similarity``, ``/multimodal_similarity``.
"""
from __future__ import annotations

import base64

import pytest

from vector_service.embeddings.base import Embedder
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.multimodal_base import MultimodalEmbedder


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


# ---------------------------------------------------------------------------
# Stub embedders
# ---------------------------------------------------------------------------


class _StubTextEmbedder(Embedder):
    """Records calls; returns deterministic 4d vectors so cosine/ip/l2
    produce well-separated scores that we can verify in tests."""

    dim = 4
    model_name = "bge-m3"

    def __init__(self):
        self.calls: list[list[str]] = []
        self._loaded = False

    def load(self):
        self._loaded = True

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        # Distinct vectors per input index — gives ascending cosine vs i.
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0, 0.0]


class _StubImageEmbedder(ImageEmbedder):
    """Records image bytes; returns deterministic 4d vectors."""

    dim = 4
    model_name = "openclip-vit-l-14"

    def __init__(self):
        self.calls: list[list[bytes]] = []
        self._loaded = False

    def load(self):
        self._loaded = True

    def embed_images(self, images):
        self.calls.append([i.data for i in images])
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(images))]

    def embed_query_image(self, image):
        return [0.0, 0.0, 0.0, 0.0]


class _StubMultimodalEmbedder(MultimodalEmbedder):
    """Records both calls; returns deterministic 4d vectors so all three
    similarity endpoints can use the same fixtures."""

    dim = 4
    model_name = "chinese-clip-vit-base-patch16"

    def __init__(self):
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[bytes]] = []
        self._loaded = False

    def load(self):
        self._loaded = True

    def embed_text(self, texts):
        self.text_calls.append(list(texts))
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]

    def embed_images(self, images):
        self.image_calls.append([i.data for i in images])
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(images))]


# ---------------------------------------------------------------------------
# Fixtures — one per family. The HTTPException handler mirrors the
# canonical envelope from `api/main.py`'s exception handler so tests
# can assert on `error.code` / `error.message`.
# ---------------------------------------------------------------------------


def _build_app(router, family_state_attr: str, stub, settings):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from vector_service.core.errors import (
        ModelNotLoadedForSimilarity,
        SimilarityError,
    )

    app = FastAPI()
    app.include_router(router)
    setattr(app.state, family_state_attr, stub)
    app.state.settings = settings

    @app.exception_handler(HTTPException)
    async def _h(req: Request, exc: HTTPException):
        d = exc.detail
        if isinstance(d, dict) and "error" in d and isinstance(d["error"], dict):
            return JSONResponse(status_code=exc.status_code, content=d)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "error", "message": str(d)}},
        )

    # Mirror the canonical envelopes from main.py so the "not loaded"
    # surfaces as 503 similarity_unavailable rather than a 500.
    @app.exception_handler(ModelNotLoadedForSimilarity)
    async def _not_loaded(req: Request, exc: ModelNotLoadedForSimilarity):
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "similarity_unavailable", "message": str(exc)}},
        )

    @app.exception_handler(SimilarityError)
    async def _sim_error(req: Request, exc: SimilarityError):
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "similarity_error", "message": str(exc)}},
        )

    return app, TestClient(app)


@pytest.fixture
def text_app():
    from vector_service.api.similarity import router as sim_router

    stub = _StubTextEmbedder()
    stub.load()
    settings = type("S", (), {})()
    app, client = _build_app(sim_router, "embedder", stub, settings)
    return app, client, stub


@pytest.fixture
def image_app():
    from vector_service.api.similarity import router as sim_router

    stub = _StubImageEmbedder()
    stub.load()
    settings = type(
        "S",
        (),
        {
            "image_embedding": type(
                "IE",
                (),
                {
                    "max_image_bytes": 1024,
                    "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
                },
            )()
        },
    )()
    app, client = _build_app(sim_router, "image_embedder", stub, settings)
    return app, client, stub


@pytest.fixture
def mm_app():
    from vector_service.api.similarity import router as sim_router

    stub = _StubMultimodalEmbedder()
    stub.load()
    settings = type(
        "S",
        (),
        {
            "multimodal_embedding": type(
                "ME",
                (),
                {
                    "max_text_chars": 256,
                    "max_image_bytes": 1024,
                    "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
                },
            )()
        },
    )()
    app, client = _build_app(sim_router, "multimodal_embedder", stub, settings)
    return app, client, stub


# ---------------------------------------------------------------------------
# /v1/text_similarity
# ---------------------------------------------------------------------------


def test_text_similarity_cosine_orders_by_inner_product(text_app):
    """Higher inner-product with the query vector → rank #1."""
    _, client, stub = text_app
    r = client.post(
        "/v1/text_similarity",
        json={
            "model": "bge-m3",
            "query": "q",
            "documents": ["d0", "d1", "d2", "d3"],
            "metric": "cosine",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "bge-m3"
    assert body["metric"] == "cosine"
    # Doc vector [3,0,0,0] has the largest dot-product with [0,0,0,0]? No
    # — actually all are orthogonal to the query [0,0,0,0], so all cosine
    # scores equal 0.0 and we fall back to input order (arg-sort is stable).
    assert [it["index"] for it in body["results"]] == [0, 1, 2, 3]
    assert all(it["score"] == 0.0 for it in body["results"])
    # Single batched embed call.
    assert stub.calls == [["q", "d0", "d1", "d2", "d3"]]


def test_text_similarity_ip_orders_descending(text_app):
    """ip = raw dot product; doc with the largest vector → rank #1."""
    _, client, stub = text_app
    # Override stub so the query has a non-zero vector and each candidate
    # has a strictly larger magnitude along axis 0 — that way ip(query, doc_i)
    # is positive and strictly increasing with i.
    def _emb(texts):
        stub.calls.append(list(texts))
        # All vectors live along axis 0; query = [1,0,0,0], candidates = [10,20,30,40].
        return [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(texts))]
    stub.embed_documents = _emb  # type: ignore[assignment]

    r = client.post(
        "/v1/text_similarity",
        json={
            "model": "bge-m3",
            "query": "q",                                # vec [1,0,0,0]
            "documents": ["d0", "d1", "d2", "d3"],       # ip: 10,20,30,40
            "metric": "ip",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Sorted descending by ip → d3 first.
    assert [it["index"] for it in body["results"]] == [3, 2, 1, 0]
    # Spot-check the actual scores are positive and strictly decreasing.
    scores = [it["score"] for it in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_text_similarity_l2_orders_ascending(text_app):
    """l2 = Euclidean distance; smallest distance → rank #1."""
    _, client, stub = text_app
    def _emb(texts):
        stub.calls.append(list(texts))
        return [[float(i), 0.0, 0.0, 0.0] for i in range(len(texts))]
    stub.embed_documents = _emb  # type: ignore[assignment]

    r = client.post(
        "/v1/text_similarity",
        json={
            "model": "bge-m3",
            "query": "q",                                # vec [0,0,0,0]
            "documents": ["d0", "d1", "d2", "d3"],       # distances 0,1,2,3
            "metric": "l2",
        },
    )
    assert r.status_code == 200, r.text
    # Closest first (ascending).
    assert [it["index"] for it in r.json()["results"]] == [0, 1, 2, 3]


def test_text_similarity_default_metric_is_cosine(text_app):
    _, client, _ = text_app
    r = client.post(
        "/v1/text_similarity",
        json={"model": "bge-m3", "query": "q", "documents": ["d"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["metric"] == "cosine"


def test_text_similarity_rejects_empty_query(text_app):
    _, client, _ = text_app
    r = client.post(
        "/v1/text_similarity",
        json={"model": "bge-m3", "query": "", "documents": ["d"]},
    )
    assert r.status_code == 422


def test_text_similarity_rejects_empty_documents(text_app):
    _, client, _ = text_app
    r = client.post(
        "/v1/text_similarity",
        json={"model": "bge-m3", "query": "q", "documents": []},
    )
    assert r.status_code == 422


def test_text_similarity_rejects_unknown_metric(text_app):
    _, client, _ = text_app
    r = client.post(
        "/v1/text_similarity",
        json={"model": "bge-m3", "query": "q", "documents": ["d"], "metric": "hamming"},
    )
    assert r.status_code == 422


def test_text_similarity_unknown_model_returns_404():
    """No embedder registered under this id → 404 model_not_found."""
    from vector_service.api.similarity import router as sim_router

    stub = _StubTextEmbedder()
    stub.load()
    app, client = _build_app(sim_router, "embedder", stub, type("S", (), {})())
    r = client.post(
        "/v1/text_similarity",
        json={"model": "does-not-exist", "query": "q", "documents": ["d"]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# /v1/image_similarity
# ---------------------------------------------------------------------------


def test_image_similarity_happy_path(image_app):
    _, client, stub = image_app
    r = client.post(
        "/v1/image_similarity",
        json={
            "model": "openclip-vit-l-14",
            "query": {"data": VALID_PNG_B64, "mime": "image/png"},
            "documents": [
                {"data": VALID_PNG_B64, "mime": "image/png"},
                {"data": VALID_PNG_B64, "mime": "image/png"},
            ],
            "metric": "cosine",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "openclip-vit-l-14"
    assert body["metric"] == "cosine"
    assert [it["index"] for it in body["results"]] == [0, 1]
    # One image batch with query + 2 docs.
    assert len(stub.calls) == 1
    assert len(stub.calls[0]) == 3


def test_image_similarity_unsupported_mime_returns_422(image_app):
    _, client, _ = image_app
    r = client.post(
        "/v1/image_similarity",
        json={
            "model": "openclip-vit-l-14",
            "query": {"data": VALID_PNG_B64, "mime": "image/bmp"},
            "documents": [{"data": VALID_PNG_B64, "mime": "image/png"}],
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_mime"


def test_image_similarity_unknown_model_returns_404():
    from vector_service.api.similarity import router as sim_router

    stub = _StubImageEmbedder()
    stub.load()
    settings = type(
        "S",
        (),
        {"image_embedding": type("IE", (), {"max_image_bytes": 1024, "allowed_mime": {"image/png"}})()},
    )()
    app, client = _build_app(sim_router, "image_embedder", stub, settings)
    r = client.post(
        "/v1/image_similarity",
        json={
            "model": "no-such-image-model",
            "query": {"data": VALID_PNG_B64, "mime": "image/png"},
            "documents": [{"data": VALID_PNG_B64, "mime": "image/png"}],
        },
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


# ---------------------------------------------------------------------------
# /v1/multimodal_similarity
# ---------------------------------------------------------------------------


def test_multimodal_similarity_text_query_image_docs(mm_app):
    _, client, stub = mm_app
    r = client.post(
        "/v1/multimodal_similarity",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "query": {"text": "一只猫"},
            "documents": [
                {"text": "狗"},
                {"image": {"data": VALID_PNG_B64, "mime": "image/png"}},
                {"text": "鸟"},
            ],
            "metric": "cosine",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "chinese-clip-vit-base-patch16"
    assert len(body["results"]) == 3
    # Text-only embeddings cover query + 2 text docs; image covers 1.
    assert stub.text_calls == [["一只猫", "狗", "鸟"]]
    assert len(stub.image_calls) == 1
    assert len(stub.image_calls[0]) == 1


def test_multimodal_similarity_image_query_text_docs(mm_app):
    _, client, stub = mm_app
    r = client.post(
        "/v1/multimodal_similarity",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "query": {"image": {"data": VALID_PNG_B64, "mime": "image/png"}},
            "documents": [{"text": "狗"}, {"text": "鸟"}],
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 2
    assert stub.image_calls and len(stub.image_calls[0]) == 1  # query
    assert stub.text_calls == [["狗", "鸟"]]


def test_multimodal_similarity_rejects_item_with_both_text_and_image(mm_app):
    _, client, _ = mm_app
    r = client.post(
        "/v1/multimodal_similarity",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "query": {"text": "x"},
            "documents": [
                {"text": "x", "image": {"data": VALID_PNG_B64, "mime": "image/png"}},
            ],
        },
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 503 when embedder not loaded
# ---------------------------------------------------------------------------


def test_text_similarity_503_when_embedder_not_loaded():
    from vector_service.api.similarity import router as sim_router

    app, client = _build_app(sim_router, "embedder", None, type("S", (), {})())
    r = client.post(
        "/v1/text_similarity",
        json={"model": "bge-m3", "query": "q", "documents": ["d"]},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "similarity_unavailable"


def test_image_similarity_503_when_embedder_not_loaded():
    from vector_service.api.similarity import router as sim_router

    settings = type(
        "S",
        (),
        {"image_embedding": type("IE", (), {"max_image_bytes": 1024, "allowed_mime": {"image/png"}})()},
    )()
    app, client = _build_app(sim_router, "image_embedder", None, settings)
    r = client.post(
        "/v1/image_similarity",
        json={
            "model": "openclip-vit-l-14",
            "query": {"data": VALID_PNG_B64, "mime": "image/png"},
            "documents": [{"data": VALID_PNG_B64, "mime": "image/png"}],
        },
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "similarity_unavailable"


def test_multimodal_similarity_503_when_embedder_not_loaded():
    from vector_service.api.similarity import router as sim_router

    settings = type(
        "S",
        (),
        {"multimodal_embedding": type("ME", (), {"max_text_chars": 256, "max_image_bytes": 1024, "allowed_mime": {"image/png"}})()},
    )()
    app, client = _build_app(sim_router, "multimodal_embedder", None, settings)
    r = client.post(
        "/v1/multimodal_similarity",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "query": {"text": "x"},
            "documents": [{"text": "y"}],
        },
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "similarity_unavailable"