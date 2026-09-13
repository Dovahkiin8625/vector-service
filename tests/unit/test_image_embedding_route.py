"""POST /v1/image_embeddings route behavior."""
from __future__ import annotations

import base64

import pytest

from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
VALID_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode("ascii")
BMP_B64 = base64.b64encode(b"fake-bmp").decode("ascii")
HUGE_B64 = base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode("ascii")


class _StubImageEmbedder(ImageEmbedder):
    """Records calls; returns deterministic vectors."""

    dim = 768
    model_name = "stub"

    def __init__(self):
        self.calls = []

    def load(self): self._impl = object()
    @property
    def _impl(self): return getattr(self, "_impl_obj", None)
    @_impl.setter
    def _impl(self, v): self._impl_obj = v

    def embed_images(self, images):
        self.calls.append(("batch", [i.data for i in images]))
        return [[float(i)] * 768 for i in range(len(images))]

    def embed_query_image(self, image):
        self.calls.append(("query", image.data))
        return [1.0] * 768


@pytest.fixture
def app_with_stub(monkeypatch):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.testclient import TestClient
    from fastapi.responses import JSONResponse

    from vector_service.api.image_embeddings import router as img_router

    stub = _StubImageEmbedder()
    stub.load()  # mark _impl so /readyz-style checks pass

    app = FastAPI()
    app.include_router(img_router)
    app.state.image_embedder = stub
    app.state.settings = type("S", (), {
        "image_embedding": type("IE", (), {
            "max_images_per_request": 2,
            "max_image_bytes": 1024,
            "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
        })(),
    })()

    # Mirror main.py's HTTPException envelope so tests see the canonical shape.
    @app.exception_handler(HTTPException)
    async def _http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if (
            isinstance(detail, dict)
            and "error" in detail
            and isinstance(detail["error"], dict)
        ):
            inner = detail["error"]
            extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {**inner, **extras}},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "error", "message": str(detail)}},
        )

    return app, stub, TestClient(app)


def test_happy_path_single_image(app_with_stub):
    _, stub, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "openclip-vit-l-14"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "image_embedding"
    assert len(body["data"][0]["embedding"]) == 768
    assert body["usage"]["prompt_tokens"] == 1


def test_happy_path_list_of_images(app_with_stub):
    _, stub, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": [
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_JPEG_B64, "mime": "image/jpeg"},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2
    assert [d["index"] for d in body["data"]] == [0, 1]


def test_unknown_model_returns_model_not_found(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "does-not-exist",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_unsupported_mime_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": BMP_B64, "mime": "image/bmp"},
    })
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "unsupported_mime"


def test_image_too_large_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": HUGE_B64, "mime": "image/png"},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "image_too_large"


def test_too_many_images_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": [
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_PNG_B64, "mime": "image/png"},
            {"data": VALID_PNG_B64, "mime": "image/png"},
        ],
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_many_images"


def test_image_decode_failed_returns_422(app_with_stub):
    _, _, client = app_with_stub
    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": "!!!not-base64!!!", "mime": "image/png"},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "image_decode_failed"


def test_embedder_inference_failure_returns_503(app_with_stub):
    app, stub, client = app_with_stub

    class _BrokenEmbedder(_StubImageEmbedder):
        def embed_images(self, images):
            from vector_service.core.errors import ImageEmbedderError
            raise ImageEmbedderError("inference failed")

    app.state.image_embedder = _BrokenEmbedder()
    app.state.image_embedder.load()

    r = client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "image_embedder_unavailable"


def test_metrics_recorded(app_with_stub):
    _, stub, client = app_with_stub
    client.post("/v1/image_embeddings", json={
        "model": "openclip-vit-l-14",
        "input": {"data": VALID_PNG_B64, "mime": "image/png"},
    })
    from vector_service.core import metrics
    sample = metrics.IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(
        model="openclip-vit-l-14", status="ok"
    )._value.get()  # type: ignore[attr-defined]
    assert sample >= 1
