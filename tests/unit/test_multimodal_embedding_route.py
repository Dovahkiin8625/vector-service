"""POST /v1/multimodal_embeddings route behavior."""
from __future__ import annotations

import base64

import pytest

from vector_service.embeddings.image_base import ImageInput
from vector_service.embeddings.multimodal_base import MultimodalEmbedder


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")
HUGE_B64 = base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode("ascii")


class _StubMultimodalEmbedder(MultimodalEmbedder):
    """Records calls; returns deterministic vectors."""

    dim = 512
    model_name = "chinese-clip-vit-base-patch16"

    def __init__(self):
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[bytes]] = []
        self._loaded = False

    def load(self):
        self._loaded = True

    def embed_text(self, texts):
        self.text_calls.append(list(texts))
        return [[float(i)] * 512 for i in range(len(texts))]

    def embed_images(self, images):
        self.image_calls.append([i.data for i in images])
        return [[float(i)] * 512 for i in range(len(images))]


@pytest.fixture
def app_with_stub():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from vector_service.api.multimodal_embeddings import router as mm_router

    stub = _StubMultimodalEmbedder()
    stub.load()

    app = FastAPI()
    app.include_router(mm_router)
    app.state.multimodal_embedder = stub
    app.state.settings = type(
        "S",
        (),
        {
            "multimodal_embedding": type(
                "ME",
                (),
                {
                    "max_items_per_request": 4,
                    "max_text_chars": 256,
                    "max_image_bytes": 1024,
                    "allowed_mime": {"image/jpeg", "image/png", "image/webp"},
                },
            )()
        },
    )()

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

    return app, TestClient(app), stub


# ---- happy paths -----------------------------------------------------------


def test_text_only_input(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{"text": "一只猫"}, {"text": "一只狗"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "chinese-clip-vit-base-patch16"
    assert len(body["data"]) == 2
    assert all(len(d["embedding"]) == 512 for d in body["data"])
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["index"] == 1
    assert body["usage"]["prompt_tokens"] == 2
    # text tower called once with both items batched
    assert stub.text_calls == [["一只猫", "一只狗"]]
    assert stub.image_calls == []


def test_image_only_input(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{"image": {"data": VALID_PNG_B64, "mime": "image/png"}}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 512
    assert stub.text_calls == []
    assert len(stub.image_calls) == 1


def test_mixed_text_and_image_preserves_input_order(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [
                {"text": "第一"},
                {"image": {"data": VALID_PNG_B64, "mime": "image/png"}},
                {"text": "第三"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert len(body["data"]) == 3
    # Batched: one text call (positions 0,2), one image call (position 1).
    assert stub.text_calls == [["第一", "第三"]]
    assert len(stub.image_calls) == 1
    # Output order matches input order (text→0, image→1, text→2).
    assert body["data"][0]["embedding"][0] == 0.0  # text "第一" → i=0
    assert body["data"][1]["embedding"][0] == 0.0  # image position → i=0 in its batch
    assert body["data"][2]["embedding"][0] == 1.0  # text "第三" → i=1 in its batch


def test_single_item_object_form(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": {"text": "单条"},
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 1
    assert stub.text_calls == [["单条"]]


# ---- error paths -----------------------------------------------------------


def test_unknown_model_returns_404():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from vector_service.api.multimodal_embeddings import router as mm_router

    app = FastAPI()
    app.include_router(mm_router)
    app.state.multimodal_embedder = _StubMultimodalEmbedder()
    app.state.multimodal_embedder.load()
    app.state.settings = type(
        "S",
        (),
        {"multimodal_embedding": type("ME", (), {"max_items_per_request": 4, "max_text_chars": 256, "max_image_bytes": 1024, "allowed_mime": {"image/png"}})()},
    )()

    @app.exception_handler(HTTPException)
    async def _h(req: Request, exc: HTTPException):
        d = exc.detail
        if isinstance(d, dict) and "error" in d:
            return JSONResponse(status_code=exc.status_code, content=d)
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": "error", "message": str(d)}})

    client = TestClient(app)
    r = client.post(
        "/v1/multimodal_embeddings",
        json={"model": "does-not-exist", "input": [{"text": "x"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_embedder_not_loaded_returns_503():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from vector_service.api.multimodal_embeddings import router as mm_router

    app = FastAPI()
    app.include_router(mm_router)
    # Intentionally no multimodal_embedder on app.state.
    app.state.settings = type(
        "S",
        (),
        {"multimodal_embedding": type("ME", (), {"max_items_per_request": 4, "max_text_chars": 256, "max_image_bytes": 1024, "allowed_mime": {"image/png"}})()},
    )()

    @app.exception_handler(HTTPException)
    async def _h(req: Request, exc: HTTPException):
        d = exc.detail
        if isinstance(d, dict) and "error" in d:
            return JSONResponse(status_code=exc.status_code, content=d)
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": "error", "message": str(d)}})

    client = TestClient(app)
    r = client.post(
        "/v1/multimodal_embeddings",
        json={"model": "chinese-clip-vit-base-patch16", "input": [{"text": "x"}]},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "multimodal_embedder_unavailable"


def test_too_many_items_returns_422(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{"text": str(i)} for i in range(5)],  # limit is 4
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "too_many_items"


def test_text_too_long_returns_422(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{"text": "x" * 1024}],  # max_text_chars=256
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "text_too_long"


def test_unsupported_mime_returns_422(app_with_stub):
    app, client, stub = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{"image": {"data": VALID_PNG_B64, "mime": "image/bmp"}}],
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unsupported_mime"


def test_both_text_and_image_in_one_item_rejected_by_pydantic(app_with_stub):
    """Schema validator rejects items containing both text and image."""
    app, client, _ = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [
                {"text": "x", "image": {"data": VALID_PNG_B64, "mime": "image/png"}}
            ],
        },
    )
    assert r.status_code == 422


def test_neither_text_nor_image_rejected_by_pydantic(app_with_stub):
    app, client, _ = app_with_stub
    r = client.post(
        "/v1/multimodal_embeddings",
        json={
            "model": "chinese-clip-vit-base-patch16",
            "input": [{}],
        },
    )
    assert r.status_code == 422