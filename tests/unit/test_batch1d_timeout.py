"""Inference timeout: ``settings.inference_timeout_seconds`` is honored.

The embeddings route reads the timeout straight from settings and
wraps the executor call in ``asyncio.wait_for``; when the model takes
longer than configured the client gets 503 ``embedder_unavailable``.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


class _Settings:
    embedding_max_texts_per_request = 256
    embedding_max_chars_per_text = 8192
    inference_timeout_seconds = 0.2


class _SlowEmbedder:
    model_name = "bge-m3"
    dim = 4

    def embed_documents(self, texts):
        time.sleep(1.0)
        return [[0.0] * 4 for _ in texts]


@pytest.fixture
def client():
    from vector_service.api.embeddings import router as embeddings_router
    from vector_service.embeddings import registry as embeddings_registry

    # Save the built-in so the finalizer restores it rather than
    # leaving bge-m3 unregistered for later test files.
    saved = embeddings_registry.EMBEDDER_REGISTRY["bge-m3"]
    embeddings_registry.EMBEDDER_REGISTRY["bge-m3"] = _SlowEmbedder
    try:
        app = FastAPI()
        app.include_router(embeddings_router)
        app.state.settings = _Settings()
        app.state.embedder = _SlowEmbedder()

        @app.exception_handler(HTTPException)
        async def _http_error_handler(request: Request, exc: HTTPException):
            # Same envelope unwrap main.py installs: route detail is
            # {"error": {...}}; return it at top level.
            detail = exc.detail
            if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
                return JSONResponse(status_code=exc.status_code, content=detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": "error", "message": str(detail)}},
            )

        yield TestClient(app)
    finally:
        embeddings_registry.EMBEDDER_REGISTRY["bge-m3"] = saved


def test_inference_exceeding_timeout_is_503(client):
    r = client.post("/v1/embeddings", json={"input": "hi", "model": "bge-m3"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "embedder_unavailable"
    assert "did not finish within 0.2s" in body["error"]["message"]
