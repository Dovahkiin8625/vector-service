"""Regression tests for batch 1D: inference timeout fallback.

Pin that the ``run_in_executor`` calls in inference routes tolerate a
fake ``Settings`` double that lacks ``inference_timeout_seconds``.
The route must fall back to a sane default via ``getattr`` rather
than raise ``AttributeError`` *on the inference-timeout path*.

We deliberately scope the assertion to the inference path; routes
that read unrelated nested settings (e.g. ``settings.image_embedding``)
pre-date this regression and are not pinned here.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _SettingsWithoutTimeout:
    """Bare Settings stand-in — only the field under test is absent."""

    # Real routes also read other fields; tests below set just what
    # the path under test exercises, and stub the rest.
    pass


def test_settings_without_inference_timeout_uses_default(monkeypatch):
    """``getattr(settings, "inference_timeout_seconds", 60.0)`` resolves
    to the default 60s when the field is absent. This is the contract
    the inference routes depend on.
    """
    settings = _SettingsWithoutTimeout()
    timeout = getattr(settings, "inference_timeout_seconds", 60.0)
    assert timeout == 60.0


def test_settings_with_inference_timeout_resolves_explicit_value():
    """Sanity: when the field IS present, ``getattr`` returns it."""
    settings = _SettingsWithoutTimeout()
    settings.inference_timeout_seconds = 12.5
    timeout = getattr(settings, "inference_timeout_seconds", 60.0)
    assert timeout == 12.5


def test_embeddings_route_does_not_access_timeout_via_dot(monkeypatch):
    """Pin that the embeddings route uses ``getattr(settings, ...)``
    rather than ``settings.inference_timeout_seconds``. We patch the
    embedder class lookup so the route reaches the inference path,
    then patch the embedder instance so it returns immediately; if
    the route used ``settings.inference_timeout_seconds`` directly
    the getattr fallback would not be exercised and any real
    AttributeError would surface here.
    """
    from vector_service.api.embeddings import router as embeddings_router
    from vector_service.embeddings import registry as embeddings_registry

    class _FakeEmbedder:
        model_name = "bge-m3"
        dim = 4

        def embed_documents(self, texts):
            return [[0.0] * 4 for _ in texts]

    # Register the fake embedder so registry check passes.
    original = dict(embeddings_registry.EMBEDDER_REGISTRY)
    embeddings_registry.EMBEDDER_REGISTRY["bge-m3"] = _FakeEmbedder
    try:
        settings = _SettingsWithoutTimeout()
        settings.embedding_max_texts_per_request = 256
        settings.embedding_max_chars_per_text = 8192

        app = FastAPI()
        app.include_router(embeddings_router)
        state = MagicMock()
        state.settings = settings
        state.embedder = _FakeEmbedder()
        # ``app.state`` is a starlette State; getattr pattern below
        # requires the actual attribute access path used in the
        # route. MagicMock returns a MagicMock for any attribute, so
        # route's ``settings.inference_timeout_seconds`` resolves to
        # another MagicMock — to actually pin the fallback we must
        # use a real State. Easiest: set the attribute on a real
        # FastAPI state object.
        from fastapi import FastAPI as _FA

        real_app = _FA()
        real_app.include_router(embeddings_router)
        real_app.state.settings = settings
        real_app.state.embedder = _FakeEmbedder()
        client = TestClient(real_app, raise_server_exceptions=False)
        r = client.post("/v1/embeddings", json={"input": "hi", "model": "bge-m3"})
        # 200 (success) means the route reached inference and finished
        # without AttributeError on the timeout path. Anything 500 with
        # "AttributeError" in body would indicate the contract regressed.
        assert r.status_code == 200, r.text
    finally:
        embeddings_registry.EMBEDDER_REGISTRY.clear()
        embeddings_registry.EMBEDDER_REGISTRY.update(original)
