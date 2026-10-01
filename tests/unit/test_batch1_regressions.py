"""Regression tests for batch 1 reliability fixes.

Pinned scenarios:

- ``test_lifespan_cleans_up_embedder_when_reranker_load_raises`` —
  when the reranker eager-load raises AFTER the text embedder
  already loaded, the finally clause must release the embedder's
  resources. Without the fix the embedder instance is silently
  abandoned until process death.
- ``test_lifespan_reranker_failure_is_fail_open`` — reranker
  auto-load failure no longer re-raises; the process comes up and
  ``/readyz`` still responds.
"""
from __future__ import annotations

from vector_service.embeddings.base import Embedder
from vector_service.rerankers.base import Reranker


def test_embedder_base_load_is_abstract():
    """Sanity: ensure the Embedder ABC still requires load()."""
    import pytest

    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


# ---- shared fakes --------------------------------------------------------


class _RecordingEmbedder(Embedder):
    """Records whether `load()` ran and lets a test inject a failure.

    Mirrors the `_impl` convention used by `BGEM3Embedder` so that
    `health._embedder_loaded()` can observe load state the same way.
    """

    dim = 4
    model_name = "recording"

    def __init__(self, raise_on_load: Exception | None = None):
        self.load_called = 0
        self.unload_called = 0
        self._raise = raise_on_load
        self._impl = None

    def load(self) -> None:
        self.load_called += 1
        if self._raise is not None:
            raise self._raise
        self._impl = object()  # sentinel — non-None = loaded

    def unload(self) -> None:
        self.unload_called += 1
        self._impl = None

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


class _RecordingReranker(Reranker):
    """Records whether `load()` ran. Mirrors `_RecordingEmbedder`."""

    model_name = "recording"

    def __init__(self, raise_on_load: Exception | None = None):
        self.load_called = 0
        self.unload_called = 0
        self._raise = raise_on_load

    def load(self) -> None:
        self.load_called += 1
        if self._raise is not None:
            raise self._raise

    def unload(self) -> None:
        self.unload_called += 1

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit
        return [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))]


def _settings_with_auto_load():
    from vector_service.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    settings.embedding_auto_load = True
    settings.reranker.auto_load = True
    settings.image_embedding.auto_load = True
    settings.multimodal_embedding.auto_load = True
    return settings


def _fake_store():
    return type(
        "S",
        (),
        {
            "backend_name": "fake",
            "uri": "",
            "_ensure_connected": lambda self: None,
            "close": lambda self: None,
            "list_databases": lambda self: [],
        },
    )()


def _fake_image_embedder():
    return type(
        "IE",
        (),
        {
            "_impl": object(),
            "model_name": "fake-img",
            "dim": 4,
            "load": lambda self: None,
            "unload": lambda self: None,
        },
    )()


# ---- 1. lifespan cleanup -------------------------------------------------


def test_lifespan_cleans_up_embedder_when_reranker_load_raises(monkeypatch):
    """If the reranker eager-load raises AFTER the text embedder was
    successfully loaded, the finally clause in lifespan must call
    ``unload()`` on every populated slot so resources are released.
    Pins batch-1 fix (resource leak on startup failure).
    """
    from vector_service.core import lifespan as lifespan_mod

    _settings_with_auto_load()

    embedder = _RecordingEmbedder()
    image_embedder = _fake_image_embedder()

    class _BoomReranker:
        model_name = "boom-reranker"

        def load(self) -> None:
            raise RuntimeError("reranker weights missing")

        def unload(self) -> None:
            pass

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: _fake_store())
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: image_embedder)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _BoomReranker())

    from fastapi import FastAPI

    app = FastAPI(lifespan=lifespan_mod.lifespan)

    # Must NOT raise: lifespan is now uniformly fail-open.
    from fastapi.testclient import TestClient
    with TestClient(app):
        pass

    assert embedder.load_called == 1
    assert embedder.unload_called == 1
    assert embedder._impl is None


def test_lifespan_reranker_failure_is_fail_open(monkeypatch):
    """Reranker auto-load failure must NOT kill the process (pins
    batch-1 fix: lifespan fail-open/closed semantics unified).
    """
    from vector_service.core import lifespan as lifespan_mod
    from vector_service.api.health import router as health_router

    _settings_with_auto_load()

    embedder = _RecordingEmbedder()
    image_embedder = _fake_image_embedder()

    class _BoomReranker:
        model_name = "boom"

        def load(self) -> None:
            raise KeyError("unknown backend 'whatever'")

        def unload(self) -> None:
            pass

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: _fake_store())
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: image_embedder)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _BoomReranker())

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
