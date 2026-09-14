"""Image embedder lifespan wiring + /readyz reporting."""
from __future__ import annotations

import pytest

from vector_service.core import lifespan as lifespan_mod
from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder


class _RecordingImageEmbedder(ImageEmbedder):
    dim = 4
    model_name = "recording"

    def __init__(self, raise_on_load=None):
        self.load_called = 0
        self._raise = raise_on_load
        self._impl = None  # sentinel pattern

    def load(self):
        self.load_called += 1
        if self._raise is not None:
            raise self._raise
        self._impl = object()

    def embed_images(self, images):
        return [[0.0] * self.dim for _ in images]

    def embed_query_image(self, image):
        return [0.0] * self.dim


class _FakeTextEmbedder:
    model_name = "text"
    dim = 4

    def load(self):
        self.load_called = getattr(self, "load_called", 0) + 1


class _FakeStore:
    backend_name = "fake"
    uri = ""

    def _ensure_connected(self):
        pass

    def list_databases(self):
        return []

    def close(self):
        pass


def _enable_auto_load():
    """Flip every family's auto_load to True so the lifespan mirrors
    the legacy eager-load behaviour these regression tests were
    written against."""
    from vector_service.core.config import get_settings

    s = get_settings()
    s.embedding_auto_load = True
    s.reranker.auto_load = True
    s.image_embedding.auto_load = True
    s.multimodal_embedding.auto_load = True
    return s


@pytest.fixture
def patched_lifespan_deps(monkeypatch):
    _enable_auto_load()
    img_embedder = _RecordingImageEmbedder()
    text_embedder = _FakeTextEmbedder()
    fake_store = _FakeStore()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: text_embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: img_embedder)
    monkeypatch.setattr(
        lifespan_mod,
        "build_reranker",
        lambda s: type("R", (), {"model_name": "r", "load": lambda self: None})(),
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    return img_embedder, app


def test_lifespan_calls_image_embedder_load(patched_lifespan_deps):
    img, app = patched_lifespan_deps
    from fastapi.testclient import TestClient

    with TestClient(app):
        assert img.load_called == 1


def test_lifespan_sets_image_embedder_on_state(patched_lifespan_deps):
    img, app = patched_lifespan_deps
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        # The embedder is reachable via the app.state we built
        assert app.state.image_embedder is img


def test_readyz_reports_image_embedder_loaded(patched_lifespan_deps):
    _, app = patched_lifespan_deps
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["image_embedder"] == "loaded"


def test_readyz_reports_image_embedder_not_loaded_on_failure(monkeypatch):
    """A failed image-embedder load is logged but does NOT 503 /readyz.

    /readyz gates only on the vector store now; per-family load state
    is surfaced via the response body.
    """
    _enable_auto_load()
    img = _RecordingImageEmbedder(raise_on_load=ModelNotLoadedForImages("disk full"))
    text_embedder = _FakeTextEmbedder()
    fake_store = _FakeStore()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: text_embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: img)
    monkeypatch.setattr(
        lifespan_mod,
        "build_reranker",
        lambda s: type("R", (), {"model_name": "r", "load": lambda self: None})(),
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["image_embedder"] == "not_loaded"
        assert body["store"] == "ok"


def test_lifespan_fails_open_on_unexpected_image_embedder_exception(monkeypatch):
    """Regression: lifespan must NOT propagate non-ModelNotLoadedForImages
    exceptions (e.g. huggingface_hub.RepositoryNotFoundError from a bad
    hf_repo, or a network error). The image embedder fails closed (the
    app comes up but /readyz reports image_embedder=not_loaded), mirroring
    the text embedder's fail-open policy.
    """
    class _BoomEmbedder(ImageEmbedder):
        dim = 768
        model_name = "boom"

        def __init__(self):
            self._impl = None

        def load(self):
            raise RuntimeError("disk I/O error")  # not ModelNotLoadedForImages

        def embed_images(self, images): return []
        def embed_query_image(self, image): return []

    img = _BoomEmbedder()
    text_embedder = _FakeTextEmbedder()
    fake_store = _FakeStore()
    _enable_auto_load()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: text_embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_image_embedder", lambda s: img)
    monkeypatch.setattr(
        lifespan_mod,
        "build_reranker",
        lambda s: type("R", (), {"model_name": "r", "load": lambda self: None})(),
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    # App must come up despite the unexpected exception.
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["image_embedder"] == "not_loaded"
        assert body["embedder"] == "loaded"  # text embedder still works
        assert body["store"] == "ok"