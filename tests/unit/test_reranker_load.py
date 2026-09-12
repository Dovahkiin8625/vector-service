"""Unit tests for ``Reranker`` ABC and lifespan integration."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vector_service.core.errors import RerankerNotLoaded
from vector_service.core.metrics import MODEL_LOADED
from vector_service.rerankers.base import Reranker, ScoredHit


class _FakeReranker(Reranker):
    """Reranker double with predictable, monotonic scores."""

    model_name = "fake-reranker"

    def __init__(self, settings=None) -> None:
        self._impl: object | None = None
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self._impl = "ready"

    def rerank(self, query, documents, top_n=None):
        n = len(documents) if top_n is None else min(top_n, len(documents))
        scored = sorted(
            [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))],
            key=lambda h: h.score,
            reverse=True,
        )
        return scored[:n]


def test_reranker_is_abstract():
    with pytest.raises(TypeError):
        Reranker()  # type: ignore[abstract]


def test_fake_reranker_load_is_idempotent():
    r = _FakeReranker()
    r.load()
    r.load()
    assert r.load_calls == 2
    assert r._impl == "ready"


def test_scored_hit_is_frozen():
    h = ScoredHit(index=0, score=0.5)
    with pytest.raises(Exception):  # FrozenInstanceError
        h.index = 1  # type: ignore[misc]


# ----- lifespan / /readyz integration ----------------------------------


def _make_app(reranker):
    """Build a minimal app that mounts only the health router.

    We deliberately bypass ``create_app()`` so the production ``lifespan``
    (which would try to load a real embedder + reranker) is not exercised.
    Instead we set ``app.state.reranker`` directly so ``/readyz`` can
    report the reranker field correctly.
    """
    from vector_service.api.health import router as health_router

    app = FastAPI()
    app.include_router(health_router)
    # Embedder + image embedder must also be present — without them /readyz
    # would report them as not_loaded (which is fine for these assertions,
    # but setting minimal stand-ins keeps the body's "reranker" field the
    # sole variable under test).
    app.state.embedder = _AlwaysLoaded()
    app.state.image_embedder = _AlwaysLoadedImage()
    app.state.store = _NoopStore()
    app.state.reranker = reranker
    return app


class _AlwaysLoaded:
    """Embedder stand-in that reports loaded without a real model."""

    _impl = object()  # non-None sentinel

    def load(self):
        return None


class _AlwaysLoadedImage:
    """Image embedder stand-in that reports loaded without a real model."""

    _impl = object()  # non-None sentinel

    def load(self):
        return None


class _NoopStore:
    backend_name = "fake"
    uri = ""

    def _ensure_connected(self):
        return None

    def list_databases(self):
        return []

    def close(self):
        return None


def test_readyz_reports_reranker_ready():
    r = _FakeReranker()
    r.load()
    app = _make_app(r)
    MODEL_LOADED.labels(kind="reranker").set(1)
    with TestClient(app) as client:
        resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reranker"] == "ready"
    assert body["status"] == "ready"


def test_readyz_reports_reranker_not_loaded():
    r = _FakeReranker()  # never .load()'d
    app = _make_app(r)
    MODEL_LOADED.labels(kind="reranker").set(0)
    with TestClient(app) as client:
        resp = client.get("/readyz")
    # status still gated on embedder+store; assert at minimum the
    # reranker field is "not_loaded".
    body = resp.json()
    assert body["reranker"] == "not_loaded"


def test_reranker_not_loaded_exc_subclass():
    assert issubclass(RerankerNotLoaded, Exception)
    msg = RerankerNotLoaded("boom").args[0]
    assert msg == "boom"
