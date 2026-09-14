"""Unit tests for the hot-load / hot-unload endpoints and ``ModelSlot``.

The routes live on the existing ``models`` router:

    POST /v1/models/{model_id}/load
    POST /v1/models/{model_id}/unload

Behaviour under test:

- 404 ``model_not_found`` for unknown model ids.
- 200 ``loaded`` with ``{id, type, status, dimensions}`` on successful load.
- Load is idempotent for the same id; a different id while one is loaded
  surfaces 409 ``conflict_loaded`` (caller must unload first).
- A load failure (e.g. backend raises ``ModelNotLoaded``) surfaces as
  503 ``model_load_failed`` and leaves the slot in its prior state.
- Unload clears the slot and returns 200 ``unloaded``; unloading a slot
  that is already empty returns 409 ``not_loaded``.
- Per-family ``ModelSlot`` exposes a non-blocking lock so concurrent
  load/unload calls return 409 ``model_busy``.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.models import router as models_router
from vector_service.core.errors import (
    EmbedderError,
    ImageEmbedderError,
    MultimodalEmbedderError,
    RerankerError,
)
from vector_service.core.logging import request_id_var
from vector_service.core.middleware import RequestIDMiddleware
from vector_service.core.model_lifecycle import (
    ConcurrentModelOperation,
    DifferentModelLoaded,
    ModelSlot,
)


# ---- fakes ---------------------------------------------------------------


class _FakeEmbedder:
    """Stand-in for any concrete Embedder subclass used by tests."""

    dim = 4
    model_name = "fake-embedder"

    instances: list["_FakeEmbedder"] = []  # class-level: every ctor appends

    def __init__(self, settings=None):
        self._settings = settings
        self.load_called = 0
        self.unload_called = 0
        type(self).instances.append(self)

    def load(self):
        self.load_called += 1

    def unload(self):
        self.unload_called += 1

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


class _FakeImageEmbedder:
    dim = 8
    model_name = "fake-image-embedder"
    instances: list[Any] = []

    def __init__(self, settings=None):
        self.load_called = 0
        self.unload_called = 0
        type(self).instances.append(self)

    def load(self):
        self.load_called += 1

    def unload(self):
        self.unload_called += 1

    def embed_images(self, images):
        return [[0.0] * self.dim for _ in images]

    def embed_query_image(self, image):
        return [0.0] * self.dim


class _FakeMultimodalEmbedder:
    dim = 16
    model_name = "fake-multimodal-embedder"
    instances: list[Any] = []

    def __init__(self, settings=None):
        self.load_called = 0
        self.unload_called = 0
        type(self).instances.append(self)

    def load(self):
        self.load_called += 1

    def unload(self):
        self.unload_called += 1

    def embed_text(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_images(self, images):
        return [[0.0] * self.dim for _ in images]


class _FakeReranker:
    model_name = "fake-reranker"
    instances: list[Any] = []

    def __init__(self, settings=None):
        self.load_called = 0
        self.unload_called = 0
        type(self).instances.append(self)

    def load(self):
        self.load_called += 1

    def unload(self):
        self.unload_called += 1

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit
        return [ScoredHit(index=i, score=1.0 - i * 0.1) for i in range(len(documents))]


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    """Reset class-level instance logs between tests so counts are isolated."""
    _FakeEmbedder.instances.clear()
    _FakeImageEmbedder.instances.clear()
    _FakeMultimodalEmbedder.instances.clear()
    _FakeReranker.instances.clear()
    yield


# ---- app fixture ---------------------------------------------------------


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
            first_msg = f"{loc}: {raw_msg}" if loc else f"{err_type}: {raw_msg}"
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
    """A FastAPI app whose models router points at fakes for all 4 families."""
    from vector_service.embeddings import image_registry, multimodal_registry
    from vector_service.embeddings import registry as emb_registry
    from vector_service.rerankers import registry as rerank_registry

    monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "fake-embedder", _FakeEmbedder)
    monkeypatch.setitem(image_registry.IMAGE_EMBEDDER_REGISTRY, "fake-image-embedder", _FakeImageEmbedder)
    monkeypatch.setitem(
        multimodal_registry.MULTIMODAL_EMBEDDER_REGISTRY,
        "fake-multimodal-embedder",
        _FakeMultimodalEmbedder,
    )
    monkeypatch.setitem(rerank_registry.RERANKER_REGISTRY, "fake-reranker", _FakeReranker)

    a = FastAPI()
    a.add_middleware(RequestIDMiddleware)
    a.include_router(models_router)
    a.add_exception_handler(HTTPException, _http_error_handler)
    a.add_exception_handler(RequestValidationError, _validation_handler)

    # The lifespan is NOT exercised in this fixture — slots are wired by
    # the route layer the first time a load/unload is requested. We
    # pre-create empty slots mirroring the lifespan layout so the routes
    # have something to read.
    from vector_service.core.model_lifecycle import attach_default_slots

    attach_default_slots(a, settings=_SettingsStub())
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


class _SettingsStub:
    """Stand-in for ``Settings`` exposing the four settings blocks."""

    def __init__(self):
        self.embedding = _DummySettings()
        self.image_embedding = _DummySettings()
        self.multimodal_embedding = _DummySettings()
        self.reranker = _DummySettings()


class _DummySettings:
    pass


# ---- ModelSlot unit tests ------------------------------------------------


class TestModelSlot:
    def test_load_stores_instance_and_calls_load(self):
        slot = ModelSlot("embedder")
        out = slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        assert isinstance(out, _FakeEmbedder)
        assert out.load_called == 1
        assert slot.get() is out

    def test_load_is_idempotent_for_same_class(self):
        slot = ModelSlot("embedder")
        a = slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        b = slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        # Same id — load() on existing returns current, doesn't construct new.
        assert a is b
        assert a.load_called == 1
        assert len(_FakeEmbedder.instances) == 1

    def test_load_different_id_raises_different_model_loaded(self):
        """If the slot already holds an instance whose id differs from the
        one we just built, we abort and the slot keeps the original."""
        slot = ModelSlot("embedder")

        class _OtherEmbedder(_FakeEmbedder):
            model_name = "other-embedder"

        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        first = slot.get()
        with pytest.raises(DifferentModelLoaded):
            slot.load(_OtherEmbedder, lambda: _OtherEmbedder())
        # Slot still points at the first instance; the freshly-built one
        # had its unload() called as cleanup.
        assert slot.get() is first

    def test_load_instance_load_failure_keeps_slot_empty(self):
        """When construction succeeds but ``instance.load()`` raises,
        the slot must remain empty — a half-populated instance would
        leak GPU memory and confuse subsequent requests."""
        slot = ModelSlot("embedder")

        class _ExploderEmbedder:
            dim = 0
            model_name = "exploder"

            def __init__(self, settings=None):
                self._impl = "partial"

            def load(self):
                raise EmbedderError("backend init failed")

            def unload(self):
                # The slot must call this on the partial instance so we
                # don't leak ``_impl``. Verify it does.
                self._impl = None

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        partial = []
        def _factory():
            inst = _ExploderEmbedder()
            partial.append(inst)
            return inst

        with pytest.raises(EmbedderError):
            slot.load(_ExploderEmbedder, _factory)
        assert slot.get() is None
        # The freshly-built instance had ``unload()`` called so its
        # partial state is cleaned up.
        assert partial and partial[0]._impl is None

    def test_unload_releases_instance(self):
        slot = ModelSlot("embedder")
        instance = slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        assert slot.unload() is True
        assert slot.get() is None
        assert instance.unload_called == 1

    def test_unload_when_empty_returns_false(self):
        slot = ModelSlot("embedder")
        assert slot.unload() is False

    def test_concurrent_load_raises_concurrent(self):
        """If the slot is busy, a second caller gets ConcurrentModelOperation."""
        slot = ModelSlot("embedder")
        started = threading.Event()
        proceed = threading.Event()

        def _slow_factory():
            started.set()
            proceed.wait(timeout=2.0)
            return _FakeEmbedder()

        def _first():
            slot.load(_FakeEmbedder, _slow_factory)

        t = threading.Thread(target=_first)
        t.start()
        try:
            started.wait(timeout=2.0)
            # While the first load holds the slot, a second load must fail.
            with pytest.raises(ConcurrentModelOperation):
                slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        finally:
            proceed.set()
            t.join(timeout=2.0)

    def test_concurrent_unload_raises_concurrent(self):
        slot = ModelSlot("embedder")
        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())

        started = threading.Event()
        proceed = threading.Event()

        def _hold_lock():
            # Manually grab the lock to simulate an inflight op.
            assert slot._lock.acquire(blocking=True)
            try:
                started.set()
                proceed.wait(timeout=2.0)
            finally:
                slot._lock.release()

        t = threading.Thread(target=_hold_lock)
        t.start()
        try:
            started.wait(timeout=2.0)
            with pytest.raises(ConcurrentModelOperation):
                slot.unload()
        finally:
            proceed.set()
            t.join(timeout=2.0)

    def test_loaded_id_reports_model_name_or_none(self):
        slot = ModelSlot("embedder")
        assert slot.loaded_id is None
        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        assert slot.loaded_id == "fake-embedder"
        slot.unload()
        assert slot.loaded_id is None


# ---- route tests ---------------------------------------------------------


class TestLoadRoute:
    def test_unknown_model_returns_404(self, client):
        r = client.post("/v1/models/no-such-model/load")
        assert r.status_code == 404
        body = r.json()["error"]
        assert body["code"] == "model_not_found"

    def test_load_text_embedder_success(self, client):
        r = client.post("/v1/models/fake-embedder/load")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fake-embedder"
        assert body["type"] == "embedder"
        assert body["status"] == "loaded"
        assert body["dimensions"] == 4

    def test_load_image_embedder_success(self, client):
        r = client.post("/v1/models/fake-image-embedder/load")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fake-image-embedder"
        assert body["type"] == "image_embedder"
        assert body["status"] == "loaded"
        assert body["dimensions"] == 8

    def test_load_multimodal_embedder_success(self, client):
        r = client.post("/v1/models/fake-multimodal-embedder/load")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fake-multimodal-embedder"
        assert body["type"] == "multimodal_embedder"
        assert body["status"] == "loaded"
        assert body["dimensions"] == 16

    def test_load_reranker_success(self, client):
        r = client.post("/v1/models/fake-reranker/load")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fake-reranker"
        assert body["type"] == "reranker"
        assert body["status"] == "loaded"
        assert body["dimensions"] is None

    def test_list_models_reports_loaded_for_each_family(self, client):
        """Regression: ``GET /v1/models`` must set ``loaded=True`` for
        the currently held instance of every family — including
        rerankers, whose ``dimensions`` is permanently ``null``. The
        dashboard's 已加载 / 卸载 toggle is keyed off ``loaded``, not
        ``dimensions``, so a missing or stale ``loaded`` flag would
        leave the load button stuck after a successful ``/load``.
        """
        client.post("/v1/models/fake-embedder/load")
        client.post("/v1/models/fake-image-embedder/load")
        client.post("/v1/models/fake-multimodal-embedder/load")
        client.post("/v1/models/fake-reranker/load")

        listed = client.get("/v1/models").json()["data"]
        by_id = {m["id"]: m for m in listed}
        # All four families carry ``loaded=True`` after their /load round-trip.
        assert by_id["fake-embedder"]["loaded"] is True
        assert by_id["fake-embedder"]["dimensions"] == 4
        assert by_id["fake-image-embedder"]["loaded"] is True
        assert by_id["fake-image-embedder"]["dimensions"] == 8
        assert by_id["fake-multimodal-embedder"]["loaded"] is True
        assert by_id["fake-multimodal-embedder"]["dimensions"] == 16
        # Reranker: ``dimensions`` stays ``null``, but ``loaded`` flips
        # to True — this is the only signal the dashboard has to drive
        # the 已加载 / 卸载 toggle for this family.
        assert by_id["fake-reranker"]["loaded"] is True
        assert by_id["fake-reranker"]["dimensions"] is None

    def test_list_models_reports_unloaded_for_never_loaded_ids(self, client):
        """A fresh process has no embedders loaded; every registered
        model must surface ``loaded=False`` so the dashboard shows the
        load button — not the unload button."""
        listed = client.get("/v1/models").json()["data"]
        assert listed, "expected at least one registered model"
        for m in listed:
            assert m["loaded"] is False, (
                f"{m['id']} unexpectedly reports loaded=True before any /load call"
            )

    def test_unload_clears_loaded_flag(self, client):
        """After ``/unload`` the corresponding ``GET /v1/models`` row
        must flip back to ``loaded=False`` so the dashboard re-renders
        the load button."""
        client.post("/v1/models/fake-reranker/load")
        before = next(
            m for m in client.get("/v1/models").json()["data"]
            if m["id"] == "fake-reranker"
        )
        assert before["loaded"] is True

        client.post("/v1/models/fake-reranker/unload")
        after = next(
            m for m in client.get("/v1/models").json()["data"]
            if m["id"] == "fake-reranker"
        )
        assert after["loaded"] is False

    def test_load_is_idempotent_for_same_id(self, client):
        r1 = client.post("/v1/models/fake-embedder/load")
        r2 = client.post("/v1/models/fake-embedder/load")
        assert r1.status_code == r2.status_code == 200
        # Exactly one constructor call across both requests.
        assert len(_FakeEmbedder.instances) == 1

    def test_load_failure_returns_503_and_keeps_slot_empty(self, client, monkeypatch):
        from vector_service.core.model_lifecycle import ModelSlot

        def _boom():
            raise EmbedderError("nope")

        # Patch the slot's load path for one model id by monkeypatching the
        # factory function the route picks. We do this by overriding the
        # slot directly: the route consults settings.embedding etc., so
        # the cleanest patch is to swap the registered class to one whose
        # load() always raises.
        from vector_service.embeddings import registry as emb_registry

        class _Exploder:
            dim = 0
            model_name = "fake-embedder"

            def __init__(self, settings=None):
                pass

            def load(self):
                raise EmbedderError("backend down")

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "fake-embedder", _Exploder)

        r = client.post("/v1/models/fake-embedder/load")
        assert r.status_code == 503
        body = r.json()["error"]
        assert body["code"] == "model_load_failed"
        # Slot is still empty — failure must not leave a half-loaded instance.
        slot = client.app.state._slot_embedder
        assert slot.get() is None

    def test_load_different_id_while_loaded_returns_409(self, client, monkeypatch):
        # Load the embedder first, then try to load a *different* embedder.
        client.post("/v1/models/fake-embedder/load")

        from vector_service.embeddings import registry as emb_registry

        class _Alt:
            dim = 0
            model_name = "alt-embedder"

            def __init__(self, settings=None):
                pass

            def load(self):
                pass

            def unload(self):
                pass

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "alt-embedder", _Alt)

        r = client.post("/v1/models/alt-embedder/load")
        assert r.status_code == 409
        body = r.json()["error"]
        assert body["code"] == "conflict_loaded"

    def test_load_returns_409_busy_when_lock_held(self, client):
        slot: ModelSlot = client.app.state._slot_embedder
        # Hold the slot's lock from a background thread to simulate an
        # in-flight load. The route must return 409 model_busy without
        # constructing any new instance.
        ready = threading.Event()
        release = threading.Event()

        def _hold():
            slot._lock.acquire()
            try:
                ready.set()
                release.wait(timeout=2.0)
            finally:
                slot._lock.release()

        t = threading.Thread(target=_hold)
        t.start()
        try:
            assert ready.wait(timeout=2.0)
            r = client.post("/v1/models/fake-embedder/load")
            assert r.status_code == 409
            body = r.json()["error"]
            assert body["code"] == "model_busy"
        finally:
            release.set()
            t.join(timeout=2.0)


class TestUnloadRoute:
    def test_unload_unknown_model_returns_404(self, client):
        r = client.post("/v1/models/no-such-model/unload")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_unload_embedder_success(self, client):
        # Pre-load so there is something to unload.
        client.post("/v1/models/fake-embedder/load")
        r = client.post("/v1/models/fake-embedder/unload")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "fake-embedder"
        assert body["type"] == "embedder"
        assert body["status"] == "unloaded"
        # Slot is empty after the call.
        assert client.app.state._slot_embedder.get() is None

    def test_unload_image_embedder_success(self, client):
        client.post("/v1/models/fake-image-embedder/load")
        r = client.post("/v1/models/fake-image-embedder/unload")
        assert r.status_code == 200
        assert r.json()["status"] == "unloaded"
        assert client.app.state._slot_image.get() is None

    def test_unload_reranker_success(self, client):
        client.post("/v1/models/fake-reranker/load")
        r = client.post("/v1/models/fake-reranker/unload")
        assert r.status_code == 200
        assert r.json()["status"] == "unloaded"
        assert client.app.state._slot_reranker.get() is None

    def test_unload_when_empty_returns_409_not_loaded(self, client):
        r = client.post("/v1/models/fake-embedder/unload")
        assert r.status_code == 409
        body = r.json()["error"]
        assert body["code"] == "not_loaded"

    def test_unload_returns_409_busy_when_lock_held(self, client):
        client.post("/v1/models/fake-embedder/load")
        slot: ModelSlot = client.app.state._slot_embedder
        ready = threading.Event()
        release = threading.Event()

        def _hold():
            slot._lock.acquire()
            try:
                ready.set()
                release.wait(timeout=2.0)
            finally:
                slot._lock.release()

        t = threading.Thread(target=_hold)
        t.start()
        try:
            assert ready.wait(timeout=2.0)
            r = client.post("/v1/models/fake-embedder/unload")
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "model_busy"
        finally:
            release.set()
            t.join(timeout=2.0)

    def test_unload_calls_instance_unload(self, client):
        client.post("/v1/models/fake-embedder/load")
        # The single instance the load created should receive unload().
        instance = _FakeEmbedder.instances[0]
        client.post("/v1/models/fake-embedder/unload")
        assert instance.unload_called == 1


# ---- type discrimination --------------------------------------------------


def test_lookup_prefers_embedder_over_reranker_when_id_collides(monkeypatch):
    """When the same id is registered in two registries (unusual but
    possible), the embedder wins because the lookup walks the families in
    a deterministic order. Documented behaviour."""
    from vector_service.embeddings import registry as emb_registry
    from vector_service.rerankers import registry as rerank_registry

    class _DupEmbedder:
        dim = 1
        model_name = "dup"
        def __init__(self, settings=None): pass
        def load(self): pass
        def embed_documents(self, texts): return [[0.0]]
        def embed_query(self, text): return [0.0]

    class _DupReranker:
        model_name = "dup"
        def __init__(self, settings=None): pass
        def load(self): pass
        def rerank(self, q, d, top_n=None): return []

    monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "dup", _DupEmbedder)
    monkeypatch.setitem(rerank_registry.RERANKER_REGISTRY, "dup", _DupReranker)

    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    app.include_router(models_router)
    app.add_exception_handler(HTTPException, _http_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    from vector_service.core.model_lifecycle import attach_default_slots

    attach_default_slots(app, settings=_SettingsStub())

    client = TestClient(app)
    r = client.post("/v1/models/dup/load")
    assert r.status_code == 200
    assert r.json()["type"] == "embedder"
