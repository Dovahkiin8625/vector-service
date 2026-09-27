"""Unit tests for the hot-load / hot-unload endpoints and ``ModelSlot``.

The routes live on the existing ``models`` router:

    POST /v1/models/{model_id}/load
    POST /v1/models/{model_id}/unload

Behaviour under test:

- 404 ``model_not_found`` for unknown model ids.
- Load is asynchronous: POST returns **202** ``{status: "loading"}`` at
  once and the build/load continues on a worker; clients observe the
  outcome by polling ``GET /v1/models`` — the row ends at
  ``load_status="loaded"`` (dimensions populated, mirror installed) or
  ``load_status="failed"`` with a ``load_error`` message.
- Load is idempotent for the same id: a second POST while loaded returns
  **200** ``loaded`` without rebuilding. A different id while one is
  loaded surfaces 409 ``conflict_loaded`` (caller must unload first).
- A load failure (e.g. backend raises ``EmbedderError``) leaves the slot
  empty and surfaces asynchronously via ``load_status="failed"`` — there
  is no synchronous 503 anymore.
- Unload stays synchronous: clears the slot and returns 200 ``unloaded``;
  unloading a slot that is already empty returns 409 ``not_loaded``.
- Per-family ``ModelSlot`` exposes a non-blocking lock so concurrent
  load/unload calls return 409 ``model_busy`` — including a load/unload
  arriving while a background load is in flight.
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
    # Context-managed: a persistent anyio portal/event loop across
    # requests, so a background load task spawned by POST survives until
    # the polling GETs observe it. A bare ``TestClient(app)`` tears the
    # loop down after every request and cancels the task.
    with TestClient(app) as client:
        yield client


class _SettingsStub:
    """Stand-in for ``Settings`` exposing the four settings blocks."""

    def __init__(self):
        self.embedding = _DummySettings()
        self.image_embedding = _DummySettings()
        self.multimodal_embedding = _DummySettings()
        self.reranker = _DummySettings()


class _DummySettings:
    pass


# ---- async-load polling helper -------------------------------------------


def _wait_row(client, model_id, target, *, timeout=5.0):
    """Poll GET /v1/models until ``model_id``'s ``load_status`` reaches
    ``target`` (or the opposite terminal state), returning the row.

    The background load runs on the persistent TestClient event loop in
    an anyio portal thread, so it makes progress while this test thread
    sleeps.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        rows = client.get("/v1/models").json()["data"]
        last = next((m for m in rows if m["id"] == model_id), None)
        assert last is not None, f"{model_id} vanished from GET /v1/models"
        if last["load_status"] == target:
            return last
        if last["load_status"] in ("loaded", "failed"):
            raise AssertionError(
                f"{model_id} settled at {last['load_status']} "
                f"(wanted {target}); error={last.get('load_error')!r}"
            )
        time.sleep(0.01)
    raise AssertionError(
        f"timed out waiting for {model_id} load_status={target}; last={last}"
    )


def _post_load(client, model_id):
    """POST /load and assert the immediate 202 ``loading`` envelope."""
    r = client.post(f"/v1/models/{model_id}/load")
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["id"] == model_id
    assert body["status"] == "loading"
    assert body["dimensions"] is None
    return body


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
        # The failed load is observable via the state fields.
        assert slot.load_state == "failed"
        assert slot.loading_id == "exploder"
        assert slot.load_error == "backend init failed"
        assert slot.load_error_type == "EmbedderError"
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

    # ---- two-phase begin_load / finish_load ---------------------------

    def test_begin_finish_load_drives_loading_to_loaded(self):
        slot = ModelSlot("embedder")
        assert slot.load_state == "unloaded"
        # begin: lock is held and the slot advertises loading.
        assert slot.begin_load(_FakeEmbedder) is None
        assert slot.load_state == "loading"
        assert slot.loading_id == "fake-embedder"
        # While begin holds the lock, a concurrent op is rejected.
        with pytest.raises(ConcurrentModelOperation):
            slot.unload()
        # finish: builds, installs, releases the lock.
        out = slot.finish_load(lambda: _FakeEmbedder())
        assert isinstance(out, _FakeEmbedder)
        assert slot.get() is out
        assert slot.load_state == "loaded"
        assert slot.loaded_id == "fake-embedder"
        # Lock was released — a subsequent unload succeeds.
        assert slot.unload() is True

    def test_begin_load_idempotent_returns_existing_and_releases_lock(self):
        slot = ModelSlot("embedder")
        held = slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        again = slot.begin_load(_FakeEmbedder)
        assert again is held
        assert slot.load_state == "loaded"
        # Lock must have been released — no spurious busy afterwards.
        assert slot.begin_load(_FakeEmbedder) is held

    def test_begin_load_conflict_releases_lock(self):
        slot = ModelSlot("embedder")
        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())

        class _OtherEmbedder(_FakeEmbedder):
            model_name = "other-embedder"

        with pytest.raises(DifferentModelLoaded):
            slot.begin_load(_OtherEmbedder)
        # Lock released: the existing instance stays usable and can be
        # unloaded (would raise ConcurrentModelOperation if still held).
        assert slot.loaded_id == "fake-embedder"
        assert slot.unload() is True

    def test_finish_load_failure_marks_failed_and_releases_lock(self):
        slot = ModelSlot("embedder")

        class _Exploder:
            dim = 0
            model_name = "exploder"

            def __init__(self, settings=None):
                self._impl = "partial"

            def load(self):
                raise EmbedderError("worker boom")

            def unload(self):
                self._impl = None

        assert slot.begin_load(_Exploder) is None
        with pytest.raises(EmbedderError):
            slot.finish_load(lambda: _Exploder())
        assert slot.get() is None
        assert slot.load_state == "failed"
        assert slot.loading_id == "exploder"
        assert slot.load_error == "worker boom"
        # Lock released: a retry can begin a new load.
        assert slot.begin_load(_FakeEmbedder) is None
        assert slot.finish_load(lambda: _FakeEmbedder()).load_called == 1

    def test_set_instance_marks_state_loaded(self):
        slot = ModelSlot("image_embedder")
        inst = _FakeImageEmbedder()
        slot.set_instance(inst)
        assert slot.get() is inst
        assert slot.load_state == "loaded"
        assert slot.loading_id == "fake-image-embedder"
        assert slot.load_error is None

    def test_unload_resets_load_state(self):
        slot = ModelSlot("embedder")
        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        slot.unload()
        assert slot.load_state == "unloaded"
        assert slot.loading_id is None
        assert slot.load_error is None

    def test_load_records_duration_and_unload_clears_it(self):
        slot = ModelSlot("embedder")
        assert slot.load_duration is None
        slot.load(_FakeEmbedder, lambda: _FakeEmbedder())
        duration = slot.load_duration
        assert duration is not None
        assert duration >= 0.0
        slot.unload()
        assert slot.load_duration is None

    def test_finish_load_records_duration(self):
        slot = ModelSlot("embedder")
        slot.begin_load(_FakeEmbedder)
        slot.finish_load(lambda: _FakeEmbedder())
        assert slot.load_duration is not None
        assert slot.load_duration >= 0.0

    def test_failed_finish_load_leaves_duration_unset(self):
        slot = ModelSlot("embedder")

        class _Exploder(_FakeEmbedder):
            model_name = "exploder-duration"

            def load(self):
                raise EmbedderError("boom")

        slot.begin_load(_Exploder)
        with pytest.raises(EmbedderError):
            slot.finish_load(lambda: _Exploder())
        assert slot.load_duration is None

    def test_set_instance_accepts_optional_duration(self):
        slot = ModelSlot("image_embedder")
        inst = _FakeImageEmbedder()
        # Default: unknown (lifespan did not measure it).
        slot.set_instance(inst)
        assert slot.load_duration is None
        slot.set_instance(inst, load_duration=3.25)
        assert slot.load_duration == 3.25


# ---- route tests ---------------------------------------------------------


class TestLoadRoute:
    def test_unknown_model_returns_404(self, client):
        r = client.post("/v1/models/no-such-model/load")
        assert r.status_code == 404
        body = r.json()["error"]
        assert body["code"] == "model_not_found"

    def test_load_text_embedder_success(self, client):
        _post_load(client, "fake-embedder")
        row = _wait_row(client, "fake-embedder", "loaded")
        assert row["type"] == "embedder"
        assert row["dimensions"] == 4
        # The inference-side mirror flips only once the worker settles.
        assert client.app.state.embedder is not None
        assert client.app.state.embedder.model_name == "fake-embedder"

    def test_loaded_row_carries_model_info(self, client):
        _post_load(client, "fake-embedder")
        row = _wait_row(client, "fake-embedder", "loaded")
        info = row["model_info"]
        # The fakes hold no torch.nn.Module, so the resource fields are
        # null — but the slot still reports how long the load took.
        assert info is not None
        assert info["param_count"] is None
        assert info["memory_bytes"] is None
        assert info["device"] is None
        assert isinstance(info["load_duration_seconds"], (int, float))

        # Single-model lookup carries the same block.
        single = client.get("/v1/models/fake-embedder").json()
        assert single["model_info"] is not None
        assert single["model_info"]["load_duration_seconds"] is not None

        # Registered-but-unloaded rows expose no info block.
        rows = client.get("/v1/models").json()["data"]
        by_id = {m["id"]: m for m in rows}
        assert by_id["fake-image-embedder"]["model_info"] is None
        assert by_id["fake-reranker"]["model_info"] is None

    def test_load_image_embedder_success(self, client):
        _post_load(client, "fake-image-embedder")
        row = _wait_row(client, "fake-image-embedder", "loaded")
        assert row["type"] == "image_embedder"
        assert row["dimensions"] == 8
        assert client.app.state.image_embedder.model_name == "fake-image-embedder"

    def test_load_multimodal_embedder_success(self, client):
        _post_load(client, "fake-multimodal-embedder")
        row = _wait_row(client, "fake-multimodal-embedder", "loaded")
        assert row["type"] == "multimodal_embedder"
        assert row["dimensions"] == 16
        assert (
            client.app.state.multimodal_embedder.model_name
            == "fake-multimodal-embedder"
        )

    def test_load_reranker_success(self, client):
        _post_load(client, "fake-reranker")
        row = _wait_row(client, "fake-reranker", "loaded")
        assert row["type"] == "reranker"
        assert row["dimensions"] is None
        assert client.app.state.reranker.model_name == "fake-reranker"

    def test_list_models_reports_loaded_for_each_family(self, client):
        """Regression: ``GET /v1/models`` must set ``loaded=True`` for
        the currently held instance of every family — including
        rerankers, whose ``dimensions`` is permanently ``null``. The
        dashboard's 已加载 / 卸载 toggle is keyed off ``loaded``, not
        ``dimensions``, so a missing or stale ``loaded`` flag would
        leave the load button stuck after a successful ``/load``.
        """
        for mid in (
            "fake-embedder",
            "fake-image-embedder",
            "fake-multimodal-embedder",
            "fake-reranker",
        ):
            client.post(f"/v1/models/{mid}/load")
            _wait_row(client, mid, "loaded")

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
            assert m["load_status"] == "unloaded", m
            assert m["load_error"] is None, m

    def test_unload_clears_loaded_flag(self, client):
        """After ``/unload`` the corresponding ``GET /v1/models`` row
        must flip back to ``loaded=False`` so the dashboard re-renders
        the load button."""
        client.post("/v1/models/fake-reranker/load")
        _wait_row(client, "fake-reranker", "loaded")
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
        assert after["load_status"] == "unloaded"

    def test_load_is_idempotent_for_same_id(self, client):
        r1 = client.post("/v1/models/fake-embedder/load")
        # First call starts the background load: 202.
        assert r1.status_code == 202
        _wait_row(client, "fake-embedder", "loaded")
        # Second call while the id is held: synchronous 200, no rebuild.
        r2 = client.post("/v1/models/fake-embedder/load")
        assert r2.status_code == 200
        assert r2.json()["status"] == "loaded"
        assert r2.json()["dimensions"] == 4
        # Exactly one constructor call across both requests.
        assert len(_FakeEmbedder.instances) == 1

    def test_load_failure_surfaces_via_poll_and_keeps_slot_empty(self, client, monkeypatch):
        # Swap the registered class for one whose load() always raises.
        # The request itself is accepted (202); the failure surfaces
        # asynchronously on GET /v1/models (load_status=failed) rather
        # than as a synchronous 503.
        from vector_service.embeddings import registry as emb_registry

        class _Exploder:
            dim = 0
            model_name = "fake-embedder"

            def __init__(self, settings=None):
                pass

            def load(self):
                raise EmbedderError("backend down")

            def unload(self):
                pass

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "fake-embedder", _Exploder)

        assert client.post("/v1/models/fake-embedder/load").status_code == 202
        row = _wait_row(client, "fake-embedder", "failed")
        assert row["loaded"] is False
        assert row["load_error"] == "backend down"
        # Slot is still empty and the inference mirror unset — failure
        # must not leave a half-loaded instance.
        slot = client.app.state._slot_embedder
        assert slot.get() is None
        assert slot.load_error_type == "EmbedderError"
        # No inference-side mirror installed (the no-lifespan test app
        # never creates the attribute at all).
        assert getattr(client.app.state, "embedder", None) is None

    def test_loading_state_only_targets_requested_id(self, client, monkeypatch):
        """Immediately after POST, the target id advertises ``loading``
        while every other registered id (including same-family ones)
        stays ``unloaded``."""
        from vector_service.embeddings import registry as emb_registry

        proceed = threading.Event()

        class _SlowEmbedder:
            dim = 4
            model_name = "slow-embedder"

            def __init__(self, settings=None):
                pass

            def load(self):
                proceed.wait(timeout=5.0)

            def unload(self):
                pass

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "slow-embedder", _SlowEmbedder)
        try:
            assert client.post("/v1/models/slow-embedder/load").status_code == 202
            by_id = {
                m["id"]: m
                for m in client.get("/v1/models").json()["data"]
            }
            assert by_id["slow-embedder"]["load_status"] == "loading"
            assert by_id["slow-embedder"]["loaded"] is False
            # Another registered embedder id is NOT dragged into loading.
            assert by_id["fake-embedder"]["load_status"] == "unloaded"

            # Unload while loading: lock held -> 409 model_busy.
            busy = client.post("/v1/models/slow-embedder/unload")
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "model_busy"

            proceed.set()
            row = _wait_row(client, "slow-embedder", "loaded")
            assert row["dimensions"] == 4
        finally:
            proceed.set()

    def test_failed_load_can_be_retried(self, client, monkeypatch):
        """After a failed background load the slot is unlocked; swapping
        in a healthy class and POSTing again loads normally."""
        from vector_service.embeddings import registry as emb_registry

        class _Exploder:
            dim = 0
            model_name = "fake-embedder"

            def __init__(self, settings=None):
                pass

            def load(self):
                raise EmbedderError("backend down")

            def unload(self):
                pass

            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "fake-embedder", _Exploder)
        client.post("/v1/models/fake-embedder/load")
        _wait_row(client, "fake-embedder", "failed")

        monkeypatch.setitem(emb_registry.EMBEDDER_REGISTRY, "fake-embedder", _FakeEmbedder)
        client.post("/v1/models/fake-embedder/load")
        row = _wait_row(client, "fake-embedder", "loaded")
        assert row["dimensions"] == 4
        assert row["load_error"] is None

    def test_load_different_id_while_loaded_returns_409(self, client, monkeypatch):
        # Load the embedder first, then try to load a *different* embedder.
        client.post("/v1/models/fake-embedder/load")
        _wait_row(client, "fake-embedder", "loaded")

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
        _wait_row(client, "fake-embedder", "loaded")
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
        _wait_row(client, "fake-image-embedder", "loaded")
        r = client.post("/v1/models/fake-image-embedder/unload")
        assert r.status_code == 200
        assert r.json()["status"] == "unloaded"
        assert client.app.state._slot_image.get() is None

    def test_unload_reranker_success(self, client):
        client.post("/v1/models/fake-reranker/load")
        _wait_row(client, "fake-reranker", "loaded")
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
        _wait_row(client, "fake-embedder", "loaded")
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
        _wait_row(client, "fake-embedder", "loaded")
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

    with TestClient(app) as client:
        assert client.post("/v1/models/dup/load").status_code == 202
        row = _wait_row(client, "dup", "loaded")
        assert row["type"] == "embedder"
