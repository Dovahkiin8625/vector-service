"""RED #1: the Embedder base must expose a `load()` method."""
from __future__ import annotations

import pytest

from vector_service.core.errors import ModelNotLoaded
from vector_service.embeddings.base import Embedder
from vector_service.rerankers.base import Reranker


def test_embedder_base_has_load_method():
    assert hasattr(Embedder, "load")
    assert callable(getattr(Embedder, "load"))


def test_embedder_base_load_is_abstract():
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


# ---- helpers ---------------------------------------------------------------


class _FakeSettings:
    """Minimal Settings stand-in — only the attrs BGEM3Embedder reads."""

    def __init__(self, model_dir):
        self.embedding_device = "cpu"
        self.embedding_batch_size = 8
        self.embedding_max_length = 64
        self.embedding_model_dir = model_dir
        self.embedding_auto_download = False
        self.embedding_download_source = "huggingface"
        self.embedding_hf_repo = "fake/repo"
        self.embedding_ms_repo = "fake/repo"


# ---- 2. BGEM3Embedder.load() is the only entry point ----------------------


def test_bge_m3_load_calls_backend_init_and_warmup(monkeypatch, tmp_path):
    from vector_service.embeddings.bge_m3 import BGEM3Embedder

    s = _FakeSettings(model_dir=tmp_path)
    embedder = BGEM3Embedder(settings=s)  # type: ignore[arg-type]

    inits: list[tuple] = []
    encodes: list[tuple] = []

    class _FakeBackend:
        def __init__(self, model_dir, device, max_length, batch_size):
            inits.append((str(model_dir), device, max_length, batch_size))

        def encode(self, texts, is_query):
            encodes.append((list(texts), is_query))
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._dir_has_model", lambda p: True
    )
    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._TorchBackend", _FakeBackend
    )

    embedder.load()

    assert len(inits) == 1
    assert inits[0][0] == str(tmp_path)

    warmup_calls = [c for c in encodes if c[0] == [""] * 5 and c[1] is False]
    assert len(warmup_calls) == 1

    encodes.clear()
    vec = embedder.embed_documents(["hello"])
    assert len(inits) == 1  # still one — no new instance
    assert vec == [[0.0] * 4]


def test_bge_m3_load_is_idempotent(monkeypatch, tmp_path):
    """Two `load()` calls must instantiate the backend exactly once."""
    from vector_service.embeddings.bge_m3 import BGEM3Embedder

    s = _FakeSettings(model_dir=tmp_path)
    embedder = BGEM3Embedder(settings=s)  # type: ignore[arg-type]

    inits: list[tuple] = []

    class _FakeBackend:
        def __init__(self, model_dir, device, max_length, batch_size):
            inits.append((str(model_dir), device))

        def encode(self, texts, is_query):
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._dir_has_model", lambda p: True
    )
    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._TorchBackend", _FakeBackend
    )

    embedder.load()
    embedder.load()
    assert len(inits) == 1


def test_bge_m3_load_propagates_model_not_loaded(monkeypatch, tmp_path):
    from vector_service.embeddings.bge_m3 import BGEM3Embedder

    s = _FakeSettings(model_dir=tmp_path)
    embedder = BGEM3Embedder(settings=s)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._dir_has_model", lambda p: True
    )

    def _boom(*a, **kw):
        raise RuntimeError("disk error")

    monkeypatch.setattr(
        "vector_service.embeddings.bge_m3._TorchBackend", _boom
    )

    with pytest.raises(ModelNotLoaded):
        embedder.load()


# ---- 3. lifespan wiring ---------------------------------------------------


class _RecordingEmbedder(Embedder):
    """Records whether `load()` ran and lets a test inject a failure.

    Mirrors the `_impl` convention used by `BGEM3Embedder` so that
    `health._embedder_loaded()` can observe load state the same way.
    """

    dim = 4
    model_name = "recording"

    def __init__(self, raise_on_load: Exception | None = None):
        self.load_called = 0
        self._raise = raise_on_load
        self._impl = None

    def load(self) -> None:
        self.load_called += 1
        if self._raise is not None:
            raise self._raise
        self._impl = object()  # sentinel — non-None = loaded

    def embed_documents(self, texts):
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text):
        return [0.0] * self.dim


class _RecordingReranker(Reranker):
    """Records whether `load()` ran; lifespan integration tests use this
    so they don't need a real cross-encoder model on disk. Mirrors the
    `_RecordingEmbedder` shape so the monkeypatch pattern is uniform."""

    model_name = "recording"

    def __init__(self, raise_on_load: Exception | None = None):
        self.load_called = 0
        self._raise = raise_on_load

    def load(self) -> None:
        self.load_called += 1
        if self._raise is not None:
            raise self._raise

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit
        return [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))]


def test_lifespan_calls_embedder_load(monkeypatch):
    """The real `lifespan()` must invoke `embedder.load()` exactly once
    during startup. We monkeypatch `build_store` + `build_embedder` to
    avoid standing up Milvus or a real model."""
    from vector_service.core import lifespan as lifespan_mod

    embedder = _RecordingEmbedder()
    fake_store = type("S", (), {
        "backend_name": "fake",
        "uri": "",
        "_ensure_connected": lambda self: None,
        "close": lambda self: None,
        "list_databases": lambda self: [],
    })()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _RecordingReranker())

    from fastapi import FastAPI
    from vector_service.api.health import router as health_router

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    from fastapi.testclient import TestClient
    with TestClient(app):
        assert embedder.load_called == 1


def test_lifespan_keeps_app_up_when_load_fails(monkeypatch):
    """If `embedder.load()` raises `ModelNotLoaded`, the process must
    still come up; `MODEL_LOADED` stays 0; /readyz reports 503."""
    from vector_service.core import lifespan as lifespan_mod
    from vector_service.core.metrics import MODEL_LOADED

    try:
        MODEL_LOADED.remove("embedder")
    except KeyError:
        pass

    embedder = _RecordingEmbedder(raise_on_load=ModelNotLoaded("boom"))
    fake_store = type("S", (), {
        "backend_name": "fake",
        "uri": "",
        "_ensure_connected": lambda self: None,
        "close": lambda self: None,
        "list_databases": lambda self: [],
    })()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _RecordingReranker())

    from fastapi import FastAPI
    from vector_service.api.health import router as health_router
    from fastapi.testclient import TestClient

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        assert embedder.load_called == 1
        # /readyz must reflect the failed load.
        r = client.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["embedder"] == "not_loaded"
        assert body["store"] == "ok"
        # MODEL_LOADED gauge must NOT have been bumped.
        sample = MODEL_LOADED.labels(kind="embedder")._value.get()  # type: ignore[attr-defined]
        assert sample == 0


def test_readyz_reports_loaded_after_successful_lifespan(monkeypatch):
    """After a successful load, /readyz reports embedder=loaded and 200."""
    from vector_service.core import lifespan as lifespan_mod

    embedder = _RecordingEmbedder()
    fake_store = type("S", (), {
        "backend_name": "fake",
        "uri": "",
        "_ensure_connected": lambda self: None,
        "close": lambda self: None,
        "list_databases": lambda self: [],
    })()

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _RecordingReranker())

    from fastapi import FastAPI
    from vector_service.api.health import router as health_router
    from fastapi.testclient import TestClient

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["embedder"] == "loaded"
        assert body["store"] == "ok"


def test_readyz_503_when_store_list_raises_even_after_load(monkeypatch):
    """A working embedder but a broken store must still yield 503, with
    embedder=loaded and store=down."""
    from vector_service.core import lifespan as lifespan_mod

    embedder = _RecordingEmbedder()

    class _BrokenStore:
        backend_name = "fake"
        uri = ""
        def _ensure_connected(self): return None
        def list_databases(self): raise RuntimeError("store offline")
        def close(self): pass

    monkeypatch.setattr(lifespan_mod, "build_embedder", lambda s: embedder)
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: _BrokenStore())
    monkeypatch.setattr(lifespan_mod, "build_reranker", lambda s: _RecordingReranker())

    from fastapi import FastAPI
    from vector_service.api.health import router as health_router
    from fastapi.testclient import TestClient

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    app.include_router(health_router)

    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "degraded"
        assert body["embedder"] == "loaded"
        assert body["store"] == "down"
