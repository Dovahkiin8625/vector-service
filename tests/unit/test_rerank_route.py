"""Unit tests for ``/v1/rerank``."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vector_service.api.rerank import router as rerank_router
from vector_service.core.config import RerankerSettings, Settings
from vector_service.core.errors import RerankerError, RerankerNotLoaded
from vector_service.core.middleware import RequestIDMiddleware
from vector_service.main import _err
from vector_service.rerankers import cross_encoder  # noqa: F401 — registers "bge-reranker-v2-m3"
from vector_service.rerankers.base import Reranker, ScoredHit


class _FakeReranker(Reranker):
    """Stable scores: index 0 → 1.0, index 1 → 0.5, index 2 → 0.25, …"""

    model_name = "fake-reranker"

    def __init__(self, settings=None) -> None:
        self._impl: object | None = None

    def load(self) -> None:
        self._impl = "ready"

    def rerank(self, query, documents, top_n=None):
        n = len(documents) if top_n is None else min(top_n, len(documents))
        scored = sorted(
            [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))],
            key=lambda h: h.score,
            reverse=True,
        )
        return scored[:n]


def _make_app(reranker: Reranker | None, *, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="rerank-test")
    app.add_middleware(RequestIDMiddleware)
    app.include_router(rerank_router)
    app.state.settings = settings or Settings(
        reranker=RerankerSettings(backend="bge-reranker-v2-m3")
    )
    # Ensure the reranker is marked as loaded so the 503 path is exercised
    # only by tests that explicitly pass an unloaded double.
    if reranker is not None and getattr(reranker, "_impl", None) is None:
        reranker.load()
    app.state.reranker = reranker

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RerankerNotLoaded)
    async def _h(req, exc):
        return _err("reranker_not_loaded", str(exc), 503, exc=exc)

    @app.exception_handler(RerankerError)
    async def _re(req, exc):
        # RerankerNotLoaded is a subclass; FastAPI dispatches to the most
        # specific handler, so this only fires for plain RerankerError.
        return _err("reranker_error", str(exc) or "rerank failed", 503, exc=exc)

    @app.exception_handler(RequestValidationError)
    async def _v(req, exc):
        return _err("invalid_request", str(exc), 422, exc=exc)

    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def _http(req, exc):
        d = exc.detail
        if isinstance(d, dict) and "error" in d:
            inner = d["error"]
            extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
            return _err(
                inner.get("code", "error"),
                inner.get("message", str(d)),
                exc.status_code,
                extras,
            )
        return _err("error", str(d), exc.status_code)

    return app


# ---- happy path --------------------------------------------------------


def test_rerank_returns_results_sorted_by_score_desc():
    app = _make_app(_FakeReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a", "b", "c"]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "bge-reranker-v2-m3"
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [r["index"] for r in body["results"]] == [0, 1, 2]
    assert "request_id" in body and body["request_id"]


def test_rerank_top_n_truncates():
    app = _make_app(_FakeReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a", "b", "c", "d"], "top_n": 2},
        )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_rerank_uses_default_top_n():
    app = _make_app(
        _FakeReranker(),
        settings=Settings(
            reranker=RerankerSettings(
                backend="bge-reranker-v2-m3", top_n_default=2
            )
        ),
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a", "b", "c"]},
        )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_rerank_request_id_is_propagated():
    app = _make_app(_FakeReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a"]},
            headers={"X-Request-ID": "req_test_xyz"},
        )
    assert resp.json()["request_id"] == "req_test_xyz"


# ---- error paths -------------------------------------------------------


def test_rerank_404_when_model_unknown():
    app = _make_app(_FakeReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a"], "model": "nope"},
        )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_rerank_422_too_many_documents():
    settings = Settings(
        reranker=RerankerSettings(
            backend="bge-reranker-v2-m3", max_documents_per_request=2
        )
    )
    app = _make_app(_FakeReranker(), settings=settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a", "b", "c"]},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "too_many_documents"


def test_rerank_422_document_too_long():
    settings = Settings(
        reranker=RerankerSettings(
            backend="bge-reranker-v2-m3", max_chars_per_doc=4
        )
    )
    app = _make_app(_FakeReranker(), settings=settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["ok", "thisistoolong"]},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "document_too_long"


def test_rerank_422_query_too_long():
    settings = Settings(
        reranker=RerankerSettings(
            backend="bge-reranker-v2-m3", max_query_chars=2
        )
    )
    app = _make_app(_FakeReranker(), settings=settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "toolong", "documents": ["a"]},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "query_too_long"


def test_rerank_422_invalid_top_n():
    settings = Settings(
        reranker=RerankerSettings(
            backend="bge-reranker-v2-m3", max_top_n=2, top_n_default=2
        )
    )
    app = _make_app(_FakeReranker(), settings=settings)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a"], "top_n": 5},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_top_n"


def test_rerank_503_when_reranker_not_loaded():
    class _Unloaded(Reranker):
        model_name = "unloaded"

        def __init__(self):
            self._impl = None

        def load(self):
            self._impl = "ready"

        def rerank(self, q, docs, top_n=None):
            return [ScoredHit(index=0, score=1.0)]

    # NOTE: _make_app() auto-loads a None-_impl reranker; we need to
    # pass it through WITHOUT auto-loading. Build the app directly.
    app = FastAPI(title="rerank-test")
    app.add_middleware(RequestIDMiddleware)
    app.include_router(rerank_router)
    app.state.settings = Settings(
        reranker=RerankerSettings(backend="bge-reranker-v2-m3")
    )
    app.state.reranker = _Unloaded()  # never .load()'d

    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(RerankerNotLoaded)
    async def _h(req, exc):
        return _err("reranker_not_loaded", str(exc), 503, exc=exc)

    @app.exception_handler(RequestValidationError)
    async def _v(req, exc):
        return _err("invalid_request", str(exc), 422, exc=exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http(req, exc):
        d = exc.detail
        if isinstance(d, dict) and "error" in d:
            inner = d["error"]
            extras = {k: v for k, v in inner.items() if k not in ("code", "message")}
            return _err(
                inner.get("code", "error"),
                inner.get("message", str(d)),
                exc.status_code,
                extras,
            )
        return _err("error", str(d), exc.status_code)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a"]},
        )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "reranker_not_loaded"


def test_rerank_422_pydantic_rejects_empty_documents():
    app = _make_app(_FakeReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": []},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_rerank_503_reranker_error_from_defensive_wrap():
    """Unexpected exceptions from reranker.rerank() inside the executor
    are caught by the ``except Exception`` defensive wrap in
    ``api/rerank.py`` and re-raised as ``RerankerError``, which the main
    handler maps to 503 with code ``reranker_error``.
    """

    class _BoomReranker(Reranker):
        model_name = "fake-reranker"

        def __init__(self) -> None:
            self._impl = None

        def load(self) -> None:
            self._impl = "ready"

        def rerank(self, query, documents, top_n=None):
            raise RuntimeError("boom")

    app = _make_app(_BoomReranker())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["a", "b"]},
        )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "reranker_error"
