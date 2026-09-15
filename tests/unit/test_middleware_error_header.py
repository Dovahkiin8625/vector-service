"""Regression test for RequestIDMiddleware header propagation.

Pins that the ``X-Request-ID`` header is present on BOTH success and
error responses, regardless of whether the caller supplied the value
or the middleware auto-generated it. The existing envelope tests
(e.g. ``test_management_routes.test_envelope_shape``) already pin that
``request_id`` is populated in the error JSON body; this test covers
the wire-level header so a future change to ``RequestIDMiddleware``
can't silently drop the correlation id on failure paths.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from vector_service.core.logging import request_id_var
from vector_service.core.middleware import RequestIDMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/boom")
    def _boom():
        raise HTTPException(status_code=500, detail="nope")

    return app


def test_request_id_header_on_error_response():
    """Even when the route raises, the X-Request-ID response header
    must be set so clients can correlate the error.
    """
    app = _build_app()
    with TestClient(app) as client:
        resp = client.get("/boom", headers={"X-Request-ID": "req_test_error"})
    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "req_test_error"


def test_request_id_header_on_success_response():
    """Success path: existing behaviour, regression pin."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ok")
    def _ok():
        return {"request_id": request_id_var.get()}

    with TestClient(app) as client:
        resp = client.get("/ok", headers={"X-Request-ID": "req_test_ok"})
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "req_test_ok"
    assert resp.json()["request_id"] == "req_test_ok"


def test_request_id_header_auto_generated_on_error():
    """When no X-Request-ID is supplied, the middleware generates one
    and still attaches it to the error response.
    """
    app = _build_app()
    with TestClient(app) as client:
        resp = client.get("/boom")
    assert resp.status_code == 500
    rid = resp.headers.get("X-Request-ID")
    assert rid is not None
    assert rid.startswith("req_")