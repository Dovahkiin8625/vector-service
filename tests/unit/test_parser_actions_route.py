"""Tests for the Docling cache-action routes.

- ``POST /v1/parser/warm`` — pre-build one profile's converter;
- ``POST /v1/parser/evict`` — release one profile's cached converter.

The real converter is never touched: a fake parser with ``warm`` /
``release`` methods replaces the process singleton via monkeypatch.
Also covers the DoclingParser-level ``release`` behaviour on cold
profiles, which needs no docling install.
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from vector_service.api import parse as parse_api
from vector_service.main import app


class _FakeParser:
    def __init__(self, *, warm_error: Exception | None = None):
        self.warm_calls: list[str] = []
        self.release_calls: list[str] = []
        self._warm_error = warm_error

    def warm(self, profile: str) -> str:
        self.warm_calls.append(profile)
        if self._warm_error is not None:
            raise self._warm_error
        return profile

    def release(self, profile: str) -> bool:
        self.release_calls.append(profile)
        return True


@pytest.fixture
def fake_parser(monkeypatch) -> _FakeParser:
    parser = _FakeParser()
    monkeypatch.setattr(parse_api, "get_docling_parser", lambda: parser)
    return parser


# ---------------------------------------------------------------------------
# Route level
# ---------------------------------------------------------------------------


def test_warm_returns_warm_profile(fake_parser):
    with TestClient(app) as client:
        r = client.post("/v1/parser/warm", json={"profile": "standard"})
    assert r.status_code == 200, r.text
    assert r.json() == {"profile": "standard", "warm": True}
    assert fake_parser.warm_calls == ["standard"]


def test_warm_defaults_to_standard(fake_parser):
    with TestClient(app) as client:
        r = client.post("/v1/parser/warm", json={})
    assert r.status_code == 200
    assert fake_parser.warm_calls == ["standard"]


def test_warm_rejects_unknown_profile(fake_parser):
    with TestClient(app) as client:
        r = client.post("/v1/parser/warm", json={"profile": "nope"})
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["code"] == "invalid_profile"
    assert fake_parser.warm_calls == []


def test_warm_maps_unavailable_to_503(monkeypatch):
    from vector_service.parsers.docling_parser import ParserUnavailable

    parser = _FakeParser(
        warm_error=ParserUnavailable("docling is not installed"),
    )
    monkeypatch.setattr(parse_api, "get_docling_parser", lambda: parser)
    with TestClient(app) as client:
        r = client.post("/v1/parser/warm", json={"profile": "standard"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "parser_unavailable"


def test_warm_maps_build_failure_to_500(monkeypatch):
    parser = _FakeParser(
        warm_error=RuntimeError("failed to build docling converter"),
    )
    monkeypatch.setattr(parse_api, "get_docling_parser", lambda: parser)
    with TestClient(app) as client:
        r = client.post("/v1/parser/warm", json={"profile": "vlm"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "parser_failed"


def test_evict_releases_profile(fake_parser):
    with TestClient(app) as client:
        r = client.post("/v1/parser/evict", json={"profile": "vlm"})
    assert r.status_code == 200, r.text
    assert r.json() == {"profile": "vlm", "warm": False, "released": True}
    assert fake_parser.release_calls == ["vlm"]


def test_evict_rejects_unknown_profile(fake_parser):
    with TestClient(app) as client:
        r = client.post("/v1/parser/evict", json={"profile": "bogus"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_profile"
    assert fake_parser.release_calls == []


# ---------------------------------------------------------------------------
# DoclingParser level (cold cache, no docling build needed)
# ---------------------------------------------------------------------------


def test_release_cold_profile_is_false():
    from vector_service.parsers.docling_parser import DoclingParser

    assert DoclingParser().release("standard") is False


def test_release_unknown_profile_raises():
    from vector_service.parsers.docling_parser import DoclingParser

    with pytest.raises(ValueError, match="unknown parse profile"):
        DoclingParser().release("bogus")


def test_warm_resolves_alias_standard():
    """``warm`` must return the resolved key so the route reports the
    converter it actually built."""
    from vector_service.parsers.docling_parser import DoclingParser

    # Stub _ensure_converter so no docling import / build happens; the
    # resolved key is asserted from its argument.
    seen: list[str] = []
    parser = DoclingParser()
    parser._ensure_converter = lambda profile: seen.append(profile)
    assert parser.warm("native") == "native"
    assert seen == ["native"]
