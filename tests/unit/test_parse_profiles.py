"""Unit tests for parse profiles (auto/standard/native/vlm).

Two layers, both without real model loads:

- profile plumbing inside ``DoclingParser`` / ``_build_format_options``
  (option objects construct cheaply; pipelines only build on convert);
- the ``profile`` form field of /v1/parse (classic and stream routes):
  unknown values → 400 ``invalid_profile``, the chosen profile is
  forwarded to the parser, and raster image MIMEs are accepted uploads.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api import parse as parse_mod
from vector_service.api.parse import router as parse_router
from vector_service.parsers.base import ParsedDocument


# ---- parser-level profile plumbing ------------------------------------


def test_resolve_profile_explicit_profiles():
    from vector_service.parsers.docling_parser import DoclingParser

    assert DoclingParser._resolve_profile("standard") == "standard"
    assert DoclingParser._resolve_profile("native") == "native"
    assert DoclingParser._resolve_profile("vlm") == "vlm"
    # No document context: auto lands conservatively on standard.
    assert DoclingParser._resolve_profile("auto") == "standard"
    with pytest.raises(ValueError, match="unknown parse profile"):
        DoclingParser._resolve_profile("turbo")


def test_auto_profile_picks_by_document_type():
    from vector_service.parsers.docling_parser import DoclingParser
    from vector_service.parsers.pdf_probe import (
        DIGITAL,
        MIXED,
        SCANNED,
        PdfProbe,
    )

    resolve = DoclingParser._resolve_profile

    def probe(kind: str) -> PdfProbe:
        return PdfProbe(page_count=4, kind=kind)

    # A confirmed text layer -> the fast model-free native backend;
    # anything uncertain stays standard (selective OCR repairs it).
    assert resolve("auto", "application/pdf", probe(DIGITAL)) == "native"
    assert resolve("auto", "application/pdf", probe(SCANNED)) == "standard"
    assert resolve("auto", "application/pdf", probe(MIXED)) == "standard"
    assert resolve("auto", "application/pdf", None) == "standard"

    # Raster images always need layout + OCR.
    for mime in (
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/webp",
        "image/bmp",
    ):
        assert resolve("auto", mime) == "standard"

    # Office formats reuse the warm standard converter (Docling routes
    # them through its model-free SimplePipeline underneath).
    assert (
        resolve(
            "auto",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        == "standard"
    )


def test_build_format_options_per_profile():
    from docling.datamodel.base_models import InputFormat
    from vector_service.parsers.docling_parser import _build_format_options

    for profile in ("standard", "native", "vlm"):
        options = _build_format_options(profile)
        pdf_opt = options[InputFormat.PDF]
        img_opt = options[InputFormat.IMAGE]
        # The three profile names map to three distinct PDF pipeline
        # classes; both PDF and IMAGE are configured for every profile
        # (native stays PDF-only at the pipeline level).
        assert (
            pdf_opt.pipeline_cls.__name__
            == {
                "standard": "StandardPdfPipeline",
                "native": "NativePdfPipeline",
                "vlm": "VlmPipeline",
            }[profile]
        )
        assert (
            img_opt.pipeline_cls.__name__
            == {
                "standard": "StandardPdfPipeline",
                "native": "StandardPdfPipeline",
                "vlm": "VlmPipeline",
            }[profile]
        )


def test_profile_selects_distinct_cached_converters(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    # Patch the converter class with a parameterless fake so no real
    # Docling subclass machinery runs; the parser cache key (profile)
    # is what this test asserts.
    class _C:
        def __init__(self, *a, **k):
            pass

        def convert(self, src):
            raise RuntimeError("not a real test path")

    monkeypatch.setattr("docling.document_converter.DocumentConverter", _C)

    parser = DoclingParser()
    std1 = parser._ensure_converter("standard")
    std2 = parser._ensure_converter("auto")  # auto -> standard cache slot
    native = parser._ensure_converter("native")
    assert std1 is std2
    assert native is not std1
    assert set(parser._converters) == {"standard", "native"}


# ---- route-level: /v1/parse -------------------------------------------


class _ParserSettings:
    max_file_size_mb = 16


class _FakeSettings:
    parser = _ParserSettings()
    inference_timeout_seconds = 30.0


class _RecordingParser:
    """Remembers the kwargs each parse was called with."""

    def __init__(self):
        self.calls: list[dict] = []

    async def parse_bytes(self, data, mime, on_progress=None, **kwargs):
        self.calls.append({"mime": mime, **kwargs})
        return ParsedDocument(
            markdown="# ok\n\ntext",
            metadata={"mime_type": mime, "page_count": 1},
        )


@pytest.fixture
def parse_client(monkeypatch):
    recording = _RecordingParser()
    monkeypatch.setattr(parse_mod, "get_docling_parser", lambda: recording)
    a = FastAPI()
    a.include_router(parse_router)
    a.state.settings = _FakeSettings()

    @a.exception_handler(HTTPException)
    async def _h(_request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            return JSONResponse(
                status_code=exc.status_code, content={"error": detail["error"]}
            )
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"message": str(detail)}}
        )

    a.recording = recording
    return TestClient(a)


def test_parse_route_invalid_profile_returns_400(parse_client):
    r = parse_client.post(
        "/v1/parse",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        data={"profile": "turbo"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_profile"
    # Validation runs before any parse work.
    assert parse_client.app.recording.calls == []


def test_parse_stream_invalid_profile_returns_400(parse_client):
    r = parse_client.post(
        "/v1/parse/stream",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        data={"profile": "nope"},
    )
    assert r.status_code == 400
    assert "application/json" in r.headers["content-type"]
    assert r.json()["error"]["code"] == "invalid_profile"


def test_parse_route_forwards_profile(parse_client):
    r = parse_client.post(
        "/v1/parse",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        data={"profile": "vlm"},
    )
    assert r.status_code == 200
    assert parse_client.app.recording.calls == [
        {"mime": "application/pdf", "profile": "vlm"}
    ]


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("scan.jpg", "image/jpeg"),
        ("scan.jpeg", "image/jpeg"),
        ("pic.png", "image/png"),
        ("pic.tif", "image/tiff"),
        ("pic.tiff", "image/tiff"),
        ("pic.webp", "image/webp"),
        ("pic.bmp", "image/bmp"),
    ],
)
def test_parse_route_accepts_image_uploads(parse_client, filename, content_type):
    r = parse_client.post(
        "/v1/parse",
        files={"file": (filename, b"fakepixels", content_type)},
    )
    assert r.status_code == 200
    assert parse_client.app.recording.calls[0]["mime"] == content_type


def test_parse_route_accepts_image_by_extension_without_content_type(parse_client):
    r = parse_client.post(
        "/v1/parse/stream",
        files={"file": ("scan.png", b"fakepixels", "")},
    )
    assert r.status_code == 200
    assert parse_client.app.recording.calls[0]["mime"] == "image/png"
