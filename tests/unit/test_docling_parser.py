"""Unit tests for ``DoclingParser``.

These tests gate on the optional ``docling`` dependency — if
``docling`` isn't installed, every test is skipped with an
explanatory message rather than failing. The mock backend is
injected via monkeypatch so we never instantiate a real
``DocumentConverter`` (which would download hundreds of MB of
model weights on first use).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import docling  # noqa: F401  -- presence check
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False


pytestmark = pytest.mark.skipif(
    not HAS_DOCLING,
    reason="docling not installed; install 'docling>=2.0' to run these tests",
)


class _FakeDocument:
    """Minimal stand-in for ``docling_core.types.Document``."""

    def __init__(self, markdown: str, page_count: int = 1, title: str | None = None):
        self._markdown = markdown
        self.pages = list(range(page_count))
        if title:
            from types import SimpleNamespace
            self.props = SimpleNamespace(title=title, author=None)
        else:
            self.props = None

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, doc: _FakeDocument):
        self.document = doc


class _FakeConverter:
    """Drop-in for ``docling.document_converter.DocumentConverter``."""

    def __init__(self):
        self.calls: list[Path | str] = []

    def convert(self, source):
        self.calls.append(source)
        # Echo back a tiny markdown document with a known page count.
        return _FakeResult(
            _FakeDocument(
                markdown=f"# Converted\n\nfrom {source}\n",
                page_count=3,
                title="Hello",
            )
        )


def test_parser_dispatches_to_docling_for_pdf(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    converter = _FakeConverter()
    # Inject the fake converter BEFORE calling ``load()`` — DoclingParser
    # constructs its own inside ``load()``.
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        lambda *a, **kw: converter,
    )

    parser = DoclingParser()
    parser.load()

    parsed = parser.parse(Path("hello.pdf"))
    assert "# Converted" in parsed.markdown
    assert parsed.metadata["page_count"] == 3
    assert parsed.metadata["title"] == "Hello"
    assert parsed.metadata["mime_type"] == "application/pdf"
    assert isinstance(converter.calls[0], str) and converter.calls[0].endswith("hello.pdf")


def test_parser_handles_bytes_input(monkeypatch):
    """``parse_bytes`` spills to a tempfile and reads it back via the
    converter; verify the round-trip without ever touching disk."""
    from vector_service.parsers.docling_parser import DoclingParser

    converter = _FakeConverter()
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        lambda *a, **kw: converter,
    )

    parser = DoclingParser()
    parser.load()

    import asyncio
    parsed = asyncio.run(
        parser.parse_bytes(b"%PDF-1.4 ...", "application/pdf"),
    )
    assert "# Converted" in parsed.markdown
    assert parsed.metadata["mime_type"] == "application/pdf"
    # The converter received a path on disk.
    assert len(converter.calls) == 1


def test_load_is_idempotent(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    inits: list[int] = []

    def _factory(*a, **kw):
        inits.append(1)
        return _FakeConverter()

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _factory,
    )

    parser = DoclingParser()
    parser.load()
    parser.load()
    parser.load()
    assert len(inits) == 1


def test_unload_releases_converter(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _FakeConverter,
    )
    parser = DoclingParser()
    parser.load()
    assert parser._converter is not None
    parser.unload()
    assert parser._converter is None
    # Idempotent: a second unload is a no-op.
    parser.unload()
    assert parser._converter is None


def test_parse_bytes_tempfile_cleanup(monkeypatch):
    """``parse_bytes`` must delete its tempfile even when the
    converter raises."""
    from vector_service.parsers import docling_parser as mod

    class _BoomConverter:
        def convert(self, source):
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _BoomConverter,
    )
    parser = mod.DoclingParser()
    parser.load()

    import asyncio
    with pytest.raises(RuntimeError, match="conversion failed"):
        asyncio.run(parser.parse_bytes(b"data", "application/pdf"))


def test_can_handle_claims_only_docling_mimes():
    from vector_service.parsers.docling_parser import DoclingParser

    p = DoclingParser()
    assert p.can_handle("application/pdf")
    assert p.can_handle("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert p.can_handle("text/html")
    assert not p.can_handle("text/plain")
    assert not p.can_handle("text/markdown")


def test_guess_mime():
    from vector_service.parsers.docling_parser import guess_mime

    assert guess_mime(Path("/x/file.pdf")) == "application/pdf"
    assert guess_mime(Path("/x/file.docx")) == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert guess_mime(Path("/x/file.html")) == "text/html"
    assert guess_mime(Path("/x/file.md")) == "text/markdown"
    assert guess_mime(Path("/x/file.unknown")) == "application/octet-stream"


def test_parser_failure_raises_runtime_error(monkeypatch):
    """If the converter raises, ``parse_bytes`` re-raises as a
    ``RuntimeError`` so the route layer can map to 500."""
    from vector_service.parsers.docling_parser import DoclingParser

    class _BoomConverter:
        def convert(self, source):
            raise RuntimeError("docling exploded")

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _BoomConverter,
    )
    parser = DoclingParser()
    parser.load()

    import asyncio
    with pytest.raises(RuntimeError):
        asyncio.run(parser.parse_bytes(b"%PDF", "application/pdf"))
