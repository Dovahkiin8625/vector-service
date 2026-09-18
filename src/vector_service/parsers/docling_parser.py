"""Docling-backed parser for PDF / DOCX / PPTX / HTML.

Mirrors the lazy-load singleton pattern from
:mod:`vector_service.embeddings.bge_m3`: a single
:class:`DocumentConverter` is constructed on the first call to
:meth:`DoclingParser.parse` / :meth:`parse_bytes` and cached on the
instance. Docling model downloads are heavy on first use; keeping the
singleton on the parser lets the lifespan step warm it up explicitly
(via ``VS_PARSER__AUTO_LOAD=true``) and keeps concurrent FastAPI
workers from racing each other to instantiate the converter.

Docling is an optional dependency — if it is not installed the parser
still imports cleanly but raises :class:`ParserUnavailable` at parse
time. That keeps the rest of the route layer usable on hosts that
have not opted into binary document parsing.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vector_service.core.errors import VectorServiceError
from vector_service.parsers.base import DocumentParser, ParsedDocument

if TYPE_CHECKING:  # pragma: no cover — import only for typing
    from docling.document_converter import DocumentConverter


class ParserUnavailable(VectorServiceError):
    """The Docling backend is not available (not installed or not loaded)."""


class DoclingParser(DocumentParser):
    """DocumentConverter-based parser.

    Accepts the binary formats that Lumos currently ingests — PDF,
    DOCX, PPTX, and HTML. Markdown and plain text are handled by
    :class:`vector_service.parsers.markdown_parser.MarkdownParser`
    instead (they don't need Docling's layout model).
    """

    accepted_mime: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html",
        "application/xhtml+xml",
    )

    #: Map from MIME type to Docling's ``InputFormat`` value. Pulled
    #: from ``docling.document_converter.FormattingOption`` so we
    #: don't hardcode the exact string twice.
    _MIME_TO_INPUT: dict[str, str] = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "text/html": "html",
        "application/xhtml+xml": "html",
    }

    def __init__(self) -> None:
        self._converter: DocumentConverter | None = None
        self._lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------

    def load(self) -> None:
        """Eagerly instantiate the ``DocumentConverter``.

        Idempotent and 409-tolerant — calling ``load`` twice is a
        no-op. Idempotent loading is required so the lifespan handler
        can call this on every startup (cold + warm starts alike)
        without surfacing spurious errors.
        """
        with self._lock:
            if self._converter is not None:
                return
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as e:
                raise ParserUnavailable(
                    "docling is not installed; install it to enable binary "
                    "document parsing (pip install 'docling>=2.0')"
                ) from e
            self._converter = DocumentConverter()

    def unload(self) -> None:
        """Release the cached converter. Idempotent."""
        with self._lock:
            self._converter = None

    def _ensure_loaded(self) -> "DocumentConverter":
        if self._converter is None:
            self.load()
        assert self._converter is not None  # for type checkers
        return self._converter

    # ---- parsing ------------------------------------------------------

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a file on disk via Docling's ``DocumentConverter``.

        The conversion is CPU/GPU heavy — call sites should dispatch to
        a thread executor (the route does this via
        ``loop.run_in_executor``).
        """
        converter = self._ensure_loaded()
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        try:
            result = converter.convert(str(path))
        except ParserUnavailable:
            raise
        except Exception as e:
            raise RuntimeError(f"docling conversion failed for {path}: {e}") from e
        return _result_to_parsed(result, mime=guess_mime(path))

    async def parse_bytes(self, data: bytes, mime: str) -> ParsedDocument:
        """Parse in-memory bytes via a temporary file.

        Docling's converter is path-oriented; the smallest integration
        surface is to spill ``data`` to a tempfile and let Docling read
        it back. The temp file is cleaned up via ``tempfile``'s context
        manager semantics.
        """
        loop = asyncio.get_running_loop()

        def _work() -> ParsedDocument:
            import tempfile
            import os

            converter = self._ensure_loaded()
            suffix = _suffix_for_mime(mime)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="vs_ingest_",
            ) as fh:
                fh.write(data)
                tmp_path = fh.name
            try:
                result = converter.convert(tmp_path)
            except ParserUnavailable:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"docling conversion failed for in-memory {mime}: {e}"
                ) from e
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return _result_to_parsed(result, mime=mime)

        return await loop.run_in_executor(None, _work)


def _result_to_parsed(result: Any, *, mime: str) -> ParsedDocument:
    """Translate a Docling ``ConversionResult`` to :class:`ParsedDocument`.

    Docling exposes ``.document.export_to_markdown()`` for the textual
    representation and a ``.document.pages`` collection for page
    counting. We also pull a couple of common provenance fields
    (``title``, ``author``) from the parsed document when Docling
    manages to extract them.
    """
    try:
        markdown = result.document.export_to_markdown()
    except Exception as e:  # pragma: no cover — defensive
        raise RuntimeError(f"docling produced no markdown: {e}") from e

    metadata: dict[str, Any] = {"mime_type": mime}

    # Page count: Docling stores per-page artifacts under .pages; the
    # length of that collection is the page count.
    pages = getattr(result.document, "pages", None)
    if pages is not None:
        try:
            metadata["page_count"] = len(pages)
        except TypeError:
            metadata["page_count"] = None
    else:
        metadata["page_count"] = None

    # Best-effort provenance — Docling puts these on the document's
    # ``props`` when available. Guarded so a Docling version without
    # those attributes still returns a valid ParsedDocument.
    props = getattr(result.document, "props", None)
    if props is not None:
        for key in ("title", "author"):
            value = getattr(props, key, None)
            if value:
                metadata[key] = str(value)

    return ParsedDocument(markdown=markdown or "", metadata=metadata)


def guess_mime(path: Path) -> str:
    """Best-effort MIME guess from a file extension.

    Returns ``"application/octet-stream"`` for unknown extensions —
    callers that need a strict MIME should look at the multipart
    upload's ``content_type`` instead.
    """
    suffix = path.suffix.lower()
    return _EXT_TO_MIME.get(suffix, "application/octet-stream")


def _suffix_for_mime(mime: str) -> str:
    """Reverse mapping used to spill bytes to a tempfile."""
    inv = {
        v: k for k, v in _EXT_TO_MIME.items() if not isinstance(k, tuple)
    }
    if mime in _EXT_TO_MIME_BY_MIME:
        return _EXT_TO_MIME_BY_MIME[mime]
    return inv.get(mime, ".bin")


_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


_EXT_TO_MIME_BY_MIME: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/html": ".html",
    "application/xhtml+xml": ".xhtml",
    "text/markdown": ".md",
    "text/plain": ".txt",
}
