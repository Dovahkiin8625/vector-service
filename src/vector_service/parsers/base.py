"""Document parser abstract base class.

Concrete parsers (Docling for binary formats, markdown passthrough for
text) must implement ``parse(path)`` and ``parse_bytes(data, mime)``
returning a :class:`ParsedDocument`. The orchestrator route
(``POST /v1/parse``) dispatches on MIME type / file extension to pick
the right backend; adding a new format is a matter of registering
another subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: Per-page progress sink: ``(pages_done, total_pages)``. Fired after a
#: page has passed every pipeline stage (layout, OCR, ...); paginated
#: binary parsers emit one call per page, text parsers emit none.
ProgressCallback = Callable[[int, int], None]

#: Named parse strategies. ``auto`` picks by document type (digital
#: PDF → ``native``; scanned/mixed PDF + raster images → ``standard``;
#: office formats use Docling's model-free SimplePipeline regardless);
#: the others force one of Docling's pipelines:
#:
#: - ``standard`` — layout + TableFormer + selective RapidOCR (PDF +
#:   images; DOCX/PPTX/HTML always use Docling's model-free
#:   SimplePipeline regardless of this setting);
#: - ``native``   — model-free docling-parse extraction, digital PDFs
#:   only;
#: - ``vlm``      — end-to-end vision-language model conversion (PDF +
#:   images only).
ParseProfile = str
PROFILE_AUTO = "auto"
PROFILE_STANDARD = "standard"
PROFILE_NATIVE = "native"
PROFILE_VLM = "vlm"
PARSE_PROFILES: tuple[str, ...] = (
    PROFILE_AUTO,
    PROFILE_STANDARD,
    PROFILE_NATIVE,
    PROFILE_VLM,
)


@dataclass
class SavedImage:
    """One picture extracted from a parsed document and saved locally."""

    filename: str
    """``image_000000_<hash>.png`` — Docling's content-addressed name."""
    uri: str
    """Reference placed in the markdown, e.g.
    ``/artifacts/<stem>/images/image_000000_<hash>.png``."""
    path: Path


@dataclass
class ParsedDocument:
    """The result of parsing a single source document.

    ``markdown`` is the canonical text representation produced by the
    parser — Docling's ``DocumentConverter.export_to_markdown`` for
    binary formats, the original bytes decoded for plain-text formats.
    ``metadata`` is a free-form dict the parser populates with whatever
    it can extract (page count, title, author, mime type, doc_kind,
    profile, image count, ...). It is surfaced verbatim by the HTTP
    layer so downstream callers can decide what to persist.

    When picture extraction is enabled, ``images`` lists every image
    written under ``artifacts_dir`` and the markdown references them
    through ``uri`` (served statically by the service); both stay
    empty for text-only documents.
    """

    markdown: str
    metadata: dict = field(default_factory=dict)
    images: list[SavedImage] = field(default_factory=list)
    artifacts_dir: Path | None = None


class DocumentParser(ABC):
    """Abstract base for document parsers.

    Mirrors the lightweight contract used by ``Embedder`` and
    ``Reranker``: a tiny surface area, no concrete I/O assumptions, and
    one sync + one async entry point so callers can choose how to
    dispatch without the parser caring.
    """

    #: MIME types this parser claims. ``dispatch_mime`` uses this for
    #: route-level MIME-based selection.
    accepted_mime: tuple[str, ...] = ()

    @abstractmethod
    def parse(
        self,
        path: Path,
        on_progress: ProgressCallback | None = None,
        *,
        profile: ParseProfile = PROFILE_AUTO,
        artifact_stem: str | None = None,
    ) -> ParsedDocument:
        """Parse a file on disk and return its markdown + metadata.

        ``on_progress`` is called as ``(pages_done, total_pages)`` after
        each completed page for paginated formats; text-only parsers
        never call it. ``profile`` forces one of the named pipelines
        (``auto`` lets the parser choose). ``artifact_stem`` names the
        per-document subdirectory for extracted images (a random id when
        omitted).
        """

    @abstractmethod
    async def parse_bytes(
        self,
        data: bytes,
        mime: str,
        on_progress: ProgressCallback | None = None,
        *,
        profile: ParseProfile = PROFILE_AUTO,
        artifact_stem: str | None = None,
    ) -> ParsedDocument:
        """Parse in-memory bytes with the given MIME type.

        Async because the docling backend can hold the event loop on a
        long conversion; the markdown passthrough is trivial enough that
        it just returns synchronously. ``on_progress`` mirrors
        :meth:`parse` and fires from the converter's worker thread.
        ``profile`` / ``artifact_stem`` mirror :meth:`parse`.
        """

    def can_handle(self, mime: str) -> bool:
        """``True`` if this parser claims ``mime``.

        Subclasses with a richer MIME taxonomy (e.g. ``application/*``)
        should override. The base implementation does an exact-match
        check against :attr:`accepted_mime`.
        """
        return mime in self.accepted_mime
