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


@dataclass
class ParsedDocument:
    """The result of parsing a single source document.

    ``markdown`` is the canonical text representation produced by the
    parser — Docling's ``DocumentConverter.export_to_markdown`` for
    binary formats, the original bytes decoded for plain-text formats.
    ``metadata`` is a free-form dict the parser populates with whatever
    it can extract (page count, title, author, mime type, ...). It is
    surfaced verbatim by the HTTP layer so downstream callers can
    decide what to persist.
    """

    markdown: str
    metadata: dict = field(default_factory=dict)


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
    ) -> ParsedDocument:
        """Parse a file on disk and return its markdown + metadata.

        ``on_progress`` is called as ``(pages_done, total_pages)`` after
        each completed page for paginated formats; text-only parsers
        never call it. ``None`` (the default) disables progress
        reporting.
        """

    @abstractmethod
    async def parse_bytes(
        self,
        data: bytes,
        mime: str,
        on_progress: ProgressCallback | None = None,
    ) -> ParsedDocument:
        """Parse in-memory bytes with the given MIME type.

        Async because the docling backend can hold the event loop on a
        long conversion; the markdown passthrough is trivial enough that
        it just returns synchronously. ``on_progress`` mirrors
        :meth:`parse` and fires from the converter's worker thread.
        """

    def can_handle(self, mime: str) -> bool:
        """``True`` if this parser claims ``mime``.

        Subclasses with a richer MIME taxonomy (e.g. ``application/*``)
        should override. The base implementation does an exact-match
        check against :attr:`accepted_mime`.
        """
        return mime in self.accepted_mime
