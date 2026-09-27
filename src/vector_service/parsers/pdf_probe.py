"""Cheap PDF text-layer probe — decide whether a PDF is digital or scanned.

Docling's Standard pipeline with ``OcrMode.DEFAULT`` already does
*selective* OCR per layout region, so the probe never gates OCR — its
jobs are surfacing ``doc_kind`` in the parse metadata and driving
profile auto-selection (a confirmed-digital PDF picks the ``native``
backend; scanned/mixed/unknown PDFs stay on ``standard``).

The probe runs through PyMuPDF (``fitz``, already a core dependency)
directly on the uploaded bytes. Text extraction is native code,
sub-millisecond per page, and pays no Docling model-startup cost.

Classification heuristic
-------------------------

For each page, extract the text layer, strip it, and count
non-whitespace characters. A page with at least ``min_text_chars``
characters is voted *digital*; below it *scanned* (an image-only page,
a scan, a blank page, or a cover). The document kind is decided by the
page votes:

- ``scanned`` — no page carries a usable text layer;
- ``mixed``   — some scanned pages inside an otherwise digital document;
- ``digital`` — essentially every page is digital (a minority of short
  cover/blank pages is tolerated via ``digital_ratio``).

Encrypted PDFs that PyMuPDF cannot open and malformed inputs return
``None`` ("unknown"): callers must then fall back to the conservative
Standard pipeline rather than trusting the probe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Document-level classifications. ``unknown`` is represented by
#: ``None`` (probe could not run) rather than a fifth enum value, so
#: every non-None result is a decision callers can switch on.
DIGITAL = "digital"
SCANNED = "scanned"
MIXED = "mixed"

#: Default minimum non-whitespace characters for a page to count as
#: "this page carries a real text layer". A normal text page has
#: hundreds to thousands of characters; a scanned page has zero.
DEFAULT_MIN_TEXT_CHARS = 50

#: Fraction of pages that must be digital for the whole document to be
#: classified ``digital``. A title page + blank back cover (2 short
#: pages out of 20) must not flip a normal report into ``mixed``.
DEFAULT_DIGITAL_RATIO = 0.9


@dataclass
class PdfProbe:
    """Outcome of probing one PDF."""

    page_count: int
    digital_pages: list[int] = field(default_factory=list)
    """1-based page numbers whose text layer looks real."""
    scanned_pages: list[int] = field(default_factory=list)
    """1-based page numbers with no usable text layer."""
    kind: str = DIGITAL

    @property
    def scanned_count(self) -> int:
        return len(self.scanned_pages)

    def to_metadata(self) -> dict:
        """Compact form for :class:`ParsedDocument.metadata`."""
        return {
            "doc_kind": self.kind,
            "page_count": self.page_count,
            "scanned_pages": self.scanned_count,
        }


def probe_pdf_bytes(
    data: bytes,
    *,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    digital_ratio: float = DEFAULT_DIGITAL_RATIO,
) -> PdfProbe | None:
    """Classify an in-memory PDF from its text layer.

    Returns ``None`` when PyMuPDF cannot read the file (not a PDF,
    truncated, or password-protected with no empty-password unlock) —
    the caller should treat that as "unknown" and use the conservative
    pipeline.
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover — pymupdf is a core dep
        return None

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return None

    try:
        if doc.needs_pass:
            # Try the empty password: many "encrypted" PDFs only carry an
            # owner password and open without user input.
            if not doc.authenticate(""):
                return None
        if doc.page_count == 0:
            return None

        digital: list[int] = []
        scanned: list[int] = []
        for index, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception:
                # One unreadable page must not sink the whole probe;
                # treat it conservatively as scanned.
                text = ""
            chars = sum(1 for ch in text if not ch.isspace())
            (digital if chars >= min_text_chars else scanned).append(index)

        page_count = doc.page_count
    except Exception:
        return None
    finally:
        doc.close()

    if not digital:
        kind = SCANNED
    elif len(digital) / page_count >= digital_ratio:
        kind = DIGITAL
    else:
        kind = MIXED

    return PdfProbe(
        page_count=page_count,
        digital_pages=digital,
        scanned_pages=scanned,
        kind=kind,
    )


def probe_pdf_path(
    path: Path,
    *,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    digital_ratio: float = DEFAULT_DIGITAL_RATIO,
) -> PdfProbe | None:
    """Classify a PDF already on disk; convenience wrapper.

    Returns ``None`` when the file cannot be read or is not a usable
    PDF (see :func:`probe_pdf_bytes`).
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return probe_pdf_bytes(
        data, min_text_chars=min_text_chars, digital_ratio=digital_ratio
    )
