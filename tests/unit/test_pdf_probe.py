"""Unit tests for the PDF text-layer probe.

Builds real PDFs in memory with PyMuPDF: digital pages carry inserted
text, "scanned" pages carry only a raster image. No Docling models are
involved — the probe is pure native text extraction.
"""
from __future__ import annotations

import fitz
import pytest

from vector_service.parsers.pdf_probe import (
    DIGITAL,
    MIXED,
    SCANNED,
    probe_pdf_bytes,
)


def _digital_pdf(pages: int = 1) -> bytes:
    doc = fitz.open()
    text = "The quick brown fox jumps over the lazy dog. " * 4
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontname="helv", fontsize=12)
    return doc.tobytes()


def _scanned_pdf(pages: int = 1) -> bytes:
    # A page containing only a full-page raster image, no text layer.
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, page.rect)
        pix.clear_with(128)
        page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def _mixed_pdf() -> bytes:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "The quick brown fox jumps over the lazy dog. " * 4,
        fontname="helv",
        fontsize=12,
    )
    page2 = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, page2.rect)
    pix.clear_with(200)
    page2.insert_image(page2.rect, pixmap=pix)
    return doc.tobytes()


def test_digital_pdf_classified_digital():
    probe = probe_pdf_bytes(_digital_pdf(pages=3))
    assert probe is not None
    assert probe.kind == DIGITAL
    assert probe.page_count == 3
    assert probe.scanned_count == 0


def test_scanned_pdf_classified_scanned():
    probe = probe_pdf_bytes(_scanned_pdf(pages=2))
    assert probe is not None
    assert probe.kind == SCANNED
    assert probe.digital_pages == []
    assert probe.scanned_pages == [1, 2]


def test_mixed_pdf_classified_mixed():
    probe = probe_pdf_bytes(_mixed_pdf())
    assert probe is not None
    assert probe.kind == MIXED
    assert probe.digital_pages == [1]
    assert probe.scanned_pages == [2]


def test_short_cover_page_does_not_flip_document_to_mixed():
    # 10 digital pages + 1 near-empty cover: still digital at ratio 0.9.
    data = _digital_pdf(pages=10)
    doc = fitz.open(stream=data, filetype="pdf")
    cover = doc.new_page(pno=0)
    cover.insert_text((72, 72), "Hi", fontname="helv", fontsize=12)
    probe = probe_pdf_bytes(doc.tobytes())
    assert probe is not None
    assert probe.kind == DIGITAL


def test_garbage_bytes_return_none():
    assert probe_pdf_bytes(b"not a pdf at all") is None


def test_empty_bytes_return_none():
    assert probe_pdf_bytes(b"") is None


def test_metadata_shape():
    probe = probe_pdf_bytes(_scanned_pdf())
    meta = probe.to_metadata()
    assert meta == {
        "doc_kind": SCANNED,
        "page_count": 1,
        "scanned_pages": 1,
    }
