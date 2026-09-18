"""Passthrough parser for markdown / plain-text.

These formats are already text — there's no layout to reconstruct, so
the parser just decodes the bytes and sets ``page_count=None``. Keeping
the passthrough behind the same ABC lets the orchestrator dispatch
purely on MIME without special-casing text formats.
"""
from __future__ import annotations

from pathlib import Path

from vector_service.parsers.base import DocumentParser, ParsedDocument


class MarkdownParser(DocumentParser):
    """No-op parser for ``text/markdown`` and ``text/plain``.

    Decodes bytes as UTF-8 (with a couple of permissive fallbacks for
    Windows / GBK content), strips a UTF-8 BOM if present, and returns
    the text verbatim with ``page_count=None`` — there's no concept of
    pages in a flat text document.
    """

    accepted_mime: tuple[str, ...] = ("text/markdown", "text/plain")

    def parse(self, path: Path) -> ParsedDocument:
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        data = path.read_bytes()
        return self._parse(data=data, mime=_mime_for_path(path))

    async def parse_bytes(self, data: bytes, mime: str) -> ParsedDocument:
        return self._parse(data=data, mime=mime)

    @staticmethod
    def _parse(*, data: bytes, mime: str) -> ParsedDocument:
        text = _decode(data)
        return ParsedDocument(
            markdown=text,
            metadata={"mime_type": mime, "page_count": None},
        )


def _decode(data: bytes) -> str:
    """Decode bytes as UTF-8 with permissive fallbacks.

    Three attempts in order: UTF-8 (most common, including a BOM),
    GBK (legacy Chinese-only files), and finally ``latin-1`` which
    never raises — a deterministic last resort so we always return a
    string. The route layer surfaces a 400 if the result is empty, so
    callers won't see a silent failure.
    """
    # Strip BOM explicitly so ``text.startswith("﻿")`` doesn't
    # trip downstream chunkers.
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("gbk")
    except UnicodeDecodeError:
        pass
    return data.decode("latin-1")


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return "text/markdown"
    return "text/plain"


# Re-export for convenience (parsers/__init__.py can flatten later if
# callers prefer).
__all__ = ["MarkdownParser"]
