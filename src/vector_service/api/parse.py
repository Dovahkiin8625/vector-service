"""``POST /v1/parse`` — file → markdown + metadata.

Dispatch logic:

- If the multipart upload declares a ``content_type`` we look that up
  in :func:`_parser_for_mime` and use the matching parser directly.
- Otherwise we fall back to the file extension (PDF / DOCX / PPTX /
  HTML / MD / TXT).
- Unsupported MIME / extensions yield 415 ``unsupported_mime``.

Heavy conversions (Docling on PDFs) are dispatched to a thread
executor so the FastAPI event loop never blocks on a CPU-bound
job. The route is sync at the route-handler level — the parser's
``parse_bytes`` is already async; ``parse`` (file path) runs in
the executor.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from vector_service.core.logging import get_logger
from vector_service.parsers.base import DocumentParser
from vector_service.parsers.docling_parser import DoclingParser
from vector_service.parsers.markdown_parser import MarkdownParser
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.ingest import ParseMetadata, ParseResponse

router = APIRouter(prefix="/v1", tags=["ingest"])
log = get_logger(__name__)

# Accepted MIME types — keep in sync with the parser set. The
# orchestrator (``POST /v1/ingest``) uses the same allow-list so
# ``/v1/parse`` and ``/v1/ingest`` agree on what's a valid upload.
_SUPPORTED_MIME: tuple[str, ...] = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/html",
    "application/xhtml+xml",
    "text/markdown",
    "text/plain",
)

# Extension → MIME. Used as a fallback when the multipart upload
# omitted ``content_type`` (browsers always send one, but
# programmatic clients sometimes don't).
_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


def _parser_for_mime(mime: str) -> DocumentParser:
    """Return the parser that claims ``mime``.

    Raises ``HTTPException(415)`` if no registered parser accepts
    the MIME. The Docling parser is constructed lazily on first
    use; the markdown parser is a stateless singleton.
    """
    if mime in DoclingParser.accepted_mime:
        return DoclingParser()
    if mime in MarkdownParser.accepted_mime:
        return MarkdownParser()
    raise HTTPException(
        status_code=415,
        detail={"error": {
            "code": "unsupported_mime",
            "message": f"no parser registered for MIME {mime!r}",
            "got": mime,
            "allowed": list(_SUPPORTED_MIME),
        }},
    )


def _resolve_mime(filename: str | None, content_type: str | None) -> str:
    """Pick the MIME type to dispatch on.

    Priority: explicit ``content_type`` from the multipart upload,
    then the file extension. Returns ``"application/octet-stream"``
    when both are missing — the parser dispatch will reject that
    with a 415.
    """
    if content_type:
        # Strip parameters, e.g. ``text/html; charset=utf-8``.
        base = content_type.split(";", 1)[0].strip().lower()
        if base:
            return base
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in _EXT_TO_MIME:
            return _EXT_TO_MIME[suffix]
    return "application/octet-stream"


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Empty upload."},
        413: {"model": ErrorEnvelope, "description": "Upload exceeds max size."},
        415: {"model": ErrorEnvelope, "description": "Unsupported MIME type."},
        500: {"model": ErrorEnvelope, "description": "Parser failure."},
        503: {"model": ErrorEnvelope, "description": "Parser backend unavailable."},
    },
    summary="Parse a document to markdown",
    description=(
        "Accept a single uploaded file (multipart form, field ``file``) "
        "and return its markdown representation plus extracted "
        "metadata. Supports PDF, DOCX, PPTX, HTML, MD, and TXT."
    ),
)
async def parse_document(file: UploadFile, request: Request):
    settings = request.app.state.settings.parser
    mime = _resolve_mime(file.filename, file.content_type)

    if mime not in _SUPPORTED_MIME:
        raise HTTPException(
            status_code=415,
            detail={"error": {
                "code": "unsupported_mime",
                "message": f"unsupported MIME type {mime!r}",
                "got": mime,
                "allowed": list(_SUPPORTED_MIME),
            }},
        )

    # Read with a size guard. ``UploadFile.read`` is the simplest
    # way to enforce the cap: if the caller sent more bytes than
    # the limit, we stop reading and 413.
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": {
                "code": "file_too_large",
                "message": (
                    f"upload exceeds {settings.max_file_size_mb} MB cap"
                ),
                "max_bytes": max_bytes,
            }},
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"error": {
                "code": "empty_file",
                "message": "uploaded file is empty",
            }},
        )

    parser = _parser_for_mime(mime)

    try:
        parsed = await parser.parse_bytes(data, mime)
    except RuntimeError as e:
        # Docling's conversion failures land here.
        log.warning("parser_failed", mime=mime, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": {
                "code": "parser_failed",
                "message": str(e) or "parser failed",
                "mime": mime,
                "exception_type": type(e).__name__,
            }},
        )
    except Exception as e:
        # ``ParserUnavailable`` (Docling not installed) and any
        # other backend-level failure surface here as 503.
        log.warning("parser_unavailable", mime=mime, error=str(e))
        raise HTTPException(
            status_code=503,
            detail={"error": {
                "code": "parser_unavailable",
                "message": str(e) or "parser backend unavailable",
                "mime": mime,
                "exception_type": type(e).__name__,
            }},
        )

    metadata = ParseMetadata(**parsed.metadata)
    return ParseResponse(markdown=parsed.markdown, metadata=metadata)
