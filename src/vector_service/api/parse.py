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
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vector_service.core.logging import get_logger
from vector_service.core.threadpools import run_in_model
from vector_service.parsers.base import (
    PARSE_PROFILES,
    PROFILE_NATIVE,
    PROFILE_STANDARD,
    PROFILE_VLM,
    DocumentParser,
)
from vector_service.parsers.docling_parser import (
    DoclingParser,
    ParserUnavailable,
    get_docling_parser,
)
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
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/bmp",
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
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


def _validate_profile(profile: str) -> str:
    """Validate the ``profile`` form field; raise 400 on unknown values."""
    if profile in PARSE_PROFILES:
        return profile
    raise HTTPException(
        status_code=400,
        detail={"error": {
            "code": "invalid_profile",
            "message": (
                f"unknown parse profile {profile!r}; "
                f"expected one of {', '.join(PARSE_PROFILES)}"
            ),
            "got": profile,
            "allowed": list(PARSE_PROFILES),
        }},
    )


#: Profiles explicit warm/evict actions may name (``auto`` is a parse
#: request-time choice, not a cached converter).
_ACTION_PROFILES: tuple[str, ...] = (
    PROFILE_STANDARD,
    PROFILE_NATIVE,
    PROFILE_VLM,
)


class ParserProfileAction(BaseModel):
    """JSON body for the parser cache warm/evict routes."""

    profile: str = PROFILE_STANDARD


def _validate_action_profile(profile: str) -> str:
    """Validate the warm/evict body; raise 400 on unknown values."""
    if profile in _ACTION_PROFILES:
        return profile
    raise HTTPException(
        status_code=400,
        detail={"error": {
            "code": "invalid_profile",
            "message": (
                f"unknown parser profile {profile!r}; "
                f"expected one of {', '.join(_ACTION_PROFILES)}"
            ),
            "got": profile,
            "allowed": list(_ACTION_PROFILES),
        }},
    )


def _action_http(status_code: int, code: str, message: str) -> HTTPException:
    """Canonical error envelope for warm/evict failures."""
    return HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message}},
    )


def _parser_for_mime(mime: str) -> DocumentParser:
    """Return the parser that claims ``mime``.

    Raises ``HTTPException(415)`` if no registered parser accepts
    the MIME. The Docling parser is constructed lazily on first
    use; the markdown parser is a stateless singleton.
    """
    if mime in DoclingParser.accepted_mime:
        # Process-wide singleton: building a DocumentConverter costs
        # ~10-30s of model loading; a fresh instance per request made
        # every upload pay that cost. The lifespan warmup populates the
        # same singleton when VS_PARSER__AUTO_LOAD=true.
        return get_docling_parser()
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


_PARSE_RESPONSES = {
    400: {"model": ErrorEnvelope, "description": "Empty upload or invalid profile."},
    413: {"model": ErrorEnvelope, "description": "Upload exceeds max size."},
    415: {"model": ErrorEnvelope, "description": "Unsupported MIME type."},
    500: {"model": ErrorEnvelope, "description": "Parser failure."},
    503: {"model": ErrorEnvelope, "description": "Parser backend unavailable."},
}


async def _prepare_parse(file: UploadFile, parser_settings: Any) -> tuple[str, bytes]:
    """Validate the upload and spool it into memory.

    Shared by the classic and streaming parse routes so pre-flight
    failures (MIME / size / empty) come back as ordinary JSON envelopes
    on BOTH endpoints before any stream opens.
    """
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

    # Read with a size guard. ``UploadFile.read`` is the simplest way
    # to enforce the cap: if the caller sent more bytes than the limit,
    # we stop reading and 413.
    max_bytes = parser_settings.max_file_size_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": {
                "code": "file_too_large",
                "message": (
                    f"upload exceeds {parser_settings.max_file_size_mb} MB cap"
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
    return mime, data


def _parser_failure_http(exc: Exception, mime: str) -> HTTPException:
    """Map a parser exception to the canonical 500/503 envelope."""
    if isinstance(exc, RuntimeError):
        # Docling's conversion failures are wrapped as RuntimeError.
        log.warning("parser_failed", mime=mime, error=str(exc))
        return HTTPException(
            status_code=500,
            detail={"error": {
                "code": "parser_failed",
                "message": str(exc) or "parser failed",
                "mime": mime,
                "exception_type": type(exc).__name__,
            }},
        )
    # ``ParserUnavailable`` (Docling not installed) and any other
    # backend-level failure surface here as 503.
    log.warning("parser_unavailable", mime=mime, error=str(exc))
    return HTTPException(
        status_code=503,
        detail={"error": {
            "code": "parser_unavailable",
            "message": str(exc) or "parser backend unavailable",
            "mime": mime,
            "exception_type": type(exc).__name__,
        }},
    )


def _http_error_event(exc: HTTPException) -> dict[str, Any]:
    """Serialise an ``HTTPException`` into a stream ``error`` event."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    error = detail.get("error")
    if not isinstance(error, dict):
        error = {"code": "error", "message": str(exc.detail)}
    return {"type": "error", "status": exc.status_code, "error": error}


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses=_PARSE_RESPONSES,
    summary="Parse a document to markdown",
    description=(
        "Accept a single uploaded file (multipart form, field ``file``) "
        "and return its markdown representation plus extracted "
        "metadata. Supports PDF, DOCX, PPTX, HTML, JPG/PNG/TIFF/WEBP/BMP "
        "images, MD, and TXT.\n\n"
        "Optional form field ``profile`` selects the Docling pipeline: "
        "``auto`` (default — picks by document type: digital PDFs use "
        "the fast model-free native extraction, scanned/mixed PDFs and "
        "images use Standard with selective RapidOCR; office formats "
        "use Docling's model-free SimplePipeline), ``standard`` "
        "(layout + TableFormer + selective OCR, explicit), ``native`` "
        "(fast model-free PDF text extraction), or ``vlm`` "
        "(vision-language model per ``VS_PARSER__VLM_PRESET``). The "
        "profile is ignored for text inputs.\n\n"
        "Pictures are saved server-side and referenced from the "
        "markdown via ``/artifacts/...`` URLs (see "
        "``VS_PARSER__SAVE_IMAGES``)."
    ),
)
async def parse_document(
    file: UploadFile,
    request: Request,
    profile: str = Form("auto"),
):
    profile = _validate_profile(profile)
    settings = request.app.state.settings.parser
    mime, data = await _prepare_parse(file, settings)
    parser = _parser_for_mime(mime)

    t0 = time.perf_counter()
    log.info("parse_started", mime=mime, bytes=len(data), profile=profile, stream=False)
    try:
        parsed = await parser.parse_bytes(data, mime, profile=profile)
    except HTTPException:
        raise
    except Exception as e:
        raise _parser_failure_http(e, mime)

    metadata = ParseMetadata(**parsed.metadata)
    # One structured line per successful parse — the Docling/RapidOCR
    # internals are quiet at INFO; VS_LOG_LEVEL=DEBUG restores them.
    log.info(
        "parse_done",
        mime=mime,
        bytes=len(data),
        profile=profile,
        chars=len(parsed.markdown),
        page_count=parsed.metadata.get("page_count"),
        images=len(parsed.images),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
    return ParseResponse(
        markdown=parsed.markdown,
        metadata=metadata,
        images=[img.uri for img in parsed.images],
    )


@router.post(
    "/parse/stream",
    responses=_PARSE_RESPONSES,
    summary="Parse a document to markdown with per-page progress",
    description=(
        "Same multipart contract as ``POST /v1/parse`` but responds with "
        "``application/x-ndjson``: one JSON event per line. Event shapes::"
        "\n\n"
        '    {"type": "stage", "stage": "parse"}\n'
        '    {"type": "progress", "stage": "parse", "page": 3, "total": 12}\n'
        '    {"type": "result", "markdown": "...", "metadata": {...}}\n'
        '    {"type": "error", "status": 500, "error": {"code": ...}}\n'
        "\n\n"
        "Progress events fire per completed page for paginated binary "
        "documents (PDF/images); text uploads emit none. "
        "Pre-flight failures (bad MIME / size / empty upload / unknown "
        "profile) are ordinary JSON error envelopes before the stream "
        "starts. Accepts the same ``file`` and optional ``profile`` "
        "form fields as ``/v1/parse``."
    ),
)
async def parse_document_stream(
    file: UploadFile,
    request: Request,
    profile: str = Form("auto"),
):
    """File → markdown as an NDJSON event stream with page progress."""
    profile = _validate_profile(profile)
    settings = request.app.state.settings.parser
    # Pre-flight failures come back as ordinary JSON envelopes (the
    # StreamingResponse hasn't started yet).
    mime, data = await _prepare_parse(file, settings)
    parser = _parser_for_mime(mime)

    async def event_stream():
        # Same queue-decoupled shape as /v1/ingest/stream: the converter
        # runs in a worker thread while page ticks and the terminal
        # event get flushed as soon as they happen.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def emit(event: dict[str, Any]) -> None:
            queue.put_nowait(event)

        # Ticks are scheduled with call_soon_threadsafe, so they land on
        # the loop's ready queue rather than in the queue directly. Drain
        # them before enqueuing the terminal event — otherwise a fast
        # (or same-thread) parser can put result+sentinel ahead of the
        # still-pending progress callbacks and the consumer would stop
        # without ever yielding them.
        tick_scheduled = 0
        tick_flushed = 0

        def on_progress(done: int, total: int) -> None:
            # Fires from the Docling worker thread — hop onto the event
            # loop before touching the queue.
            nonlocal tick_scheduled
            tick_scheduled += 1
            queue_event = {
                "type": "progress",
                "stage": "parse",
                "page": int(done),
                "total": int(total),
            }

            def _put_tick() -> None:
                nonlocal tick_flushed
                queue.put_nowait(queue_event)
                tick_flushed += 1

            loop.call_soon_threadsafe(_put_tick)

        async def runner() -> None:
            t0 = time.perf_counter()
            try:
                await emit({"type": "stage", "stage": "parse"})
                log.info(
                    "parse_started",
                    mime=mime,
                    bytes=len(data),
                    profile=profile,
                    stream=True,
                )
                parsed = await parser.parse_bytes(
                    data, mime, on_progress=on_progress, profile=profile
                )
                # Let every scheduled tick reach the queue before the
                # terminal result so event order is deterministic.
                while tick_flushed < tick_scheduled:
                    await asyncio.sleep(0)
            except HTTPException as e:
                queue.put_nowait(_http_error_event(e))
            except Exception as e:  # noqa: BLE001 — terminal stream event
                queue.put_nowait(_http_error_event(_parser_failure_http(e, mime)))
            else:
                payload = ParseResponse(
                    markdown=parsed.markdown,
                    metadata=ParseMetadata(**parsed.metadata),
                    images=[img.uri for img in parsed.images],
                )
                queue.put_nowait({"type": "result", **payload.model_dump()})
                log.info(
                    "parse_done",
                    mime=mime,
                    bytes=len(data),
                    profile=profile,
                    chars=len(parsed.markdown),
                    page_count=parsed.metadata.get("page_count"),
                    images=len(parsed.images),
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
            finally:
                queue.put_nowait(sentinel)

        # Not cancelled on client disconnect: Docling conversion holds
        # GPU/thread resources and wraps its own errors into a terminal
        # event, mirroring the ingest stream's runner policy.
        task = asyncio.create_task(runner())
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Parser cache lifecycle — warm / evict (NOT service load/unload)
# ---------------------------------------------------------------------------


@router.post(
    "/parser/warm",
    responses={
        400: {"model": ErrorEnvelope, "description": "Unknown profile."},
        500: {"model": ErrorEnvelope, "description": "Converter build failed."},
        503: {"model": ErrorEnvelope, "description": "Docling not installed."},
    },
    summary="Warm a Docling profile (pre-build its converter)",
    description=(
        "Build and cache the ``DocumentConverter`` for one profile "
        "(``standard`` / ``native`` / ``vlm``) ahead of the first parse, "
        "paying its model download / load cost up front. This warms a "
        "*cache* — it is not a service start; parsing works without it, "
        "and the converter is rebuilt on demand after an evict."
    ),
)
async def warm_parser(body: ParserProfileAction):
    profile = _validate_action_profile(body.profile)
    parser = get_docling_parser()
    try:
        # Converter build is heavy + synchronous; ride the model pool.
        resolved = await run_in_model(parser.warm, profile)
    except ParserUnavailable as e:
        raise _action_http(503, "parser_unavailable", str(e))
    except RuntimeError as e:
        raise _action_http(500, "parser_failed", str(e))
    log.info("parser_warmed", profile=resolved)
    return {"profile": resolved, "warm": True}


@router.post(
    "/parser/evict",
    responses={
        400: {"model": ErrorEnvelope, "description": "Unknown profile."},
    },
    summary="Evict a Docling profile's cached converter",
    description=(
        "Release the cached converter for one profile "
        "(``standard`` / ``native`` / ``vlm``) and the model weights it "
        "holds. Parsing is NOT stopped: the next request for that "
        "profile rebuilds the converter and pays the cold-start cost "
        "again. ``released`` reports whether a converter was actually "
        "held."
    ),
)
async def evict_parser(body: ParserProfileAction):
    profile = _validate_action_profile(body.profile)
    # Converter teardown is synchronous; ride the model pool like warm.
    released = await run_in_model(
        get_docling_parser().release, profile
    )
    log.info("parser_evicted", profile=profile, released=released)
    return {"profile": profile, "warm": False, "released": released}
