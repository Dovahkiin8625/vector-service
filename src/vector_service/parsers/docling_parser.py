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
import contextvars
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vector_service.core.errors import VectorServiceError
from vector_service.parsers.base import (
    DocumentParser,
    ParsedDocument,
    ProgressCallback,
)

if TYPE_CHECKING:  # pragma: no cover — import only for typing
    from docling.document_converter import DocumentConverter

# Per-convert progress sink. Docling runs the conversion on a worker
# thread (the route dispatches via run_in_executor), so the callback
# cannot travel as a plain closure-scoped global — a contextvar set
# inside the worker thread is visible to every pipeline stage there.
_current_progress: contextvars.ContextVar[ProgressCallback | None] = contextvars.ContextVar(
    "docling_parse_progress", default=None,
)


def _wrap_pipeline(pipeline: Any) -> Any:
    """Make Docling's shared, cached pipeline report per-page progress.

    ``DocumentConverter`` caches one pipeline instance per options hash
    (``document_converter.py`` ``_get_pipeline``), and every page is
    yielded from ``PaginatedPipeline._apply_on_pages`` only AFTER it has
    passed the full stage chain (layout → table → OCR → assembly — see
    ``base_pipeline.py``). Wrapping that generator once per pipeline
    gives us a reliable "page n of N finished" tick. The wrapper reads
    the current contextvar, so concurrent converts on different threads
    each see their own callback. Idempotent: repeated convert() calls
    must not stack wrappers.
    """
    if getattr(pipeline, "_vs_progress_wrapped", False):
        return pipeline
    if not hasattr(pipeline, "_apply_on_pages"):
        # Non-paginated pipeline (no per-page concept) — nothing to wrap.
        return pipeline

    original = pipeline._apply_on_pages

    def _wrapped(conv_res: Any, page_batch: Any) -> Any:
        page_count = getattr(getattr(conv_res, "input", None), "page_count", None)
        for page in original(conv_res, page_batch):
            callback = _current_progress.get()
            page_no = getattr(page, "page_no", None)
            if callback is not None and isinstance(page_no, int):
                try:
                    callback(page_no, page_count if isinstance(page_count, int) else 0)
                except Exception:  # noqa: BLE001 — progress must never break conversion
                    pass
            yield page

    pipeline._apply_on_pages = _wrapped
    pipeline._vs_progress_wrapped = True
    return pipeline


class _ProgressQueueProxy:
    """Intercept ``get_batch`` drains without touching the slotted queue.

    Delegates every other attribute (``put``, ``close``, ``closed`` …)
    to the wrapped :class:`ThreadedQueue` so the drain loop's lifecycle
    calls behave identically. The completed-page set is per-run: the
    pipeline factory constructs one proxy per convert call.
    """

    __slots__ = ("_queue", "_completed", "_total")

    def __init__(self, queue: Any) -> None:
        self._queue = queue
        self._completed: set[int] = set()
        self._total = 0

    def get_batch(self, size: Any, timeout: Any = None, *args: Any, **kwargs: Any) -> Any:
        batch = self._queue.get_batch(size, timeout, *args, **kwargs)
        callback = _current_progress.get()
        if callback and batch:
            for item in batch:
                page_no = getattr(item, "page_no", None)
                if isinstance(page_no, int):
                    self._completed.add(page_no)
                conv_res = getattr(item, "conv_res", None)
                page_count = getattr(getattr(conv_res, "input", None), "page_count", None)
                if isinstance(page_count, int):
                    self._total = page_count
            if self._completed:
                try:
                    callback(len(self._completed), self._total)
                except Exception:  # noqa: BLE001 — progress must never break conversion
                    pass
        return batch

    def close(self) -> None:
        self._queue.close()

    @property
    def closed(self) -> bool:
        return bool(self._queue.closed)

    def __getattr__(self, name: str) -> Any:
        # ``object.__getattribute__`` avoids recursion if accessed
        # before ``__init__`` assigned the ``_queue`` slot.
        return getattr(object.__getattribute__(self, "_queue"), name)


def _wrap_threaded_pdf_pipeline(pipeline: Any) -> Any:
    """Add page ticks to docling 2.12+'s threaded ``StandardPdfPipeline``.

    That pipeline (the default for PDF/DOCX/PPTX in recent docling)
    bypasses :meth:`PaginatedPipeline._apply_on_pages`: six worker
    stages communicate over bounded queues, and ``_build_document``
    drains finished pages with ``RunContext.output_queue.get_batch``.
    Wrapping the queue returned per-run by ``_create_run_ctx`` is the
    one point where every finished page (success OR failure) passes in
    page order of completion.

    Counts are unique-``page_no`` based because pages complete out of
    order; the run-scoped completed set is rebuilt by ``_create_run_ctx``
    on every convert, so a second document restarts at zero. The
    contextvar is read at drain time — drains run on the caller's
    converter thread, i.e. our executor worker.

    The real ``ThreadedQueue`` declares ``__slots__`` and has no
    ``__weakref__`` slot, so neither per-instance method patching nor a
    weakref map works: assigning ``queue.get_batch`` raises
    ``AttributeError`` INSIDE ``_build_document`` and fails every page.
    ``RunContext`` is a plain (non-frozen) dataclass — the wrapper
    replaces ``ctx.output_queue`` with a thin proxy instead. Worker
    stages were wired to the underlying queue before the factory
    returned, so only the drain loop sees the proxy.
    """
    if getattr(pipeline, "_vs_progress_wrapped", False):
        return pipeline
    create_run_ctx = getattr(pipeline, "_create_run_ctx", None)
    if not callable(create_run_ctx):
        # Legacy/simple pipeline — no threaded run context.
        return pipeline

    def _reporting_run_ctx() -> Any:
        ctx = create_run_ctx()
        out_q = getattr(ctx, "output_queue", None)
        if out_q is None or not hasattr(out_q, "get_batch"):
            return ctx
        ctx.output_queue = _ProgressQueueProxy(out_q)
        return ctx

    # Instance attr shadows the class method. The call site is
    # ``self._create_run_ctx()`` with zero args — an instance-attr
    # function is NOT bound, so the wrapper takes no ``self`` and closes
    # over ``pipeline`` instead.
    pipeline._create_run_ctx = _reporting_run_ctx
    pipeline._vs_progress_wrapped = True
    return pipeline


def _build_progress_converter(*, fast: bool = False) -> "DocumentConverter":
    """Construct a progress-reporting converter.

    Imported lazily inside ``DoclingParser.load``'s lock so the class is
    created only when Docling is actually installed.

    ``fast=True`` builds the text-layer converter: the PDF pipeline is
    created with ``do_ocr=False`` (and ``do_table_structure`` from
    ``VS_PARSER__TABLE_STRUCTURE``), so text-native PDFs skip RapidOCR
    entirely. It is a separate converter because Docling caches one
    pipeline per options hash per ``DocumentConverter`` instance.
    """
    from docling.document_converter import DocumentConverter

    # Tests inject a factory/instance in place of the class; subclassing
    # a non-class raises, so duck-type the branch.
    if not isinstance(DocumentConverter, type):
        return DocumentConverter()

    format_options = None
    if fast:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        from vector_service.core.config import get_settings

        parser_settings = get_settings().parser
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    do_table_structure=parser_settings.table_structure,
                )
            )
        }

    class _ProgressConverterImpl(DocumentConverter):
        def __init__(self) -> None:
            if format_options is not None:
                super().__init__(format_options=format_options)
            else:
                super().__init__()

        def _get_pipeline(self, doc_format: Any) -> Any:
            pipeline = super()._get_pipeline(doc_format)
            if pipeline is not None:
                # docling 2.12+ threaded StandardPdfPipeline first; the
                # legacy generator pipeline second. Each wrapper is a
                # no-op on pipelines it doesn't recognise, and both are
                # idempotent against converter-level pipeline caching.
                _wrap_threaded_pdf_pipeline(pipeline)
                _wrap_pipeline(pipeline)
            return pipeline

    return _ProgressConverterImpl()


def _apply_hub_env() -> None:
    """Configure HuggingFace Hub env vars BEFORE Docling/HF is imported.

    Docling pulls its layout/table models from the HuggingFace Hub on
    first use; on networks where huggingface.co is unreachable the
    operator sets ``VS_PARSER__HF_ENDPOINT`` (e.g.
    ``https://hf-mirror.com``). ``huggingface_hub`` reads
    ``HF_ENDPOINT`` at import time, and Docling is imported lazily
    inside :meth:`DoclingParser.load`, so setting it here is early
    enough. Download progress bars are suppressed unless the service
    runs at DEBUG — in a server log each bar becomes a stream of
    progress lines, while a stalled/failed download still surfaces as
    a regular exception either way.
    """
    from vector_service.core.config import get_settings

    endpoint = get_settings().parser.hf_endpoint.strip()
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
        # Mirrors (e.g. hf-mirror.com) serve the regular file API but
        # can't proxy the Xet CAS backend (cas-server.xethub.hf.co
        # answers 401), so force the classic HTTP download path.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if get_settings().log_level.upper() != "DEBUG":
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # Windows hosts without Developer Mode fall back to copying
        # cache files — works fine, but emits a multi-line UserWarning.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


#: A page with fewer stripped characters of embedded text than this is
#: treated as scanned (no usable text layer).
_TEXT_LAYER_MIN_CHARS = 10


def _pdf_has_text_layer(path: Path) -> bool:
    """True when EVERY page of ``path`` exposes extractable native text.

    Reads the PDF's embedded text layer via PyMuPDF (a declared
    dependency) — no rendering and no OCR, so this is cheap even for
    large files. Used by ``ocr=auto`` to route text-native PDFs onto
    the OCR-free fast pipeline while keeping the full OCR pipeline for
    scans. Any failure (not a PDF, encrypted, malformed, zero pages,
    one textless page) returns ``False`` so the caller conservatively
    keeps the full OCR pipeline instead of silently dropping text.
    """
    try:
        import fitz
    except ImportError:  # pragma: no cover — pymupdf is a declared dep
        return False
    try:
        with fitz.open(path) as doc:
            if doc.is_encrypted or doc.page_count == 0:
                return False
            for page in doc:
                if len(page.get_text("text").strip()) < _TEXT_LAYER_MIN_CHARS:
                    return False
    except Exception:  # noqa: BLE001 — any read failure => safe default
        return False
    return True


class ParserUnavailable(VectorServiceError):
    """The Docling backend is not available (not installed or not loaded)."""


class DoclingParser(DocumentParser):
    """DocumentConverter-based parser.

    Accepts the binary formats that Lumos currently ingests — PDF,
    DOCX, PPTX, and HTML. Markdown and plain text are handled by
    :class:`vector_service.parsers.markdown_parser.MarkdownParser`
    instead (they don't need Docling's layout model).

    PDFs are routed between TWO lazily built converters according to
    ``VS_PARSER__OCR`` (default ``auto``): the primary runs Docling's
    full OCR pipeline (scans), the fast one runs ``do_ocr=False`` for
    PDFs whose every page exposes an embedded text layer (see
    :func:`_pdf_has_text_layer`). DOCX/PPTX/HTML always use the
    primary.
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
        # Primary converter: Docling defaults (OCR + table structure),
        # used for scans and every non-PDF format.
        self._converter: DocumentConverter | None = None
        # Fast converter: ``do_ocr=False`` PDF pipeline, built lazily on
        # the first text-native PDF so hosts that only see scans never
        # pay for a second converter's model set.
        self._fast_converter: DocumentConverter | None = None
        self._lock = threading.Lock()
        self._fast_lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------

    def load(self) -> None:
        """Eagerly instantiate the primary ``DocumentConverter``.

        Only the full OCR converter is built here; the OCR-free fast
        converter is created lazily by the first text-native PDF.
        Idempotent and 409-tolerant — calling ``load`` twice is a
        no-op. Idempotent loading is required so the lifespan handler
        can call this on every startup (cold + warm starts alike)
        without surfacing spurious errors.
        """
        with self._lock:
            if self._converter is not None:
                return
            _apply_hub_env()
            try:
                # Import validates that docling is installed; the
                # converter itself is the progress-reporting subclass.
                from docling.document_converter import DocumentConverter  # noqa: F401
            except ImportError as e:
                raise ParserUnavailable(
                    "docling is not installed; install it to enable binary "
                    "document parsing (pip install 'docling>=2.0')"
                ) from e
            self._converter = _build_progress_converter()

    def unload(self) -> None:
        """Release the cached converters. Idempotent."""
        with self._lock:
            self._converter = None
        with self._fast_lock:
            self._fast_converter = None

    def _ensure_loaded(self) -> "DocumentConverter":
        if self._converter is None:
            self.load()
        assert self._converter is not None  # for type checkers
        return self._converter

    def _ensure_fast_converter(self) -> DocumentConverter:
        """Lazily build (and cache) the OCR-free PDF converter."""
        with self._fast_lock:
            if self._fast_converter is None:
                # load() owns the HF-env setup and the docling import
                # check; the fast converter must not be constructible
                # on a host where the primary is not.
                self._ensure_loaded()
                self._fast_converter = _build_progress_converter(fast=True)
            return self._fast_converter

    def _select_converter(self, path: Path, mime: str) -> DocumentConverter:
        """Pick the OCR or OCR-free converter for one document.

        Only PDFs are eligible for the fast path; DOCX/PPTX/HTML keep
        the primary converter so their behaviour is unchanged.
        ``VS_PARSER__OCR`` forces ``on`` (always OCR) or ``off``
        (never OCR); the default ``auto`` inspects the PDF's embedded
        text layer via :func:`_pdf_has_text_layer`.
        """
        primary = self._ensure_loaded()
        if mime != "application/pdf":
            return primary

        from vector_service.core.config import get_settings

        mode = get_settings().parser.ocr
        if mode == "on":
            return primary
        if mode == "off":
            return self._ensure_fast_converter()
        # auto
        if _pdf_has_text_layer(path):
            return self._ensure_fast_converter()
        return primary

    # ---- parsing ------------------------------------------------------

    def parse(
        self,
        path: Path,
        on_progress: ProgressCallback | None = None,
    ) -> ParsedDocument:
        """Parse a file on disk via Docling's ``DocumentConverter``.

        The conversion is CPU/GPU heavy — call sites should dispatch to
        a thread executor (the route does this via
        ``loop.run_in_executor``). ``on_progress`` ticks once per
        completed page as ``(pages_done, total_pages)``.
        """
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        mime = guess_mime(path)
        converter = self._select_converter(path, mime)
        token = _current_progress.set(on_progress)
        try:
            result = converter.convert(str(path))
        except ParserUnavailable:
            raise
        except Exception as e:
            raise RuntimeError(f"docling conversion failed for {path}: {e}") from e
        finally:
            _current_progress.reset(token)
        return _result_to_parsed(result, mime=mime)

    async def parse_bytes(
        self,
        data: bytes,
        mime: str,
        on_progress: ProgressCallback | None = None,
    ) -> ParsedDocument:
        """Parse in-memory bytes via a temporary file.

        Docling's converter is path-oriented; the smallest integration
        surface is to spill ``data`` to a tempfile and let Docling read
        it back. The temp file is cleaned up via ``tempfile``'s context
        manager semantics. The progress contextvar is set INSIDE the
        worker thread — contextvars don't cross run_in_executor threads
        automatically.
        """
        loop = asyncio.get_running_loop()

        def _work() -> ParsedDocument:
            import tempfile

            suffix = _suffix_for_mime(mime)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="vs_ingest_",
            ) as fh:
                fh.write(data)
                tmp_path = fh.name
            # Select on the worker thread once the temp file exists:
            # auto mode needs to read the PDF's text layer to decide.
            converter = self._select_converter(Path(tmp_path), mime)
            token = _current_progress.set(on_progress)
            try:
                result = converter.convert(tmp_path)
            except ParserUnavailable:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"docling conversion failed for in-memory {mime}: {e}"
                ) from e
            finally:
                _current_progress.reset(token)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return _result_to_parsed(result, mime=mime)

        return await loop.run_in_executor(None, _work)


# ---- process-wide singleton -------------------------------------------


_parser_singleton: DoclingParser | None = None
_parser_singleton_lock = threading.Lock()


def get_docling_parser() -> DoclingParser:
    """Return the process-wide :class:`DoclingParser`.

    Both the HTTP routes and the lifespan warmup must hit one cached
    ``DocumentConverter``: Docling's layout/OCR models cost ~10-30s to
    build, so constructing a fresh parser per request made every upload
    pay the cold-start cost even after warmup.
    """
    global _parser_singleton
    with _parser_singleton_lock:
        if _parser_singleton is None:
            _parser_singleton = DoclingParser()
        return _parser_singleton


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
