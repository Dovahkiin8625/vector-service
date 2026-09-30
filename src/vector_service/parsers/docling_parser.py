"""Docling-backed parser for PDF / DOCX / PPTX / HTML / images.

Mirrors the lazy-load singleton pattern from
:mod:`vector_service.embeddings.bge_m3`: Docling model downloads are
heavy on first use, so converters are constructed on demand and cached
on the singleton. The lifespan step warms the default converter
explicitly (``VS_PARSER__AUTO_LOAD=true``); concurrent FastAPI workers
share one set of converters under a lock.

Three named *profiles* are supported (see
:mod:`vector_service.parsers.base`):

- ``standard`` — DocLayNet layout + TableFormer v2 + **selective**
  RapidOCR (PP-OCRv4, ONNX Runtime). Digital pages skip OCR
  automatically (``OcrMode.DEFAULT`` only OCRs layout regions not
  covered by the PDF text layer), scanned pages OCR as needed, and
  mixed PDFs are handled per region. This is what ``auto`` picks for
  images and scanned/mixed PDFs — a confirmed-digital PDF goes Native;
- ``native``   — the model-free docling-parse/Rust PDF backend. Fast but
  weak table recovery; PDFs only (image inputs still go Standard);
- ``vlm``      — end-to-end vision-language model conversion
  (``VS_PARSER__VLM_PRESET``, Granite-Docling-258M by default). Weights
  are NOT pre-downloaded; the first ``vlm`` request pays the fetch.

DOCX / PPTX / HTML always use Docling's model-free ``SimplePipeline``
regardless of profile — those formats carry their own structure, and
every converter keeps Docling's default option for them.

When ``VS_PARSER__SAVE_IMAGES=true`` (default), pictures extracted from
the document are saved to
``artifacts_dir/<doc-stem>/images/image_NNNNNN_<hash>.png`` and the
markdown references them through ``artifacts_url_prefix`` (served by
the ``/artifacts`` static mount) instead of an ``<!-- image -->``
placeholder.

Docling is an optional dependency — if it is not installed the parser
still imports cleanly but raises :class:`ParserUnavailable` at
converter-build time.
"""
from __future__ import annotations

import contextvars
import os
import re
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vector_service.core.config import get_settings
from vector_service.core.errors import VectorServiceError
from vector_service.core.threadpools import run_in_model
from vector_service.parsers.base import (
    DocumentParser,
    PARSE_PROFILES,
    ParsedDocument,
    PROFILE_AUTO,
    PROFILE_NATIVE,
    PROFILE_STANDARD,
    PROFILE_VLM,
    ProgressCallback,
    SavedImage,
)
from vector_service.parsers.pdf_probe import (
    DIGITAL,
    PdfProbe,
    probe_pdf_bytes,
    probe_pdf_path,
)

if TYPE_CHECKING:  # pragma: no cover — import only for typing
    from docling.document_converter import DocumentConverter

#: Profiles that select a real, cached converter (``auto`` is resolved
#: by :func:`_auto_profile` before the cache is touched).
_CONCRETE_PROFILES = (PROFILE_STANDARD, PROFILE_NATIVE, PROFILE_VLM)

#: Raster image MIMEs — every image page must pass layout + OCR.
_IMAGE_MIMES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/bmp",
})


def _auto_profile(mime: str | None, probe: PdfProbe | None) -> str:
    """Pick the concrete pipeline for ``auto`` from the document type.

    - raster images → ``standard`` (layout model + OCR are mandatory);
    - a PDF the probe confirms as ``digital`` → ``native`` (model-free
      extraction, near-instant — no reason to pay the standard model
      cost); scanned / mixed / unprobeable PDFs → ``standard``, whose
      selective OCR repairs the image-only pages;
    - DOCX / PPTX / HTML → ``standard``: Docling routes them through
      the model-free SimplePipeline under EVERY converter, so this
      just reuses the warm standard converter.
    """
    if mime in _IMAGE_MIMES:
        return PROFILE_STANDARD
    if mime == "application/pdf":
        if probe is not None and probe.kind == DIGITAL:
            return PROFILE_NATIVE
        return PROFILE_STANDARD
    return PROFILE_STANDARD

# Per-convert progress sink. Docling runs the conversion on a worker
# thread (call sites dispatch through the model pool), so the callback
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

    That pipeline (the default for PDF in recent docling) bypasses
    :meth:`PaginatedPipeline._apply_on_pages`: worker stages communicate
    over bounded queues, and ``_build_document`` drains finished pages
    with ``RunContext.output_queue.get_batch``. Wrapping the queue
    returned per-run by ``_create_run_ctx`` is the one point where every
    finished page (success OR failure) passes in completion order.

    Counts are unique-``page_no`` based because pages complete out of
    order; the run-scoped completed set is rebuilt by ``_create_run_ctx``
    on every convert, so a second document restarts at zero. The
    contextvar is read at drain time — drains run on the caller's
    converter thread, i.e. our executor worker.

    The real ``ThreadedQueue`` declares ``__slots__`` and has no
    ``__weakref__`` slot, so neither per-instance method patching nor a
    weakref map works. ``RunContext`` is a plain dataclass — the wrapper
    replaces ``ctx.output_queue`` with a thin proxy instead.
    """
    if getattr(pipeline, "_vs_progress_wrapped", False):
        return pipeline
    create_run_ctx = getattr(pipeline, "_create_run_ctx", None)
    if not callable(create_run_ctx):
        # Non-threaded paginated pipeline (e.g. VlmPipeline) — the
        # generator wrapper (_wrap_pipeline) carries its progress.
        return pipeline

    def _reporting_run_ctx() -> Any:
        ctx = create_run_ctx()
        out_q = getattr(ctx, "output_queue", None)
        if out_q is None or not hasattr(out_q, "get_batch"):
            return ctx
        ctx.output_queue = _ProgressQueueProxy(out_q)
        return ctx

    # The call site is ``self._create_run_ctx()`` with zero args — an
    # instance-attr function is NOT bound, so the wrapper takes no
    # ``self`` and closes over ``pipeline`` instead.
    pipeline._create_run_ctx = _reporting_run_ctx
    pipeline._vs_progress_wrapped = True
    return pipeline


def _make_progress_converter(format_options: dict | None = None) -> "DocumentConverter":
    """Construct a converter whose pipelines report per-page progress.

    Imported lazily inside the parser's lock so the subclass is created
    only when Docling is actually installed.
    """
    from docling.document_converter import DocumentConverter

    # Tests inject a factory/instance in place of the class; subclassing
    # a non-class raises, so duck-type the branch.
    if not isinstance(DocumentConverter, type):
        return DocumentConverter()

    class _ProgressConverterImpl(DocumentConverter):
        def _get_pipeline(self, doc_format: Any) -> Any:
            pipeline = super()._get_pipeline(doc_format)
            if pipeline is not None:
                # The threaded wrapper handles 2.12+'s StandardPdfPipeline;
                # the generator wrapper handles non-threaded paginated
                # pipelines (VlmPipeline). Each is a no-op on pipeline
                # types it doesn't recognise and idempotent against
                # converter-level pipeline caching.
                _wrap_threaded_pdf_pipeline(pipeline)
                _wrap_pipeline(pipeline)
            return pipeline

    # Test doubles substituted for DocumentConverter typically keep a
    # parameterless __init__ — only the real class is known to accept
    # format_options, so branch on the class's origin module.
    if DocumentConverter.__module__ == "docling.document_converter":
        return _ProgressConverterImpl(format_options=format_options or {})
    return _ProgressConverterImpl()


def _apply_hub_env() -> None:
    """Configure HuggingFace Hub env vars BEFORE Docling/HF is imported.

    Docling pulls its layout/table/OCR models from the HuggingFace Hub
    on first use; on networks where huggingface.co is unreachable the
    operator sets ``VS_PARSER__HF_ENDPOINT`` (e.g.
    ``https://hf-mirror.com``). ``huggingface_hub`` reads
    ``HF_ENDPOINT`` at import time, and Docling is imported lazily
    inside the converter-build path, so setting it here is early
    enough. Download progress bars are suppressed unless the service
    runs at DEBUG — in a server log each bar becomes a stream of
    progress lines, while a stalled/failed download still surfaces as
    a regular exception either way.
    """
    settings = get_settings()

    endpoint = settings.parser.hf_endpoint.strip()
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
        # Mirrors (e.g. hf-mirror.com) serve the regular file API but
        # can't proxy the Xet CAS backend (cas-server.xethub.hf.co
        # answers 401), so force the classic HTTP download path.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if settings.log_level.upper() != "DEBUG":
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # Windows hosts without Developer Mode fall back to copying
        # cache files — works fine, but emits a multi-line UserWarning.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class ParserUnavailable(VectorServiceError):
    """The Docling backend is not available (not installed or not loaded)."""


class DoclingParser(DocumentParser):
    """DocumentConverter-based parser with per-profile converters.

    Accepts PDF, DOCX, PPTX, HTML and raster images. Markdown and plain
    text are handled by
    :class:`vector_service.parsers.markdown_parser.MarkdownParser`
    instead (they don't need Docling's layout model).
    """

    accepted_mime: tuple[str, ...] = (
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
    )

    #: Map from MIME type to Docling's ``InputFormat`` value.
    _MIME_TO_INPUT: dict[str, str] = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "text/html": "html",
        "application/xhtml+xml": "html",
        "image/jpeg": "image",
        "image/png": "image",
        "image/tiff": "image",
        "image/webp": "image",
        "image/bmp": "image",
    }

    def __init__(self) -> None:
        self._converters: dict[str, DocumentConverter] = {}
        self._lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------

    def load(self) -> None:
        """Eagerly warm the default (``standard``) converter.

        Idempotent — calling ``load`` twice is a no-op once the
        converter + its pipelines exist. Required so the lifespan
        handler can call it on every startup without surfacing
        spurious errors. Other profiles warm lazily on first use.
        """
        self.warm(PROFILE_STANDARD)

    def unload(self) -> None:
        """Release all cached converters. Idempotent."""
        with self._lock:
            self._converters.clear()

    def warm(self, profile: str = PROFILE_STANDARD) -> str:
        """Build the converter for one profile and pre-load its pipelines.

        Returns the resolved profile key. Heavy — callers dispatch to
        an executor. Recent Docling builds pipelines lazily on the
        first convert, so this additionally calls the converter's
        ``initialize_pipeline`` per format: the warm-up genuinely
        pays the model load cost up front instead of merely caching
        an empty converter. Unlike the four inference families a
        warmed converter is a *cache*: the parser keeps working after
        :meth:`release`, the next parse simply rebuilds it.
        """
        resolved = self._resolve_profile(profile)
        converter = self._ensure_converter(resolved)
        _preinitialize_pipelines(resolved, converter)
        return resolved

    def release(self, profile: str = PROFILE_STANDARD) -> bool:
        """Evict one profile's cached converter.

        Returns ``True`` when a converter was actually held. This does
        NOT stop the parsing service — the next request for that
        profile rebuilds the converter (cold-start cost).
        """
        resolved = self._resolve_profile(profile)
        with self._lock:
            return self._converters.pop(resolved, None) is not None

    @staticmethod
    def _resolve_profile(
        profile: str,
        mime: str | None = None,
        probe: PdfProbe | None = None,
    ) -> str:
        """Map the public profile name onto a cached converter key.

        ``auto`` is type-aware via :func:`_auto_profile`; without MIME
        context it conservatively lands on the standard converter.
        """
        if profile == PROFILE_AUTO:
            return _auto_profile(mime, probe)
        if profile in _CONCRETE_PROFILES:
            return profile
        valid = ", ".join(PARSE_PROFILES)
        raise ValueError(f"unknown parse profile {profile!r}; expected one of {valid}")

    def _ensure_converter(self, profile: str) -> "DocumentConverter":
        """Return the cached converter for ``profile``, building it once."""
        resolved = self._resolve_profile(profile)
        converter = self._converters.get(resolved)
        if converter is not None:
            return converter
        with self._lock:
            converter = self._converters.get(resolved)
            if converter is None:
                _apply_hub_env()
                try:
                    import docling  # noqa: F401 — validates the optional dependency
                except ImportError as e:
                    raise ParserUnavailable(
                        "docling is not installed; install it to enable binary "
                        "document parsing (pip install 'docling>=2.0')"
                    ) from e
                try:
                    format_options = _build_format_options(resolved)
                    converter = _make_progress_converter(format_options)
                except ParserUnavailable:
                    raise
                except Exception as e:
                    raise RuntimeError(
                        f"failed to build docling converter for profile {resolved!r}: {e}"
                    ) from e
                self._converters[resolved] = converter
            return converter

    # ---- parsing ------------------------------------------------------

    def parse(
        self,
        path: Path,
        on_progress: ProgressCallback | None = None,
        *,
        profile: str = PROFILE_AUTO,
        artifact_stem: str | None = None,
    ) -> ParsedDocument:
        """Parse a file on disk via Docling's ``DocumentConverter``.

        CPU/GPU heavy — call sites dispatch via the model thread pool
        (``run_in_model``). ``on_progress`` ticks once per completed
        page as ``(pages_done, total_pages)``.
        """
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        mime = guess_mime(path)
        probe = probe_pdf_path(path) if mime == "application/pdf" else None
        resolved = self._resolve_profile(profile, mime, probe)
        converter = self._ensure_converter(resolved)
        stem = _safe_stem(artifact_stem)

        token = _current_progress.set(on_progress)
        try:
            result = converter.convert(str(path))
        except ParserUnavailable:
            raise
        except Exception as e:
            raise RuntimeError(f"docling conversion failed for {path}: {e}") from e
        finally:
            _current_progress.reset(token)
        return self._result_to_parsed(
            result, mime=mime, profile=resolved, probe=probe, stem=stem
        )

    async def parse_bytes(
        self,
        data: bytes,
        mime: str,
        on_progress: ProgressCallback | None = None,
        *,
        profile: str = PROFILE_AUTO,
        artifact_stem: str | None = None,
    ) -> ParsedDocument:
        """Parse in-memory bytes via a temporary file.

        Docling's converter is path-oriented; the smallest integration
        surface is to spill ``data`` to a tempfile and let Docling read
        it back. The temp file is always removed; extracted images go to
        the configured artifacts directory and persist. The progress
        contextvar is set INSIDE the worker thread — contextvars don't
        cross pool threads automatically.
        """
        probe = probe_pdf_bytes(data) if mime == "application/pdf" else None
        resolved = self._resolve_profile(profile, mime, probe)
        stem = _safe_stem(artifact_stem)

        def _work() -> ParsedDocument:
            import tempfile

            converter = self._ensure_converter(resolved)
            suffix = _suffix_for_mime(mime)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="vs_ingest_",
            ) as fh:
                fh.write(data)
                tmp_path = fh.name
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
            return self._result_to_parsed(
                result, mime=mime, profile=resolved, probe=probe, stem=stem
            )

        return await run_in_model(_work)

    # ---- result handling ----------------------------------------------

    def _result_to_parsed(
        self,
        result: Any,
        *,
        mime: str,
        profile: str,
        probe: PdfProbe | None,
        stem: str,
    ) -> ParsedDocument:
        """Translate a Docling ``ConversionResult`` to ParsedDocument.

        When image saving is enabled, pictures are written to
        ``artifacts_dir/<stem>/images`` by Docling itself
        (``ImageRefMode.REFERENCED``) and referenced through the served
        URL prefix; afterwards the on-disk directory is enumerated to
        build the SavedImage list (content hashes dedupe writes).
        """
        settings = get_settings().parser

        doc_dir: Path | None = None
        images_dir: Path | None = None
        uri_prefix: str | None = None
        export_kwargs: dict[str, Any] = {}
        if settings.save_images:
            doc_dir = Path(settings.artifacts_dir) / stem
            images_dir = doc_dir / "images"
            uri_prefix = (
                f"{settings.artifacts_url_prefix.strip().rstrip('/')}/{stem}/images/"
            )
            from docling_core.types.doc import ImageRefMode

            export_kwargs = dict(
                image_mode=ImageRefMode.REFERENCED,
                image_dir=images_dir,
                image_uri_prefix=uri_prefix,
            )

        try:
            markdown = result.document.export_to_markdown(**export_kwargs)
        except Exception as e:  # pragma: no cover — defensive
            raise RuntimeError(f"docling produced no markdown: {e}") from e

        images: list[SavedImage] = []
        if images_dir is not None and uri_prefix is not None and images_dir.is_dir():
            images = [
                SavedImage(filename=p.name, uri=f"{uri_prefix}{p.name}", path=p)
                for p in sorted(images_dir.glob("*.png"))
            ]
            if not images:
                # Text-only document: Docling still created the images
                # directory; prune the empty tree so artifacts only
                # contain documents that actually produced files.
                try:
                    images_dir.rmdir()
                    if doc_dir is not None and not any(doc_dir.iterdir()):
                        doc_dir.rmdir()
                except OSError:
                    pass

        metadata: dict[str, Any] = {
            "mime_type": mime,
            "profile": profile,
            "images_count": len(images),
        }

        # Page count: Docling stores per-page artifacts under .pages.
        pages = getattr(result.document, "pages", None)
        if pages is not None:
            try:
                metadata["page_count"] = len(pages)
            except TypeError:
                metadata["page_count"] = None
        else:
            metadata["page_count"] = None

        # Best-effort table count — Docling lists TableItems on the doc.
        tables = getattr(result.document, "tables", None)
        if tables is not None:
            try:
                metadata["tables_count"] = len(tables)
            except TypeError:
                metadata["tables_count"] = None

        # Pages that actually passed through the OCR stage:
        # - the VLM pipeline has no OCR stage; DOCX/PPTX/HTML run
        #   Docling's model-free SimplePipeline; native PDFs never OCR;
        # - standard selective OCR covers at least the text-layer-less
        #   pages the probe found; standalone raster images are OCR'd
        #   page by page. An unprobeable PDF leaves the count unknown.
        if profile == PROFILE_VLM:
            metadata["ocr_pages"] = 0
        elif mime in _IMAGE_MIMES:
            # The native profile still routes images through Standard.
            metadata["ocr_pages"] = metadata["page_count"] or 0
        elif mime == "application/pdf":
            if profile == PROFILE_NATIVE:
                metadata["ocr_pages"] = 0
            elif probe is not None:
                metadata["ocr_pages"] = probe.scanned_count
            else:
                metadata["ocr_pages"] = None
        else:
            metadata["ocr_pages"] = 0

        # doc_kind comes from the cheap text-layer probe for PDFs; a
        # probe that could not read the file is reported as "unknown"
        # rather than silently dropped.
        if mime == "application/pdf":
            metadata["doc_kind"] = probe.kind if probe is not None else "unknown"
            if probe is not None:
                metadata["scanned_pages"] = probe.scanned_count

        # Best-effort provenance — Docling puts these on the document's
        # ``props`` when available.
        props = getattr(result.document, "props", None)
        if props is not None:
            for key in ("title", "author"):
                value = getattr(props, key, None)
                if value:
                    metadata[key] = str(value)

        return ParsedDocument(
            markdown=markdown or "",
            metadata=metadata,
            images=images,
            artifacts_dir=doc_dir if images else None,
        )


# ---- converter construction -------------------------------------------


def _standard_pdf_options(device: str, save_images: bool, images_scale: float, langs: list[str]):
    """One fresh Standard-pipeline options object (PDF or IMAGE format).

    Selective OCR: ``OcrMode.DEFAULT`` makes Docling OCR only layout
    clusters not already covered by PDF text cells, so digital pages
    pay no OCR cost while scanned pages are recognized automatically.
    RapidOCR's ``onnxruntime`` backend is the one that can reach the
    GPU through our onnxruntime-gpu install (see
    :mod:`vector_service.core.cuda_dlls`).
    """
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        OcrMode,
        PdfPipelineOptions,
        RapidOcrOptions,
    )

    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
        do_ocr=True,
        ocr_options=RapidOcrOptions(
            backend="onnxruntime",
            lang=list(langs),
            mode=OcrMode.DEFAULT,
        ),
        do_table_structure=True,
        generate_picture_images=save_images,
        images_scale=images_scale,
    )


def _build_format_options(profile: str) -> dict:
    """Docling ``format_options`` dict for one concrete profile."""
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import (
        ImageFormatOption,
        NativePdfFormatOption,
        PdfFormatOption,
    )

    settings = get_settings().parser
    device = settings.device
    save_images = settings.save_images
    images_scale = settings.images_scale
    langs = settings.ocr_langs

    options: dict = {}

    if profile == PROFILE_STANDARD:
        # Distinct option instances per format: Docling hashes pipeline
        # options in its per-format pipeline cache.
        options[InputFormat.PDF] = PdfFormatOption(
            pipeline_options=_standard_pdf_options(
                device, save_images, images_scale, langs
            )
        )
        options[InputFormat.IMAGE] = ImageFormatOption(
            pipeline_options=_standard_pdf_options(
                device, save_images, images_scale, langs
            )
        )
    elif profile == PROFILE_NATIVE:
        from docling.datamodel.pipeline_options import NativePdfPipelineOptions

        # NativePdfFormatOption wires NativePdfPipeline + the threaded
        # docling-parse backend itself; include_bitmap_images follows
        # generate_picture_images via its model validator.
        options[InputFormat.PDF] = NativePdfFormatOption(
            pipeline_options=NativePdfPipelineOptions(
                generate_picture_images=save_images,
                images_scale=images_scale,
            )
        )
        # The native backend is PDF-only — standalone images still get
        # the full OCR treatment even under the native profile.
        options[InputFormat.IMAGE] = ImageFormatOption(
            pipeline_options=_standard_pdf_options(
                device, save_images, images_scale, langs
            )
        )
    elif profile == PROFILE_VLM:
        from docling.datamodel.pipeline_options import AcceleratorOptions
        from docling.pipeline.vlm_pipeline import (
            VlmConvertOptions,
            VlmPipeline,
            VlmPipelineOptions,
        )

        def _vlm_options() -> VlmPipelineOptions:
            return VlmPipelineOptions(
                accelerator_options=AcceleratorOptions(device=device),
                vlm_options=VlmConvertOptions.from_preset(settings.vlm_preset),
                generate_picture_images=save_images,
                images_scale=images_scale,
            )

        options[InputFormat.PDF] = PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=_vlm_options()
        )
        options[InputFormat.IMAGE] = ImageFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=_vlm_options()
        )
    else:  # pragma: no cover — _resolve_profile guards the public surface
        raise ValueError(f"cannot build converter for profile {profile!r}")

    # DOCX / PPTX / HTML (and every other format) are intentionally
    # absent: DocumentConverter fills gaps with its built-in defaults,
    # which route office formats through the model-free SimplePipeline.
    return options


def _preinitialize_pipelines(profile: str, converter: Any) -> None:
    """Force pipeline (re: model) construction right after converter build.

    ``DocumentConverter.initialize_pipeline(format)`` builds and caches
    the pipeline — the point where Docling loads its layout/table/VLM
    weights; without it a "warm" converter in recent Docling is only an
    options object and every first parse still pays cold-start. PDF is
    pre-initialized for every profile; IMAGE only where that converter
    actually owns heavy models (the ``native`` converter routes images
    to the standard pipeline, which native warm should not pull in).

    No-op on converters without the method (test doubles, older
    Docling) so the basic converter-cache warm keeps working.
    """
    if converter is None:
        return
    initialize = getattr(converter, "initialize_pipeline", None)
    if not callable(initialize):
        return
    from docling.datamodel.base_models import InputFormat

    formats = [InputFormat.PDF]
    if profile != PROFILE_NATIVE:
        formats.append(InputFormat.IMAGE)
    for doc_format in formats:
        initialize(doc_format)


# ---- process-wide singleton -------------------------------------------


_parser_singleton: DoclingParser | None = None
_parser_singleton_lock = threading.Lock()


def get_docling_parser() -> DoclingParser:
    """Return the process-wide :class:`DoclingParser`.

    Both the HTTP routes and the lifespan warmup hit one cached set of
    converters: Docling's layout/OCR models cost ~10-30s to build, so a
    fresh parser per request made every upload pay cold-start cost even
    after warmup.
    """
    global _parser_singleton
    with _parser_singleton_lock:
        if _parser_singleton is None:
            _parser_singleton = DoclingParser()
        return _parser_singleton


# ---- helpers ----------------------------------------------------------


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
    return _MIME_TO_SUFFIX.get(mime, ".bin")


def _safe_stem(stem: str | None) -> str:
    """Sanitize the per-document artifacts subdirectory name.

    Only ``[A-Za-z0-9._-]`` survives, so caller-supplied ids cannot
    traverse out of ``artifacts_dir``. Empty/None/fully-stripped input
    yields a random id.
    """
    if stem:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._")
        if cleaned:
            return cleaned
    return uuid.uuid4().hex


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


_MIME_TO_SUFFIX: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/html": ".html",
    "application/xhtml+xml": ".xhtml",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "text/markdown": ".md",
    "text/plain": ".txt",
}
