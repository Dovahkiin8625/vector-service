"""Unit tests for ``DoclingParser``.

These tests gate on the optional ``docling`` dependency — if
``docling`` isn't installed, every test is skipped with an
explanatory message rather than failing. The mock backend is
injected via monkeypatch so we never instantiate a real
``DocumentConverter`` (which would download hundreds of MB of
model weights on first use).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import docling  # noqa: F401  -- presence check
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False


pytestmark = pytest.mark.skipif(
    not HAS_DOCLING,
    reason="docling not installed; install 'docling>=2.0' to run these tests",
)


class _FakeDocument:
    """Minimal stand-in for ``docling_core.types.Document``."""

    def __init__(self, markdown: str, page_count: int = 1, title: str | None = None):
        self._markdown = markdown
        self.pages = list(range(page_count))
        if title:
            from types import SimpleNamespace
            self.props = SimpleNamespace(title=title, author=None)
        else:
            self.props = None

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, doc: _FakeDocument):
        self.document = doc


class _FakeConverter:
    """Drop-in for ``docling.document_converter.DocumentConverter``."""

    def __init__(self):
        self.calls: list[Path | str] = []

    def convert(self, source):
        self.calls.append(source)
        # Echo back a tiny markdown document with a known page count.
        return _FakeResult(
            _FakeDocument(
                markdown=f"# Converted\n\nfrom {source}\n",
                page_count=3,
                title="Hello",
            )
        )


def test_parser_dispatches_to_docling_for_pdf(monkeypatch, tmp_path):
    from vector_service.parsers.docling_parser import DoclingParser

    converter = _FakeConverter()
    # Inject the fake converter BEFORE calling ``load()`` — DoclingParser
    # constructs its own inside ``load()``.
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        lambda *a, **kw: converter,
    )
    # parse() guards on path.exists(); create the file and chdir so the
    # relative-path assertion below keeps its original intent.
    (tmp_path / "hello.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.chdir(tmp_path)

    parser = DoclingParser()
    parser.load()

    parsed = parser.parse(Path("hello.pdf"))
    assert "# Converted" in parsed.markdown
    assert parsed.metadata["page_count"] == 3
    assert parsed.metadata["title"] == "Hello"
    assert parsed.metadata["mime_type"] == "application/pdf"
    assert isinstance(converter.calls[0], str) and converter.calls[0].endswith("hello.pdf")


def test_parser_handles_bytes_input(monkeypatch):
    """``parse_bytes`` spills to a tempfile and reads it back via the
    converter; verify the round-trip without ever touching disk."""
    from vector_service.parsers.docling_parser import DoclingParser

    converter = _FakeConverter()
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        lambda *a, **kw: converter,
    )

    parser = DoclingParser()
    parser.load()

    import asyncio
    parsed = asyncio.run(
        parser.parse_bytes(b"%PDF-1.4 ...", "application/pdf"),
    )
    assert "# Converted" in parsed.markdown
    assert parsed.metadata["mime_type"] == "application/pdf"
    # The converter received a path on disk.
    assert len(converter.calls) == 1


def test_load_is_idempotent(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    inits: list[int] = []

    def _factory(*a, **kw):
        inits.append(1)
        return _FakeConverter()

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _factory,
    )

    parser = DoclingParser()
    parser.load()
    parser.load()
    parser.load()
    assert len(inits) == 1


def test_unload_releases_converter(monkeypatch):
    from vector_service.parsers.docling_parser import DoclingParser

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _FakeConverter,
    )
    parser = DoclingParser()
    parser.load()
    assert parser._converter is not None
    parser.unload()
    assert parser._converter is None
    # Idempotent: a second unload is a no-op.
    parser.unload()
    assert parser._converter is None


def test_parse_bytes_tempfile_cleanup(monkeypatch):
    """``parse_bytes`` must delete its tempfile even when the
    converter raises."""
    from vector_service.parsers import docling_parser as mod

    class _BoomConverter:
        def convert(self, source):
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _BoomConverter,
    )
    parser = mod.DoclingParser()
    parser.load()

    import asyncio
    with pytest.raises(RuntimeError, match="conversion failed"):
        asyncio.run(parser.parse_bytes(b"data", "application/pdf"))


def test_can_handle_claims_only_docling_mimes():
    from vector_service.parsers.docling_parser import DoclingParser

    p = DoclingParser()
    assert p.can_handle("application/pdf")
    assert p.can_handle("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert p.can_handle("text/html")
    assert not p.can_handle("text/plain")
    assert not p.can_handle("text/markdown")


def test_guess_mime():
    from vector_service.parsers.docling_parser import guess_mime

    assert guess_mime(Path("/x/file.pdf")) == "application/pdf"
    assert guess_mime(Path("/x/file.docx")) == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert guess_mime(Path("/x/file.html")) == "text/html"
    assert guess_mime(Path("/x/file.md")) == "text/markdown"
    assert guess_mime(Path("/x/file.unknown")) == "application/octet-stream"


def test_parser_failure_raises_runtime_error(monkeypatch):
    """If the converter raises, ``parse_bytes`` re-raises as a
    ``RuntimeError`` so the route layer can map to 500."""
    from vector_service.parsers.docling_parser import DoclingParser

    class _BoomConverter:
        def convert(self, source):
            raise RuntimeError("docling exploded")

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _BoomConverter,
    )
    parser = DoclingParser()
    parser.load()

    import asyncio
    with pytest.raises(RuntimeError):
        asyncio.run(parser.parse_bytes(b"%PDF", "application/pdf"))


# -----------------------------------------------------------------------
# Process-wide singleton shared by routes + lifespan warmup
# -----------------------------------------------------------------------


def test_get_docling_parser_returns_same_instance(monkeypatch):
    """Routes and the lifespan warmup must hit ONE converter, otherwise
    every request rebuilds Docling's models (~10s cold start each)."""
    from vector_service.parsers import docling_parser as mod

    monkeypatch.setattr(mod, "_parser_singleton", None)
    a = mod.get_docling_parser()
    b = mod.get_docling_parser()
    assert a is b


def test_lifespan_warmup_uses_singleton_getter(monkeypatch):
    """``VS_PARSER__AUTO_LOAD=true`` warmup must call
    ``get_docling_parser()`` and publish THAT instance as
    ``app.state.parser`` — a locally constructed parser would be a
    second converter that routes never use, wasting the warmup."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vector_service.core import lifespan as lifespan_mod
    from vector_service.core.config import get_settings

    settings = get_settings()
    # Keep every other family cold so the lifespan run exercises the
    # parser branch only.
    settings.embedding_auto_load = False
    settings.reranker.auto_load = False
    settings.image_embedding.auto_load = False
    settings.multimodal_embedding.auto_load = False
    settings.parser.auto_load = True

    class _RecordingParser:
        def __init__(self):
            self.load_called = 0

        def load(self):
            self.load_called += 1

    parser = _RecordingParser()
    getter_calls: list = []
    monkeypatch.setattr(
        lifespan_mod,
        "get_docling_parser",
        lambda: (getter_calls.append(1) or parser),
    )
    fake_store = type("S", (), {
        "backend_name": "fake",
        "uri": "",
        "_ensure_connected": lambda self: None,
        "close": lambda self: None,
        "list_databases": lambda self: [],
    })()
    monkeypatch.setattr(lifespan_mod, "build_store", lambda s: fake_store)

    app = FastAPI(lifespan=lifespan_mod.lifespan)
    try:
        with TestClient(app):
            assert getter_calls == [1]
            assert parser.load_called == 1
            assert app.state.parser is parser
    finally:
        settings.parser.auto_load = False


# -----------------------------------------------------------------------
# Per-page progress callbacks
# -----------------------------------------------------------------------


class _FakeProgressPage:
    def __init__(self, page_no):
        self.page_no = page_no


class _FakeProgressInput:
    def __init__(self, page_count):
        self.page_count = page_count


class _FakeProgressConvRes:
    def __init__(self, page_count):
        self.input = _FakeProgressInput(page_count)


class _FakePaginatedPipeline:
    """Stand-in for docling's PaginatedPipeline: ``_apply_on_pages`` is
    a generator that must be exhausted by the wrapper too."""

    def _apply_on_pages(self, conv_res, pages):
        yield from pages


def test_wrap_pipeline_fires_callback_per_completed_page():
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_pipeline,
    )

    pipeline = _wrap_pipeline(_FakePaginatedPipeline())
    pages = [_FakeProgressPage(1), _FakeProgressPage(2), _FakeProgressPage(3)]
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        out = list(pipeline._apply_on_pages(_FakeProgressConvRes(3), pages))
    finally:
        _current_progress.reset(token)

    assert len(out) == 3  # pages still flow through untouched
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_wrap_pipeline_is_idempotent():
    """Wrapping the shared (cached) pipeline for every convert() must
    not stack callbacks."""
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_pipeline,
    )

    pipeline = _FakePaginatedPipeline()
    _wrap_pipeline(pipeline)
    _wrap_pipeline(pipeline)  # second convert() call
    pages = [_FakeProgressPage(1), _FakeProgressPage(2)]
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        list(pipeline._apply_on_pages(_FakeProgressConvRes(2), pages))
    finally:
        _current_progress.reset(token)

    assert seen == [(1, 2), (2, 2)]


def test_wrap_pipeline_without_callback_just_passes_pages_through():
    """No progress listener (classic JSON route) must be a no-op."""
    from vector_service.parsers.docling_parser import _wrap_pipeline

    pipeline = _wrap_pipeline(_FakePaginatedPipeline())
    pages = [_FakeProgressPage(1), _FakeProgressPage(2)]
    out = list(pipeline._apply_on_pages(_FakeProgressConvRes(2), pages))
    assert [p.page_no for p in out] == [1, 2]


def test_wrap_pipeline_swallows_callback_errors():
    """A broken progress sink must never abort the actual conversion."""
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_pipeline,
    )

    def _boom(_done, _total):
        raise RuntimeError("progress sink broke")

    pipeline = _wrap_pipeline(_FakePaginatedPipeline())
    token = _current_progress.set(_boom)
    try:
        out = list(
            pipeline._apply_on_pages(
                _FakeProgressConvRes(1), [_FakeProgressPage(1)]
            )
        )
    finally:
        _current_progress.reset(token)
    assert len(out) == 1


# -----------------------------------------------------------------------
# Threaded PDF pipeline (docling >= 2.12 default StandardPdfPipeline)
#
# That pipeline bypasses PaginatedPipeline._apply_on_pages entirely:
# six worker stages communicate over queues and ``_build_document``
# drains completed pages from ``RunContext.output_queue.get_batch``.
# The wrapper reports one tick per unique drained page_no.
# -----------------------------------------------------------------------


class _FakeThreadedItem:
    def __init__(self, page_no, page_count, *, is_failed=False):
        self.page_no = page_no
        self.is_failed = is_failed
        self.error = None
        self.conv_res = _FakeProgressConvRes(page_count)
        self.run_id = 1
        self.payload = object()


class _FakeOutputQueue:
    """Faithful stand-in for docling's ``ThreadedQueue``: it uses
    ``__slots__`` (so assigning ``get_batch`` raises AttributeError) and
    exposes the close/closed surface the drain loop relies on."""

    __slots__ = ("_batches", "_closed")

    def __init__(self, batches):
        self._batches = list(batches)
        self._closed = False

    def get_batch(self, size, timeout=None):
        if self._batches:
            return self._batches.pop(0)
        return []

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed


class _FakeRunContext:
    def __init__(self, batches):
        self.output_queue = _FakeOutputQueue(batches)


class _FakeThreadedPdfPipeline:
    """Mirrors the surface the wrapper touches on StandardPdfPipeline:
    a per-run ``_create_run_ctx()`` factory."""

    def __init__(self, batches):
        self._batches = batches
        self.calls = 0

    def _create_run_ctx(self):
        self.calls += 1
        return _FakeRunContext(self._batches)


def test_wrap_threaded_pipeline_fires_callback_per_drained_page():
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_threaded_pdf_pipeline,
    )

    # Pages can complete out of order and arrive in several drains.
    pipeline = _wrap_threaded_pdf_pipeline(
        _FakeThreadedPdfPipeline([
            [_FakeThreadedItem(2, 3), _FakeThreadedItem(1, 3)],
            [_FakeThreadedItem(3, 3)],
        ])
    )
    ctx = pipeline._create_run_ctx()
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        batch1 = ctx.output_queue.get_batch(32, timeout=0.05)
        batch2 = ctx.output_queue.get_batch(32, timeout=0.05)
        batch3 = ctx.output_queue.get_batch(32, timeout=0.05)
    finally:
        _current_progress.reset(token)

    assert [it.page_no for it in batch1] == [2, 1]  # untouched
    assert [it.page_no for it in batch2] == [3]
    assert batch3 == []
    # Counts are unique-page based, so ticks stay monotonic even out
    # of order; total comes from the item's conv_res.input.page_count.
    assert seen == [(2, 3), (3, 3)]


def test_wrap_threaded_pipeline_counts_failed_pages_as_done():
    """A page that errors in a stage is still drained once; the bar must
    not get stuck short of total."""
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_threaded_pdf_pipeline,
    )

    pipeline = _wrap_threaded_pdf_pipeline(
        _FakeThreadedPdfPipeline([
            [_FakeThreadedItem(1, 2)],
            [_FakeThreadedItem(2, 2, is_failed=True)],
        ])
    )
    ctx = pipeline._create_run_ctx()
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        ctx.output_queue.get_batch(32)
        ctx.output_queue.get_batch(32)
    finally:
        _current_progress.reset(token)
    assert seen == [(1, 2), (2, 2)]


def test_wrap_threaded_pipeline_each_run_starts_at_zero():
    """RunContext (and the completed-set) is rebuilt per convert call so
    a second document doesn't start its progress at the previous count."""
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_threaded_pdf_pipeline,
    )

    inner = _FakeThreadedPdfPipeline(
        [[_FakeThreadedItem(1, 1)]]
    )
    pipeline = _wrap_threaded_pdf_pipeline(inner)
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        ctx1 = pipeline._create_run_ctx()
        ctx1.output_queue.get_batch(32)
        # Factory returns another fresh run with its own batch script —
        # emulate the real pipeline by swapping the run script.
        inner._batches = [[_FakeThreadedItem(1, 2)], [_FakeThreadedItem(2, 2)]]
        ctx2 = pipeline._create_run_ctx()
        ctx2.output_queue.get_batch(32)
        ctx2.output_queue.get_batch(32)
    finally:
        _current_progress.reset(token)
    assert seen == [(1, 1), (1, 2), (2, 2)]


def test_wrap_threaded_pipeline_is_idempotent():
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_threaded_pdf_pipeline,
    )

    pipeline = _FakeThreadedPdfPipeline([[_FakeThreadedItem(1, 1)]])
    _wrap_threaded_pdf_pipeline(pipeline)
    _wrap_threaded_pdf_pipeline(pipeline)
    ctx = pipeline._create_run_ctx()
    seen: list[tuple] = []
    token = _current_progress.set(lambda done, total: seen.append((done, total)))
    try:
        ctx.output_queue.get_batch(32)
    finally:
        _current_progress.reset(token)
    assert seen == [(1, 1)]


def test_wrap_threaded_pipeline_without_listener_is_passthrough():
    from vector_service.parsers.docling_parser import _wrap_threaded_pdf_pipeline

    pipeline = _wrap_threaded_pdf_pipeline(
        _FakeThreadedPdfPipeline([[_FakeThreadedItem(1, 2)]])
    )
    ctx = pipeline._create_run_ctx()
    assert [it.page_no for it in ctx.output_queue.get_batch(32)] == [1]


def test_wrap_threaded_pipeline_swallows_callback_errors():
    from vector_service.parsers.docling_parser import (
        _current_progress,
        _wrap_threaded_pdf_pipeline,
    )

    def _boom(_done, _total):
        raise RuntimeError("progress sink broke")

    pipeline = _wrap_threaded_pdf_pipeline(
        _FakeThreadedPdfPipeline([[_FakeThreadedItem(1, 1)]])
    )
    ctx = pipeline._create_run_ctx()
    token = _current_progress.set(_boom)
    try:
        batch = ctx.output_queue.get_batch(32)
    finally:
        _current_progress.reset(token)
    assert [it.page_no for it in batch] == [1]


def test_wrap_threaded_pipeline_wraps_queue_not_its_methods():
    """Regression: real ``ThreadedQueue`` uses ``__slots__``, so
    ``out_q.get_batch = ...`` raised AttributeError inside
    ``_build_document`` and failed EVERY page ("Pipeline
    StandardPdfPipeline failed"). The wrapper must install a proxy on the
    RunContext instead of mutating the queue, and delegate lifecycle
    (close/closed) so the drain loop's shutdown keeps working."""
    from vector_service.parsers.docling_parser import _wrap_threaded_pdf_pipeline

    pipeline = _wrap_threaded_pdf_pipeline(
        _FakeThreadedPdfPipeline([[_FakeThreadedItem(1, 1)]])
    )
    ctx = pipeline._create_run_ctx()

    # The slots queue itself is untouched; a proxy sits on the ctx.
    assert type(ctx.output_queue).__name__ == "_ProgressQueueProxy"
    assert not ctx.output_queue.closed
    ctx.output_queue.close()
    assert ctx.output_queue.closed


def test_wrap_threaded_pipeline_skips_pipelines_without_run_ctx():
    from vector_service.parsers.docling_parser import _wrap_threaded_pdf_pipeline

    plain = object()
    assert _wrap_threaded_pdf_pipeline(plain) is plain


def test_progress_converter_wraps_threaded_pdf_pipeline():
    """The converter subclass must wrap the pipeline returned by the
    real (or fake) DocumentConverter — including the threaded PDF
    pipeline that docling 2.12+ selects by default."""
    from vector_service.parsers.docling_parser import _build_progress_converter

    class _ThreadedLike(_FakeThreadedPdfPipeline):
        pass

    threaded = _ThreadedLike([[_FakeThreadedItem(1, 1)]])

    class _Converter:
        def _get_pipeline(self, doc_format):
            return threaded

    import docling.document_converter as dc_mod
    original = dc_mod.DocumentConverter
    try:
        dc_mod.DocumentConverter = _Converter
        converter = _build_progress_converter()
    finally:
        dc_mod.DocumentConverter = original

    out = converter._get_pipeline("pdf")
    assert out is threaded
    assert getattr(threaded, "_vs_progress_wrapped", False) is True


def test_parse_bytes_invokes_on_progress_during_conversion(monkeypatch):
    """``on_progress`` must be visible to the converter even though
    conversion runs on an executor thread (contextvars do not cross
    threads automatically)."""
    import asyncio

    from vector_service.parsers.docling_parser import (
        DoclingParser,
        _current_progress,
    )

    class _ProbeConverter:
        def convert(self, source):
            cb = _current_progress.get(None)
            assert cb is not None, "progress callback not set in worker thread"
            cb(2, 5)
            return _FakeResult(_FakeDocument(markdown="ok", page_count=5))

    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        lambda *a, **kw: _ProbeConverter(),
    )
    parser = DoclingParser()
    parser.load()

    seen: list[tuple] = []
    parsed = asyncio.run(
        parser.parse_bytes(
            b"%PDF", "application/pdf",
            on_progress=lambda done, total: seen.append((done, total)),
        )
    )
    assert parsed.markdown == "ok"
    assert seen == [(2, 5)]


def test_markdown_parser_accepts_progress_callback_but_emits_none():
    """Text formats share the parser ABC; they must accept the same
    ``on_progress`` kwarg without producing any page events."""
    import asyncio

    from vector_service.parsers.markdown_parser import MarkdownParser

    seen: list[tuple] = []
    parsed = asyncio.run(
        MarkdownParser().parse_bytes(
            b"hello", "text/plain",
            on_progress=lambda done, total: seen.append((done, total)),
        )
    )
    assert parsed.markdown == "hello"
    assert seen == []
