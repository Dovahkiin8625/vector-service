"""Tests for :mod:`vector_service.core.parser_status`.

The Docling converter internals vary by version, so the tests use
small structural fakes standing in for converter → pipeline → stage
attributes — the same duck-typed surface the introspection reads
(``initialized_pipelines``, ``vars(pipeline)``). These tests must
pass without real Docling/torch model weights.
"""
from __future__ import annotations

from vector_service.core import parser_status as ps


# ---------------------------------------------------------------------------
# Structural fakes
# ---------------------------------------------------------------------------


class _FakeOcrOptions:
    backend = "onnxruntime"


class _FakePipelineOptions:
    def __init__(self):
        self.ocr_options = _FakeOcrOptions()


class _FakeOcrStage:
    """RapidOCR stand-in: no torch module, just a wrapper object."""


class _FakePipeline:
    def __init__(self, **stages):
        # pipeline bookkeeping attrs must be ignored as model stages.
        self.pipeline_options = _FakePipelineOptions()
        self.artifacts_path = None
        for name, value in stages.items():
            setattr(self, name, value)


class _FakeConverter:
    def __init__(self, pipelines):
        # Mirrors recent docling: dict keyed by (pipeline_cls, options).
        self.initialized_pipelines = {
            (type(pipeline), f"hash-{i}"): pipeline
            for i, pipeline in enumerate(pipelines)
        }


class _FakeParser:
    def __init__(self, converters):
        self._converters = dict(converters)


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def test_cold_profiles_report_cold_placeholders():
    status = ps.parser_status(_FakeParser({}))
    assert status["backend"] == "docling"
    assert set(status["profiles"]) == {"standard", "native", "vlm"}
    for profile, data in status["profiles"].items():
        assert data["warm"] is False, profile
        assert data["pipeline_count"] == 0
        assert data["components"] == []
        assert data["model_free"] is False
        assert data["torch_param_count"] is None


def test_config_block_exposes_parser_settings():
    status = ps.parser_status(_FakeParser({}))
    config = status["config"]
    assert "device" in config
    assert isinstance(config["ocr_langs"], list)
    assert "images_scale" in config
    assert "vlm_preset" in config


def test_warm_pipeline_lists_ocr_stage_as_resource_hidden():
    pipeline = _FakePipeline(ocr_model=_FakeOcrStage())
    converter = _FakeConverter([pipeline])
    status = ps.parser_status(
        _FakeParser({"standard": converter}),
    )
    data = status["profiles"]["standard"]
    assert data["warm"] is True
    assert data["pipeline_classes"] == ["_FakePipeline"]

    components = data["components"]
    assert len(components) == 1
    ocr = components[0]
    assert ocr["kind"] == "ocr"
    assert ocr["engine"] == "onnxruntime"
    assert ocr["torch"] is False
    # ONNX weights cannot be measured — the flag, not an omission.
    assert ocr["resource_visible"] is False


def test_warm_pipeline_without_models_is_model_free():
    # Native-style pipeline: no model attributes at all.
    converter = _FakeConverter([_FakePipeline()])
    status = ps.parser_status(_FakeParser({"native": converter}))
    data = status["profiles"]["native"]
    assert data["warm"] is True
    assert data["components"] == []
    assert data["model_free"] is True


def test_bookkeeping_attributes_are_not_components():
    """A list attr like build_pipe IS scanned for stages, but plain
    options/path bookkeeping must never appear as components."""
    pipeline = _FakePipeline()
    assert ps._stage_roots(pipeline) == []


def test_build_pipe_list_members_become_stage_roots():
    stage = _FakeOcrStage()
    pipeline = _FakePipeline(build_pipe=[stage])
    roots = ps._stage_roots(pipeline)
    assert len(roots) == 1
    source, root = roots[0]
    assert source == "build_pipe[0]"
    assert root is stage


def test_classify_distinguishes_stage_kinds():
    class _Layout:
        pass

    class _Table:
        pass

    assert ps._classify("layout_model", _Layout(), object()) == "layout"
    assert ps._classify("table_model", _Table(), object()) == "table"
    # Post/preprocessing must not be mislabeled as their parent stage.
    assert ps._classify("layout_postprocessing_model", object(), object()) == "other"


def test_status_never_raises_on_odd_objects():
    """Bare objects / missing attributes must fail-open."""
    status = ps.parser_status(object())
    assert set(status["profiles"]) == {"standard", "native", "vlm"}
    assert ps._converter_pipelines(object()) == []
