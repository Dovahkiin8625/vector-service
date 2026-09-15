"""Regression tests for batch 2B shared helpers (``embeddings._common``).

Pin that the new ``_common.py`` helpers behave correctly so future
backends can rely on them.
"""
from __future__ import annotations

import pytest


def test_dir_has_model_true(tmp_path):
    """When the directory contains a known marker, return True."""
    (tmp_path / "config.json").write_text("{}")
    from vector_service.embeddings._common import dir_has_model
    assert dir_has_model(str(tmp_path)) is True


def test_dir_has_model_false_for_empty_dir(tmp_path):
    from vector_service.embeddings._common import dir_has_model
    assert dir_has_model(str(tmp_path)) is False


def test_dir_has_model_false_for_missing_path(tmp_path):
    from vector_service.embeddings._common import dir_has_model
    assert dir_has_model(str(tmp_path / "nope")) is False


def test_dir_has_model_accepts_custom_markers(tmp_path):
    (tmp_path / "custom.marker").write_text("")
    from vector_service.embeddings._common import dir_has_model
    # Default markers don't include ``custom.marker`` so default
    # call returns False; custom markers list picks it up.
    assert dir_has_model(str(tmp_path)) is False
    assert dir_has_model(str(tmp_path), markers=("custom.marker",)) is True


def test_release_cuda_cache_is_silent_when_torch_missing(monkeypatch):
    """If torch isn't installed (or fails to import), the helper
    must swallow the error and return None. We arrange ``torch`` to
    raise on import and assert the call doesn't propagate.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated torch absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from vector_service.embeddings._common import release_cuda_cache
    # Must not raise even when torch import fails.
    assert release_cuda_cache() is None


# ---- batch 4: glob markers + resolve_device ---------------------------


def test_dir_has_model_glob_markers(tmp_path):
    """Markers containing ``*`` are treated as glob patterns.

    OpenCLIP and Chinese-CLIP use ``*.bin`` / ``*.safetensors`` glob
    markers; the helper must resolve them via ``Path.glob`` instead
    of an exact-name intersection.
    """
    from vector_service.embeddings._common import dir_has_model
    (tmp_path / "model.safetensors").write_text("")
    assert dir_has_model(str(tmp_path), markers=("*.safetensors",)) is True
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "pytorch_model.bin").write_text("")
    assert dir_has_model(str(tmp_path), markers=("*.bin",)) is True


def test_dir_has_model_mixes_glob_and_exact(tmp_path):
    """A marker list can mix globs and exact filenames."""
    from vector_service.embeddings._common import dir_has_model
    (tmp_path / "config.json").write_text("{}")
    # Exact match via ``config.json`` returns True; the helper should
    # resolve the glob entry even when the exact entry is absent.
    assert dir_has_model(str(tmp_path), markers=("nope.json", "*.safetensors")) is False
    (tmp_path / "config.json").unlink()
    (tmp_path / "model.safetensors").write_text("")
    assert dir_has_model(str(tmp_path), markers=("nope.json", "*.safetensors")) is True


def test_resolve_device_cpu_short_circuits(monkeypatch):
    """``"cpu"`` returns ``"cpu"`` without touching torch.

    This is the test the new contract relies on: the constructor stays
    cheap even when the embed extras aren't installed.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch should not be needed for cpu")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from vector_service.embeddings._common import resolve_device
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_returns_string_without_probe(monkeypatch):
    """``"cuda"`` returns ``"cuda"`` without probing the driver.

    Backends probe lazily at load() so the constructor stays cheap.
    The test pins that resolve_device does not touch torch at all
    when the user explicitly asked for cuda.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch should not be needed for explicit cuda")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from vector_service.embeddings._common import resolve_device
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_when_torch_missing(monkeypatch):
    """``"auto"`` returns ``"cpu"`` when torch cannot be imported."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated torch absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from vector_service.embeddings._common import resolve_device
    assert resolve_device("auto") == "cpu"
