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
