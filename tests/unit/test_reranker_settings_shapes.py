"""Constructor contract for ``CrossEncoderReranker``.

Both call sites (lifespan eager load and the hot-load route
``POST /v1/models/{id}/load``) pass the nested ``RerankerSettings``
block; ``None`` falls back to ``get_settings().reranker``. The tests
below pin that contract and verify the resulting attributes are
sourced from the passed block.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_service.rerankers.cross_encoder import CrossEncoderReranker


class _FakeRerankerSettings:
    """Minimal stand-in: only the attrs the constructor reads."""

    def __init__(self, device: str = "cpu", batch_size: int = 8,
                 max_length: int = 256, model_dir: str = "/tmp/r",
                 auto_download: bool = False, download_source: str = "modelscope",
                 hf_repo: str = "fake/hf", ms_repo: str = "fake/ms"):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_dir = model_dir
        self.auto_download = auto_download
        self.download_source = download_source
        self.hf_repo = hf_repo
        self.ms_repo = ms_repo


def test_constructor_accepts_reranker_settings_block():
    """Hot-load path: factory passes a RerankerSettings block directly."""
    block = _FakeRerankerSettings(device="cuda", batch_size=32, max_length=512)
    r = CrossEncoderReranker(settings=block)  # type: ignore[arg-type]

    assert r._device == "cuda"
    assert r._batch_size == 32
    assert r._max_length == 512
    # Auto-discovery fields round-trip too.
    assert r._auto_download is False
    assert r._download_source == "modelscope"


def test_constructor_attributes_match_passed_block():
    """Defence in depth: every attr the constructor reads must come from the block."""
    block = _FakeRerankerSettings(
        device="cpu", batch_size=64, max_length=1024,
        model_dir="/srv/models/r", auto_download=True,
        download_source="huggingface", hf_repo="org/repo-h", ms_repo="org/repo-m",
    )
    r = CrossEncoderReranker(settings=block)  # type: ignore[arg-type]
    assert r._device == block.device
    assert r._batch_size == block.batch_size
    assert r._max_length == block.max_length
    assert r._model_dir == block.model_dir
    assert r._auto_download is True
    assert r._download_source == "huggingface"
    assert r._hf_repo == "org/repo-h"
    assert r._ms_repo == "org/repo-m"
    # And the impl slot is uninitialised until ``load()`` runs.
    assert r._impl is None
