"""Unit tests for ``RerankerSettings`` validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.core.config import RerankerSettings


def test_defaults_when_only_backend_given():
    s = RerankerSettings(backend="bge-reranker-v2-m3")
    assert s.device == "auto"
    assert s.batch_size == 32
    assert s.max_documents_per_request == 256
    assert s.top_n_default == 10
    assert s.max_top_n == 64
    assert s.download_source == "modelscope"


def test_backend_required():
    # ``RerankerSettings`` is a ``BaseSettings`` subclass that reads
    # ``VS_RERANKER__BACKEND`` from the env / ``.env`` file. To prove
    # the field is genuinely required, build with an explicit ``None``
    # which pydantic treats as "missing".
    with pytest.raises(ValidationError):
        RerankerSettings(backend=None)  # type: ignore[call-arg]


def test_top_n_default_cannot_exceed_max():
    with pytest.raises(ValidationError):
        RerankerSettings(
            backend="bge-reranker-v2-m3", top_n_default=100, max_top_n=50
        )


def test_batch_size_bounds():
    with pytest.raises(ValidationError):
        RerankerSettings(backend="bge-reranker-v2-m3", batch_size=0)
    with pytest.raises(ValidationError):
        RerankerSettings(backend="bge-reranker-v2-m3", batch_size=10_000)


def test_device_literal():
    with pytest.raises(ValidationError):
        RerankerSettings(backend="bge-reranker-v2-m3", device="tpu")


def test_download_source_literal():
    with pytest.raises(ValidationError):
        RerankerSettings(backend="bge-reranker-v2-m3", download_source="civitai")
