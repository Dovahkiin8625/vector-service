"""Image embedding metrics are registered and labelable."""
from __future__ import annotations

from vector_service.core import metrics


def test_image_embedding_requests_total_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_REQUESTS_TOTAL")
    m = metrics.IMAGE_EMBEDDING_REQUESTS_TOTAL.labels(model="openclip-vit-l-14", status="ok")
    assert m is not None


def test_image_embedding_duration_seconds_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_DURATION_SECONDS")
    h = metrics.IMAGE_EMBEDDING_DURATION_SECONDS.labels(model="openclip-vit-l-14", status="ok")
    h.observe(0.01)


def test_image_embedding_inputs_total_exists():
    assert hasattr(metrics, "IMAGE_EMBEDDING_INPUTS_TOTAL")
    c = metrics.IMAGE_EMBEDDING_INPUTS_TOTAL.labels(model="openclip-vit-l-14")
    c.inc(3)


def test_model_loaded_kind_image_embedder_initialized():
    """The gauge must have a 0 sentinel for kind=image_embedder so
    /readyz reports 'not_loaded' until the lifespan sets it to 1."""
    sample = metrics.MODEL_LOADED.labels(kind="image_embedder")._value.get()  # type: ignore[attr-defined]
    assert sample == 0
