from prometheus_client import parser

from vector_service.core.metrics import (
    EMBEDDING_DURATION_SECONDS,
    EMBEDDING_TOKENS_TOTAL,
    EMBEDDING_REQUESTS_TOTAL,
    STORE_OP_DURATION_SECONDS,
    STORE_COLLECTIONS,
    STORE_VECTORS_TOTAL,
    MODEL_LOADED,
    VS_INFO,
    render_metrics,
)


def test_required_metrics_exist():
    """All metrics appear in rendered exposition format.
    Note: We check raw text (with _total suffix for Counters per OpenMetrics
    convention) rather than parsed family names. The parser removes the _total
    suffix when reading back; the rendered text keeps it for Counters.
    """
    out = render_metrics()
    expected_substrings = [
        "vs_embedding_duration_seconds",
        "vs_embedding_tokens_total",
        "vs_embedding_requests_total",
        "vs_store_operation_duration_seconds",
        "vs_store_collections",
        "vs_store_vectors_total",
        "vs_model_loaded",
        "vs_info",
    ]
    missing = [s for s in expected_substrings if s not in out]
    assert not missing, f"missing metric substrings in rendered output: {missing}"


def test_embedding_duration_has_required_labels():
    labels = EMBEDDING_DURATION_SECONDS._labelnames
    assert "model" in labels
    assert "status" in labels


def test_render_metrics_is_parseable():
    EMBEDDING_DURATION_SECONDS.labels(model="x", status="ok").observe(0.1)
    out = render_metrics()
    parsed = list(parser.text_string_to_metric_families(out))
    assert any(f.name == "vs_embedding_duration_seconds" for f in parsed)