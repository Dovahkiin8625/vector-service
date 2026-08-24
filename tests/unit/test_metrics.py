from prometheus_client import parser

from vector_service.core.metrics import (
    EMBEDDING_DURATION_SECONDS,
    EMBEDDING_TOKENS_TOTAL,
    STORE_OP_DURATION_SECONDS,
    STORE_COLLECTIONS,
    MODEL_LOADED,
    VS_INFO,
    render_metrics,
)


def test_required_metrics_exist():
    names = {
        EMBEDDING_DURATION_SECONDS._name,
        EMBEDDING_TOKENS_TOTAL._name,
        STORE_OP_DURATION_SECONDS._name,
        STORE_COLLECTIONS._name,
        MODEL_LOADED._name,
        VS_INFO._name,
    }
    assert names == {
        "vs_embedding_duration_seconds",
        "vs_embedding_tokens_total",
        "vs_store_operation_duration_seconds",
        "vs_store_collections",
        "vs_model_loaded",
        "vs_info",
    }


def test_embedding_duration_has_required_labels():
    labels = EMBEDDING_DURATION_SECONDS._labelnames
    assert "model" in labels
    assert "status" in labels


def test_render_metrics_is_parseable():
    EMBEDDING_DURATION_SECONDS.labels(model="x", status="ok").observe(0.1)
    out = render_metrics()
    # At least contains our metric name
    assert "vs_embedding_duration_seconds" in out
    # Can be parsed by prometheus_client
    parsed = list(parser.text_string_to_metric_families(out))
    assert any(f.name == "vs_embedding_duration_seconds" for f in parsed)
