"""Prometheus metric definitions."""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Global registry, all metrics share
REGISTRY = CollectorRegistry(auto_describe=True)

EMBEDDING_DURATION_SECONDS = Histogram(
    "vs_embedding_duration_seconds",
    "Embedding request duration in seconds",
    labelnames=("model", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

EMBEDDING_TOKENS_TOTAL = Counter(
    "vs_embedding_tokens_total",
    "Total tokens processed by embedder",
    labelnames=("model",),
    registry=REGISTRY,
)

EMBEDDING_REQUESTS_TOTAL = Counter(
    "vs_embedding_requests_total",
    "Total embedding requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

STORE_OP_DURATION_SECONDS = Histogram(
    "vs_store_operation_duration_seconds",
    "Vector store operation duration in seconds",
    labelnames=("op", "backend", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

STORE_COLLECTIONS = Gauge(
    "vs_store_collections",
    "Number of collections in the store",
    labelnames=("backend",),
    registry=REGISTRY,
)

STORE_VECTORS_TOTAL = Counter(
    "vs_store_vectors_total",
    "Total vector write/delete operations",
    labelnames=("op", "backend"),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "vs_model_loaded",
    "1 if the embedding model is loaded, 0 otherwise",
    labelnames=("model", "device"),
    registry=REGISTRY,
)

VS_INFO = Gauge(
    "vs_info",
    "Static build/runtime information",
    labelnames=("version", "embedding_backend", "vector_store_backend"),
    registry=REGISTRY,
)


def render_metrics() -> str:
    """Render Prometheus exposition format text."""
    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
