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
    labelnames=("op", "backend", "database", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

STORE_DATABASES = Gauge(
    "vs_store_databases",
    "Number of databases in the store",
    labelnames=("backend",),
    registry=REGISTRY,
)

STORE_COLLECTIONS = Gauge(
    "vs_store_collections",
    "Number of collections in the store, broken down by database",
    labelnames=("backend", "database"),
    registry=REGISTRY,
)

STORE_VECTORS_TOTAL = Counter(
    "vs_store_vectors_total",
    "Total vector write/delete operations",
    labelnames=("op", "backend", "database"),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "vs_model_loaded",
    "Model load state, labelled by kind (1 = loaded, 0 = not loaded).",
    labelnames=("kind",),
    registry=REGISTRY,
)
MODEL_LOADED.labels(kind="embedder").set(0)
MODEL_LOADED.labels(kind="reranker").set(0)

VS_INFO = Gauge(
    "vs_info",
    "Static build/runtime information",
    labelnames=("version", "embedding_backend", "vector_store_backend"),
    registry=REGISTRY,
)

RERANK_DURATION_SECONDS = Histogram(
    "vs_rerank_duration_seconds",
    "Rerank latency in seconds (route handler end-to-end).",
    labelnames=("model",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

RERANK_REQUESTS_TOTAL = Counter(
    "vs_rerank_requests_total",
    "Total rerank requests, labelled by model and outcome.",
    labelnames=("model", "status"),  # status in {"received", "ok", "error"}
    registry=REGISTRY,
)


def render_metrics() -> str:
    """Render Prometheus exposition format text."""
    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST