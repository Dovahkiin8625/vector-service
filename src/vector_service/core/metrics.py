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

IMAGE_EMBEDDING_REQUESTS_TOTAL = Counter(
    "vs_image_embedding_requests_total",
    "Total image embedding requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

IMAGE_EMBEDDING_DURATION_SECONDS = Histogram(
    "vs_image_embedding_duration_seconds",
    "Image embedding request duration in seconds",
    labelnames=("model", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

IMAGE_EMBEDDING_INPUTS_TOTAL = Counter(
    "vs_image_embedding_inputs_total",
    "Total images processed by image embedder",
    labelnames=("model",),
    registry=REGISTRY,
)

MULTIMODAL_EMBEDDING_REQUESTS_TOTAL = Counter(
    "vs_multimodal_embedding_requests_total",
    "Total multimodal embedding requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

MULTIMODAL_EMBEDDING_DURATION_SECONDS = Histogram(
    "vs_multimodal_embedding_duration_seconds",
    "Multimodal embedding request duration in seconds",
    labelnames=("model", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

MULTIMODAL_EMBEDDING_INPUTS_TOTAL = Counter(
    "vs_multimodal_embedding_inputs_total",
    "Total items (text + image) processed by multimodal embedder",
    labelnames=("model",),
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
MODEL_LOADED.labels(kind="image_embedder").set(0)
MODEL_LOADED.labels(kind="multimodal_embedder").set(0)

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


# ---- retrieval pipeline ----------------------------------------------

RETRIEVAL_REQUESTS_TOTAL = Counter(
    "vs_retrieval_requests_total",
    "Total retrieval pipeline runs that reached execution.",
    registry=REGISTRY,
)

RETRIEVAL_CHANNEL_RUNS_TOTAL = Counter(
    "vs_retrieval_channel_runs_total",
    "Recall legs fired, labelled by channel and outcome (ok/empty).",
    labelnames=("channel", "status"),
    registry=REGISTRY,
)

RETRIEVAL_RECALLED_HITS_TOTAL = Counter(
    "vs_retrieval_recalled_hits_total",
    "Raw hits returned at recall, per channel (recall volume).",
    labelnames=("channel",),
    registry=REGISTRY,
)

RETRIEVAL_FUSION_INPUT_TOTAL = Counter(
    "vs_retrieval_fusion_input_total",
    "Unique chunks entering fusion, per fusion method.",
    labelnames=("method",),
    registry=REGISTRY,
)

RETRIEVAL_FUSION_OUTPUT_TOTAL = Counter(
    "vs_retrieval_fusion_output_total",
    "Chunks emitted by fusion, per fusion method. Output/input ratio "
    "is the post-fusion survival rate.",
    labelnames=("method",),
    registry=REGISTRY,
)

RETRIEVAL_CHANNEL_HIT_REQUESTS_TOTAL = Counter(
    "vs_retrieval_channel_hit_requests_total",
    "Requests in which a channel contributed at least one final result "
    "(channel hit rate denominator: vs_retrieval_requests_total).",
    labelnames=("channel",),
    registry=REGISTRY,
)

RETRIEVAL_RERANK_STAGE_DURATION_SECONDS = Histogram(
    "vs_retrieval_rerank_stage_duration_seconds",
    "Rerank stage latency inside the retrieval pipeline.",
    labelnames=("model",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

RETRIEVAL_RERANK_INPUT_CHARS = Histogram(
    "vs_retrieval_rerank_input_chars",
    "Total candidate-text characters fed to the rerank stage.",
    labelnames=("model",),
    buckets=(
        100, 500, 1000, 2500, 5000, 10000, 20000, 50000, 100000,
    ),
    registry=REGISTRY,
)

RETRIEVAL_RERANK_CANDIDATES = Histogram(
    "vs_retrieval_rerank_candidates",
    "Number of candidate chunks fed to the rerank stage.",
    labelnames=("model",),
    buckets=(1, 5, 10, 20, 30, 50, 75, 100),
    registry=REGISTRY,
)


def render_metrics() -> str:
    """Render Prometheus exposition format text."""
    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST