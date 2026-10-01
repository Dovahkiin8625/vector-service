"""OpenTelemetry tracing setup.

Opt-in via :class:`~vector_service.core.config.TracingSettings` (env
prefix ``VS_TRACING__``). When enabled the SDK is installed as the
global tracer provider; spans then export over OTLP/HTTP to
``endpoint`` (e.g. ``http://collector:4318/v1/traces``). With no
endpoint the provider still records spans in-process (tests attach
their own processors) but exports nothing. When tracing is disabled
the pipeline's spans hit the default proxy and are cheap no-ops.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from vector_service.core.logging import get_logger

log = get_logger(__name__)


def init_tracing(settings: Any) -> TracerProvider | None:
    """Install the global tracer provider; return it (``None`` if disabled)."""
    cfg = settings.tracing
    if not cfg.enabled:
        return None

    resource = Resource.create({"service.name": cfg.service_name})
    sampler = ParentBased(TraceIdRatioBased(cfg.sample_ratio))
    provider = TracerProvider(resource=resource, sampler=sampler)
    if cfg.endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.endpoint))
        )
    trace.set_tracer_provider(provider)
    log.info(
        "tracing_initialized",
        service_name=cfg.service_name,
        endpoint=cfg.endpoint or None,
        sample_ratio=cfg.sample_ratio,
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Flush queued spans and stop the provider."""
    if provider is None:
        return
    provider.force_flush()
    provider.shutdown()
