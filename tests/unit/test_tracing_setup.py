"""OpenTelemetry provider lifecycle (TODO §7).

Pins :func:`init_tracing` opt-in semantics:

- disabled: returns ``None``, no SDK provider installed;
- enabled without endpoint: a provider is installed globally with the
  configured resource, but nothing is exported;
- a custom service name lands on the resource;
- shutdown with no provider is a no-op.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from vector_service.core.config import TracingSettings
from vector_service.core.tracing import init_tracing, shutdown_tracing


@pytest.fixture
def clean_provider():
    """Reset OTel's one-shot global tracer slot around each test."""
    prev = trace._TRACER_PROVIDER
    prev_done = trace._TRACER_PROVIDER_SET_ONCE._done
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    try:
        yield
    finally:
        trace._TRACER_PROVIDER = prev
        trace._TRACER_PROVIDER_SET_ONCE._done = prev_done


def _settings(**tracing_kwargs):
    return SimpleNamespace(tracing=TracingSettings(**tracing_kwargs))


def test_disabled_returns_none_and_leaves_proxy(clean_provider):
    assert init_tracing(_settings(enabled=False)) is None
    assert not isinstance(trace.get_tracer_provider(), TracerProvider)


def test_enabled_without_endpoint_installs_provider(clean_provider):
    provider = init_tracing(_settings(enabled=True, endpoint=""))
    try:
        assert isinstance(provider, TracerProvider)
        assert trace.get_tracer_provider() is provider
        assert provider.resource.attributes["service.name"] == "vector-service"
        # No endpoint -> no exporting processor was attached.
        assert provider._active_span_processor._span_processors == ()
    finally:
        provider.shutdown()


def test_enabled_with_endpoint_attaches_batch_processor(clean_provider):
    provider = init_tracing(
        _settings(enabled=True, endpoint="http://collector:4318/v1/traces")
    )
    try:
        assert len(provider._active_span_processor._span_processors) == 1
    finally:
        provider.shutdown()


def test_custom_service_name(clean_provider):
    provider = init_tracing(_settings(enabled=True, service_name="vs-retrieval"))
    try:
        assert provider.resource.attributes["service.name"] == "vs-retrieval"
    finally:
        provider.shutdown()


def test_shutdown_none_is_noop():
    shutdown_tracing(None)
