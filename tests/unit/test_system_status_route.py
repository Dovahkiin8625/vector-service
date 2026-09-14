"""Tests for ``GET /v1/system/status``.

The endpoint powers the dashboard home panel — it must:
- always return 200 (fail-open so a degraded cluster can self-diagnose),
- surface service, model, store, and system sub-payloads,
- keep working when ``psutil`` is missing (degrade gracefully).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from vector_service.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_system_status_returns_expected_shape():
    with _client() as client:
        r = client.get("/v1/system/status")
    assert r.status_code == 200, f"GET /v1/system/status returned {r.status_code}"
    body = r.json()
    # Top-level sections — every one of these must exist so the
    # dashboard can render unconditionally.
    for key in ("service", "models", "store", "system"):
        assert key in body, f"missing '{key}' section in /v1/system/status payload"
    # Service sub-shape
    svc = body["service"]
    assert "version" in svc and svc["version"], "service.version missing"
    assert "embedding_backend" in svc
    assert "vector_store_backend" in svc
    assert "uptime_seconds" in svc
    assert isinstance(svc["uptime_seconds"], (int, float))
    assert svc["uptime_seconds"] >= 0
    # Models sub-shape — every known family must appear.
    for family in ("embedder", "image_embedder", "multimodal_embedder", "reranker"):
        assert family in body["models"], f"missing models.{family}"
        m = body["models"][family]
        assert "registered" in m and isinstance(m["registered"], list)
        assert "loaded_id" in m
        assert "dimensions" in m
    # Store sub-shape
    store = body["store"]
    assert "status" in store
    assert store["status"] in {"ok", "down", "not_attached"}
    assert "backend" in store
    assert "databases" in store and isinstance(store["databases"], list)
    assert "database_count" in store
    assert isinstance(store["database_count"], int)
    # System sub-shape
    sys = body["system"]
    for key in ("cpu", "memory", "disk", "process", "gpus", "os"):
        assert key in sys, f"missing system.{key}"
    assert "psutil_available" in sys and isinstance(sys["psutil_available"], bool)
    assert isinstance(sys["gpus"], list)


def test_system_status_handles_store_failure(monkeypatch):
    """The store probe must fail-open — a broken Milvus cannot 5xx status."""
    def _boom(_self):  # noqa: ANN001 — runtime patch on a method
        raise RuntimeError("simulated milvus outage")

    from vector_service.stores.milvus import MilvusStore

    monkeypatch.setattr(MilvusStore, "list_databases", _boom)
    with _client() as client:
        r = client.get("/v1/system/status")
    assert r.status_code == 200, (
        "/v1/system/status must not 5xx when the store probe fails"
    )
    body = r.json()
    assert body["store"]["status"] == "down"
    assert "simulated milvus outage" in (body["store"]["error"] or "")


def test_system_metrics_degrade_without_psutil(monkeypatch):
    """When psutil isn't installed, snapshot must still return a valid shape."""
    import vector_service.core.system_metrics as sm

    monkeypatch.setattr(sm, "_HAS_PSUTIL", False)
    snap = sm.system_snapshot()
    assert snap["psutil_available"] is False
    # CPU and memory must report None rather than raise.
    assert snap["cpu"]["percent"] is None
    assert snap["memory"]["total_bytes"] is None
    # Logical core count still comes from os.cpu_count().
    assert snap["cpu"]["logical_cores"] >= 1
    # OS sub-shape must be populated regardless of psutil.
    assert snap["os"]["system"]
    assert snap["os"]["python"]


def test_system_metrics_gpu_returns_diagnostic_when_no_backend(monkeypatch):
    """When neither pynvml nor torch.cuda finds a device, the snapshot
    must surface a *useful* hint naming the missing dep, not just
    ``available=False``. The dashboard renders this string verbatim.
    """
    import sys
    import vector_service.core.system_metrics as sm

    # Block every GPU-detection import so the fallback path runs.
    for name in ("pynvml", "nvidia_ml_py3", "torch"):
        sys.modules[name] = None
    try:
        gpus = sm._gather_gpu()
    finally:
        for name in ("pynvml", "nvidia_ml_py3", "torch"):
            sys.modules.pop(name, None)
    assert gpus, "_gather_gpu must return at least the diagnostic stub"
    diag = gpus[0]
    assert diag.get("available") is False
    # The hint must be actionable — mention the dep by name.
    assert diag.get("hint"), "no hint surfaced when GPU detection is unavailable"
    assert "nvidia-ml-py" in diag["hint"] or "pynvml" in diag["hint"]
    # ``missing`` field lists what's not importable, for telemetry.
    assert diag.get("missing")