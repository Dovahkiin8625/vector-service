"""Regression test for the /dashboard route.

The dashboard is a self-contained HTML page that replaces the old
``/playground`` route. We pin:

- the route exists at ``/dashboard`` and returns 200,
- the page contains the expected markers (title, signature, panels),
- the legacy ``/playground`` URL is gone (so old bookmarks break loudly
  instead of silently hitting a stale endpoint).

The "models" panel remains the registry surface. The overview/home
panel (``#panel-overview``) is the default landing view — it aggregates
service, model, store, and machine metrics via ``GET /v1/system/status``.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from vector_service.main import app


def test_dashboard_route_returns_html_with_expected_markers():
    with TestClient(app) as client:
        r = client.get("/dashboard")
    assert r.status_code == 200, f"GET /dashboard returned {r.status_code}"
    ctype = r.headers.get("content-type", "")
    assert ctype.startswith("text/html"), f"unexpected content-type: {ctype!r}"
    body = r.text

    # identity
    assert "vector-service · Dashboard" in body, "page <title> not found"
    assert "Dashboard" in body, "Dashboard brand marker missing"

    # structural markers (overview panel is now the default landing view;
    # the models registry panel remains present but is no longer active).
    assert 'class="sidebar"' in body, "left sidebar nav missing"
    assert 'id="panel-overview"' in body, "overview panel must exist"
    assert '<div class="panel active" id="panel-overview"' in body, (
        "overview panel must be the default active landing view"
    )
    assert 'id="panel-models"' in body, "models panel must still exist"
    assert '<div class="panel" id="panel-models"' in body, (
        "models panel must NOT be the active default view (overview is)"
    )
    assert 'id="models-grid"' in body, "models card grid placeholder missing"
    assert 'id="dim-dots"' in body, "signature dim-dots row missing"

    # Overview panel key markers
    assert "/v1/system/status" in body, "overview panel must reference /v1/system/status endpoint"
    assert 'id="kpi-version"' in body, "version KPI missing"
    assert 'id="kpi-uptime"' in body, "uptime KPI missing"
    assert 'id="kpi-models"' in body, "models KPI missing"
    assert 'id="kpi-store"' in body, "store KPI missing"
    assert 'id="kpi-cpu"' in body, "cpu KPI missing"
    assert 'id="kpi-mem"' in body, "memory KPI missing"
    assert 'id="kpi-gpu"' in body or 'id="gpu-list"' in body, "gpu section missing"
    assert "总览首页" in body, "overview nav-item label missing"

    # no stale references in shipped HTML
    assert "playground" not in body.lower(), "stale 'playground' string found in dashboard HTML"
    assert "class=\"kpi-grid\"" not in body, "stale .kpi-grid class still shipped"
    assert "class=\"recent-table\"" not in body, "stale .recent-table still shipped"
    assert "family-card" not in body, "stale 'family-card' class found in dashboard HTML"
    assert "LIFECYCLE_FAMILIES" not in body, "stale LIFECYCLE_FAMILIES identifier in dashboard JS"
    assert 'id="panel-lifecycle"' not in body, "stale #panel-lifecycle still shipped"


def test_old_playground_route_is_gone():
    """Old URL must 404 so users notice the rename instead of silently breaking."""
    with TestClient(app) as client:
        r = client.get("/playground")
    assert r.status_code == 404, (
        f"legacy /playground still resolves ({r.status_code}); rename was incomplete"
    )


def test_dashboard_router_tag_is_dashboard():
    """The OpenAPI tag on the dashboard router is 'dashboard', not 'playground'."""
    from vector_service.api.dashboard import router

    assert "dashboard" in router.tags, (
        f"expected dashboard router tags to include 'dashboard', got {router.tags!r}"
    )
    assert "playground" not in router.tags, (
        f"stale 'playground' tag still on dashboard router: {router.tags!r}"
    )


def test_dashboard_renders_per_model_not_per_family():
    """Regression: load/unload + cards are keyed by model_id.

    The service exposes load/unload per model_id (the slot is keyed by
    family, but the API surface and the user-facing mental model are
    per-id). The dashboard must therefore render one card per
    registered model — *not* one per family — and every action button
    must carry a ``data-id`` that matches the model the user sees.
    """
    from vector_service.api.dashboard import DASHBOARD_HTML, _DASHBOARD_JS_END

    # Sanity: the JS block is present and ends on the marker.
    assert _DASHBOARD_JS_END in DASHBOARD_HTML, "dashboard JS end marker missing"

    # Per-model identifiers shipped to the browser.
    assert "renderModelCard" in DASHBOARD_HTML, (
        "renderModelCard helper missing — cards are still keyed by family"
    )
    assert "bindModelCardActions" in DASHBOARD_HTML, (
        "bindModelCardActions helper missing — actions are still bound by family selector"
    )
    assert "data-id=" in DASHBOARD_HTML, (
        "model cards must expose data-id for per-model load/unload buttons"
    )
    assert "setModelsAutoRefresh" in DASHBOARD_HTML, "auto-refresh helper missing"
    assert "btn-models-auto-refresh" in DASHBOARD_HTML, "auto-refresh button id missing"
    assert 'id="models-grid"' in DASHBOARD_HTML, "merged model card grid missing"

    # Legacy family-keyed symbols must be gone. Use word-boundary-ish
    # checks so the new helper names don't false-positive.
    assert "LIFECYCLE_FAMILIES" not in DASHBOARD_HTML, (
        "LIFECYCLE_FAMILIES still present — overview/lifecycle still grouped by family"
    )
    assert "renderFamilyCard" not in DASHBOARD_HTML, (
        "renderFamilyCard still present — cards still grouped by family"
    )
    assert "function renderLcDetail(" not in DASHBOARD_HTML, (
        "renderLcDetail(...) still present — lifecycle still groups by family"
    )
    assert "renderLcModelRow" not in DASHBOARD_HTML, (
        "renderLcModelRow still present — lifecycle row form was not removed"
    )
    assert "renderLcDetails" not in DASHBOARD_HTML, (
        "renderLcDetails still present — lifecycle details form was not removed"
    )
    assert "groupByFamily" not in DASHBOARD_HTML, (
        "groupByFamily helper still present — lifecycle group form was not removed"
    )
    assert "function refreshLifecycle" not in DASHBOARD_HTML, (
        "refreshLifecycle still present — lifecycle panel was not removed"
    )
    assert "function setLifecycleAutoRefresh" not in DASHBOARD_HTML, (
        "setLifecycleAutoRefresh still present — auto-refresh helper was not renamed"
    )
    assert "lc-card-load" in DASHBOARD_HTML and "lc-card-unload" in DASHBOARD_HTML, (
        "load/unload button classes missing"
    )

    # Busy state must be keyed by model id, not by family. We assert
    # the JS path mutates ``modelsState.busy[id]`` so that only the
    # affected card is dimmed during a load/unload round-trip.
    assert "modelsState.busy[id]" in DASHBOARD_HTML, (
        "modelsState.busy must be keyed by model id (modelsState.busy[id] = ...)"
    )
    assert "lifecycleState.busy" not in DASHBOARD_HTML, (
        "lifecycleState.busy still present — rename to modelsState.busy incomplete"
    )

    # The 已加载 / 卸载 toggle must key off the explicit ``m.loaded``
    # signal rather than ``m.dimensions != null``. ``dimensions`` is
    # permanently ``null`` for rerankers, so dimension-based detection
    # would silently mis-report every loaded reranker as 未加载 and
    # leave the load button in place after a successful POST /load.
    assert "var isLoaded = !!m.loaded;" in DASHBOARD_HTML, (
        "renderModelCard must derive 已加载 from m.loaded, not m.dimensions"
    )
    assert "m.dimensions != null" not in DASHBOARD_HTML, (
        "stale 'm.dimensions != null' load-state check still present — "
        "rerankers would always show as 未加载"
    )
