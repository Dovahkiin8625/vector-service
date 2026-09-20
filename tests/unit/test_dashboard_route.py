"""Regression tests for the /dashboard route.

The dashboard is a Vue 3 single-page app served as:
- GET /dashboard             -> minimal Jinja2 template
- /static/dashboard/...      -> CSS, Vue runtime, and per-panel
  component modules under components/*.js.

Tests pin the contract between route handler, template, static
assets, and on-disk components. They read sources directly from disk
so editing a component file in dev reflects in the next assertion.
"""
from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from vector_service.main import app


# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------
_ST = pathlib.Path("src/vector_service")
_TPL = (_ST / "templates" / "dashboard.html").read_text(encoding="utf-8")
_CSS = (_ST / "static" / "dashboard" / "dashboard.css").read_text(encoding="utf-8")
# Vue's ESM browser build is shipped in latin-1; read with the
# right codec or the inlined concat raises UnicodeDecodeError.
_VUE = (_ST / "static" / "dashboard" / "vue.esm-browser.prod.js").read_text(encoding="latin-1")
_COMP = sorted((_ST / "static" / "dashboard" / "components").glob("*.js"))
_COMP_JS = "\n\n".join(p.read_text(encoding="utf-8") for p in _COMP)
_ALL = "\n".join([_TPL, _CSS, _VUE, _COMP_JS])


def _has(s):
    return s in _ALL


# ---------------------------------------------------------------------------
# Route + template
# ---------------------------------------------------------------------------


def test_dashboard_route_serves_vue_template():
    with TestClient(app) as c:
        r = c.get("/dashboard")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    assert ctype.startswith("text/html")
    body = r.text
    assert "vector-service" in body
    assert '<div id="app">' in body
    # The template must load Vue + the app.js entry as ES modules.
    assert 'type="module"' in body
    # Static assets all serve 200.
    with TestClient(app) as c:
        for path in [
            "/static/dashboard/dashboard.css",
            "/static/dashboard/vue.esm-browser.prod.js",
            "/static/dashboard/components/app.js",
            "/static/dashboard/components/databases.js",
            "/static/dashboard/components/collections.js",
            "/static/dashboard/components/modals.js",
        ]:
            rr = c.get(path)
            assert rr.status_code == 200, f"{path} -> {rr.status_code}"
    # No legacy / playground references.
    assert "playground" not in _ALL.lower()
    assert "LIFECYCLE_FAMILIES" not in _ALL


def test_dashboard_app_renders_topbar_and_sidebar():
    """Topbar, sidebar, statusbar are part of the App component."""
    assert _has('id="led-healthz"')
    assert _has('id="led-readyz"')
    assert _has('id="dim-dots"')
    assert _has('id="crumb-cat"')
    assert _has('id="crumb-sub"')
    assert _has('class="sidebar"')
    assert _has('class="statusbar"')
    # Vue uses v-for to render nav-items per view; verify the
    # data-view binding survives.
    assert 'data-view="overview"' in _ALL
    assert 'data-view="browse"' in _ALL


def test_dashboard_renders_per_model_not_per_family():
    """ModelsPanel emits one card per registered model with
    per-id data binding; no legacy family-keyed symbols."""
    models = (_ST / "static" / "dashboard" / "components" / "models.js").read_text(encoding="utf-8")
    assert "v-for" in models and "m.id" in models
    assert ":data-id=" in models or ':data-id="' in models
    assert "store.models.busy[m.id]" in models
    assert "FAMILY_LABELS" in models
    # No legacy family grouping.
    assert "LIFECYCLE_FAMILIES" not in _ALL
    assert "renderFamilyCard" not in _ALL


def test_dashboard_exposes_similarity_panels():
    """Three similarity views share the SimilarityPanel component."""
    assert _has('data-view="text-similarity"')
    assert _has('data-view="image-similarity"')
    assert _has('data-view="mm-similarity"')
    # Endpoint URLs.
    assert "/v1/text_similarity" in _ALL
    assert "/v1/image_similarity" in _ALL
    assert "/v1/multimodal_similarity" in _ALL
    # Per-kind prop dispatch.
    assert 'kind="text"' in _ALL
    assert 'kind="image"' in _ALL
    assert 'kind="multimodal"' in _ALL


def test_dashboard_exposes_knowledge_base_panels():
    """Knowledge base uses the KnowledgeBasePanel component switched
    by a view prop."""
    assert _has('data-view="parse"')
    assert _has('data-view="chunk"')
    assert _has('data-view="ingest"')
    assert "POST /v1/parse" in _ALL
    assert "POST /v1/chunk" in _ALL
    assert "POST /v1/ingest" in _ALL
    # Form-control ids.
    for f in [
        'id="parse-file"', 'id="parse-result"', 'id="parse-markdown"',
        'id="chunk-size"', 'id="chunk-overlap"', 'id="chunk-md"',
        'id="chunk-results"',
        'id="ingest-db"', 'id="ingest-size"', 'id="ingest-overlap"',
        'id="ingest-model"', 'id="ingest-file"', 'id="ingest-metadata"',
        'id="ingest-result"',
    ]:
        assert _has(f), f"{f} missing"


def test_dashboard_ingest_panel_streams_stage_events():
    """Ingest posts to the NDJSON stream route and renders a 5-stage
    stepper (upload/parse/chunk/embed/upsert). The old manual
    'refresh dbs' button is gone — dbs/models auto-refresh when the
    tab is opened via a view watcher."""
    kb = (
        _ST / "static" / "dashboard" / "components" / "knowledge-base.js"
    ).read_text(encoding="utf-8")
    # Stream endpoint (the classic /v1/ingest pill stays matched too).
    assert "POST /v1/ingest/stream" in kb
    assert "'/v1/ingest/stream'" in kb
    # XHR is required for upload byte progress + incremental NDJSON.
    assert "XMLHttpRequest" in kb
    assert "xhr.upload.onprogress" in kb
    # Stage stepper contract.
    assert 'class="ingest-stages"' in kb
    assert ':data-ingest-stage="s.key"' in kb
    for stage in ["uploading", "parse", "chunk", "embed", "upsert"]:
        assert f"key: '{stage}'" in kb
        assert f"ingest.stage.{stage if stage != 'uploading' else 'upload'}" in kb
    # Hint keys are composed dynamically: 'ingest.stage_hint.' + stage.
    assert "'ingest.stage_hint.' + ingestStage.value" in kb
    assert "ingest.failed_at" in kb
    # Auto-refresh replaces the removed button.
    assert "watch(() => props.view" in kb
    assert "btn-ingest-refresh" not in kb
    # i18n keys exist in both dictionaries.
    app_js = (
        _ST / "static" / "dashboard" / "components" / "app.js"
    ).read_text(encoding="utf-8")
    assert app_js.count("ingest.stage.upsert") == 2  # zh + en
    assert app_js.count("ingest.failed_at") == 2
    # Stepper styles.
    assert ".ingest-stages" in _CSS
    assert ".is-failed .stage-dot" in _CSS


def test_dashboard_exposes_browse_panel():
    """BrowsePanel renders the paginated table + pager."""
    assert _has('data-view="browse"')
    assert _has("v-show=\"store.view === 'browse'\"")
    assert "/v1/databases/{db}/collections/{coll}/rows" in _ALL
    for f in [
        'id="brw-db"', 'id="brw-coll"', 'id="brw-primary"',
        'id="brw-page-size"', 'id="brw-filter"',
        'id="brw-output-tokens"', 'id="brw-table-wrap"', 'id="brw-pager"',
        'id="btn-brw-query"', 'id="btn-brw-reset"',
        'id="btn-brw-first"', 'id="btn-brw-prev"',
        'id="btn-brw-next"', 'id="btn-brw-last"',
        'id="brw-jump"', 'id="btn-brw-jump"',
        'id="brw-stat-total"', 'id="brw-stat-page"', 'id="brw-stat-returned"',
        'id="brw-pager-info"',
        # row-selection + delete toolbar
        'id="brw-select-all"', 'id="brw-delete-bar"',
        'id="btn-brw-delete-selected"', 'id="btn-brw-delete-filter"',
        'id="btn-brw-clear-sel"', 'id="brw-delete-status"',
    ]:
        assert _has(f), f"{f} missing"
    assert "const columns = computed(" in _ALL
    assert "v-for=\"row in items\"" in _ALL or "v-for='row in items'" in _ALL


def test_dashboard_browse_panel_supports_delete():
    """Browse panel can delete ticked rows (ids) or every filter match.

    Both flows POST the existing ``.../vectors/delete`` endpoint with
    exactly one of ``ids`` / ``filter_expr`` (the backend enforces the
    XOR), and destructive clicks go through a confirm() guard.
    """
    browse = (
        _ST / "static" / "dashboard" / "components" / "browse.js"
    ).read_text(encoding="utf-8")
    assert "/vectors/delete" in browse
    # Selection plumbing: per-row checkbox + select-all-on-page.
    assert 'class="brw-row-check"' in browse
    assert "toggleOne" in browse and "togglePage" in browse
    assert "allPageSelected" in browse
    # Request bodies match DeleteVectorsRequest's XOR contract.
    assert "primary_field: primary.value, ids" in browse
    assert "primary_field: primary.value, filter_expr: expr" in browse
    # Both destructive actions require confirmation.
    assert browse.count("confirm(") >= 2


def test_dashboard_browse_panel_refreshes_after_delete():
    """Post-delete freshness contract: the delete toolbar (with its
    status line) is gated on the collection selection, NOT on ``total`` —
    otherwise deleting the last rows makes 'deleted N rows.' and the
    empty state's controls vanish together. The post-delete reload keeps
    the status; manual queries clear it."""
    browse = (
        _ST / "static" / "dashboard" / "components" / "browse.js"
    ).read_text(encoding="utf-8")
    # Toolbar survives total -> 0.
    assert '<div v-if="coll" class="actions brw-delete-bar"' in browse
    assert 'v-if="total" class="actions brw-delete-bar"' not in browse
    # runQuery knows whether to preserve the delete feedback.
    assert "async function runQuery(opts = {})" in browse
    assert "opts.keepDeleteStatus" in browse
    assert "runQuery({ keepDeleteStatus: true })" in browse


def test_dashboard_exposes_create_modals():
    """Database + collection creation forms live in modals."""
    assert _has('id="btn-open-new-db"')
    assert _has('id="btn-open-new-coll"')
    assert _has('id="modal-new-db"')
    assert _has('id="modal-new-coll"')
    for cls in ["modal-overlay", "modal", "modal--wide", "modal-head",
                "modal-body", "modal-foot", "modal-close", "modal-err"]:
        assert _has(cls)
    # Modal control flow.
    assert "store.modals.newDb = true" in _ALL
    assert "store.modals.newColl = true" in _ALL
    # Form-control ids inside the modals.
    for f in [
        'id="new-db-name"', 'id="new-coll-name"', 'id="new-coll-primary"',
        'id="scalars-list"', 'id="vector-card"', 'id="index-list"',
        'id="modal-new-db-err"', 'id="modal-new-coll-err"',
    ]:
        assert _has(f), f"{f} missing"


def test_dashboard_exposes_index_management():
    """Collection detail card has per-index delete + new-index form."""
    # Index-management DOM markers (data-* attrs survive the
    # rewrite).
    assert 'data-new-index-field' in _ALL
    assert 'data-new-index-metric' in _ALL
    assert 'data-new-index-type' in _ALL
    assert 'data-new-index-params' in _ALL
    assert 'data-new-index-submit' in _ALL
    # Drop-index endpoint construction.
    assert 'index?field_name=' in _ALL
    # Index v-for over the detail payload.
    assert "v-for=\"ix in detailCache[db + '::' + name].indexes\"" in _ALL


def test_dashboard_exposes_filter_delete():
    """Records panel offers ids-or-filter delete modes."""
    for f in [
        'id="vec-del-mode"', 'id="vec-del-ids"', 'id="vec-del-ids-row"',
        'id="vec-del-filter"', 'id="vec-del-filter-row"',
    ]:
        assert _has(f), f"{f} missing"
    assert 'value="ids"' in _ALL
    assert 'value="filter"' in _ALL
    # JS body shapes per mode.
    assert "body.ids = arr" in _ALL or "body.ids = ids" in _ALL
    assert "body.filter_expr" in _ALL


def test_dashboard_exposes_db_detail_view():
    """Each db row is clickable to reveal a detail panel."""
    dbs = (_ST / "static" / "dashboard" / "components" / "databases.js").read_text(encoding="utf-8")
    assert "loadDetail" in dbs
    assert "Object.create(null)" in dbs
    assert "Promise.all" in dbs
    assert "data-detail-for-db" in dbs


# ---------------------------------------------------------------------------
# Router + OpenAPI tag (legacy pinning)
# ---------------------------------------------------------------------------


def test_old_playground_route_is_gone():
    with TestClient(app) as c:
        r = c.get("/playground")
    assert r.status_code == 404


def test_dashboard_router_tag_is_dashboard():
    from vector_service.api.dashboard import router
    assert "dashboard" in router.tags
    assert "playground" not in router.tags
