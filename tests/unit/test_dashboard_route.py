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


def test_dashboard_exposes_similarity_panel():
    """The three similarity views are merged into one SimilarityPanel
    with an internal text/image/multimodal mode switch."""
    sim = (
        _ST / "static" / "dashboard" / "components" / "similarity.js"
    ).read_text(encoding="utf-8")
    # One merged sidebar view; the old per-modality views are gone.
    assert _has('data-view="similarity"')
    assert 'data-view="text-similarity"' not in _ALL
    assert 'data-view="image-similarity"' not in _ALL
    assert 'data-view="mm-similarity"' not in _ALL
    # The mode switch still targets all three endpoints.
    assert "/v1/text_similarity" in sim
    assert "/v1/image_similarity" in sim
    assert "/v1/multimodal_similarity" in sim
    # Internal segmented mode switch (no kind prop from the App).
    assert "seg-toggle" in sim
    assert "const mode = ref('text')" in sim
    assert 'kind:' not in sim


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


def test_dashboard_exposes_retrieval_panel():
    """Retrieval panel: NDJSON-streamed multi-channel retrieval with
    mode presets, intent routing, token budget and capabilities-driven
    disabled states."""
    retrieval = (
        _ST / "static" / "dashboard" / "components" / "retrieval.js"
    ).read_text(encoding="utf-8")
    app_js = (
        _ST / "static" / "dashboard" / "components" / "app.js"
    ).read_text(encoding="utf-8")
    # Served as a static asset.
    with TestClient(app) as c:
        rr = c.get("/static/dashboard/components/retrieval.js")
    assert rr.status_code == 200
    # Nav wiring: NAV_LABELS entry, kb-group nav-item, panel mount and
    # component registration/import.
    assert _has('data-view="retrieval"')
    assert "'retrieval':" in app_js
    assert "import RetrievalPanel from './retrieval.js'" in app_js
    assert "KnowledgeBasePanel, RetrievalPanel," in app_js
    assert "<retrieval-panel v-show=\"store.view === 'retrieval'\" />" in app_js
    # Endpoints consumed by the panel.
    assert "'/v1/retrieval/stream'" in retrieval
    assert "'/v1/retrieval/capabilities" in retrieval
    # No migrate residue in the panel.
    assert "migrate" not in retrieval
    # Four mode presets.
    for m in ["'basic'", "'hybrid'", "'advanced'", "'custom'"]:
        assert m in retrieval
    # Four-channel weights, intent routing and token budget controls.
    assert "s.wSummary" in retrieval
    assert "s.wGraph" in retrieval
    assert "s.routing" in retrieval
    assert "s.maxTokens" in retrieval
    # Capabilities-driven disabled states: LLM classifier off without
    # an LLM; the whole rewrite group too.
    assert ':disabled="!s.caps.llm_configured"' in retrieval
    # NDJSON frame splitting (partial trailing line kept in buffer).
    assert "buffer.split('\\n')" in retrieval
    # i18n keys exist in both dictionaries (nav.retrieval is also used
    # by NAV_LABELS and the nav-item's t() -> 4 occurrences).
    assert app_js.count("'nav.retrieval'") == 4
    assert app_js.count("'retrieval.trace'") == 2
    # Dead migrate i18n keys are gone from both locales.
    assert "retrieval.v1_banner" not in app_js
    assert "retrieval.migrate" not in app_js
    # Styles.
    assert ".retrieval-modes" in _CSS
    assert ".retrieval-migrate-banner" not in _CSS
    assert ".retrieval-chunk" in _CSS


def test_dashboard_exposes_operations_panels():
    """Operations group (TODO §7): job queue, index rebuild, consistency
    scan and evaluation/gates panels wired into the app shell."""
    app_js = (
        _ST / "static" / "dashboard" / "components" / "app.js"
    ).read_text(encoding="utf-8")
    # All five new modules serve as static assets.
    with TestClient(app) as c:
        for path in [
            "/static/dashboard/components/ops-common.js",
            "/static/dashboard/components/ops-queue.js",
            "/static/dashboard/components/ops-reindex.js",
            "/static/dashboard/components/ops-consistency.js",
            "/static/dashboard/components/ops-eval.js",
        ]:
            rr = c.get(path)
            assert rr.status_code == 200, f"{path} -> {rr.status_code}"
    # Imports + component registration + panel mounts.
    for stmt in [
        "import OpsQueuePanel from './ops-queue.js'",
        "import OpsReindexPanel from './ops-reindex.js'",
        "import OpsConsistencyPanel from './ops-consistency.js'",
        "import OpsEvalPanel from './ops-eval.js'",
    ]:
        assert stmt in app_js
    assert "OpsQueuePanel, OpsReindexPanel, OpsConsistencyPanel, OpsEvalPanel," in app_js
    for tag in [
        "<ops-queue-panel v-show=\"store.view === 'queue'\" />",
        "<ops-reindex-panel v-show=\"store.view === 'reindex'\" />",
        "<ops-consistency-panel v-show=\"store.view === 'consistency'\" />",
        "<ops-eval-panel v-show=\"store.view === 'eval'\" />",
    ]:
        assert tag in app_js
    # Sidebar group + data-view bindings + NAV_LABELS entries.
    assert 'data-group="ops"' in app_js
    for view in ["queue", "reindex", "consistency", "eval"]:
        assert f'data-view="{view}"' in app_js
        assert f"'{view}':" in app_js
    # Endpoints consumed by the panels (pills + template strings are
    # both covered by _ALL).
    for ep in [
        "/v1/jobs", "/v1/jobs/{id}", "/v1/jobs/{id}/cancel",
        "/v1/jobs/{id}/events",
        "/v1/databases/{db}/collections/{coll}/reindex",
        "/v1/databases/{db}/collections/{coll}/consistency",
        "/v1/evaluation/sets", "/v1/evaluation/gates",
        "/v1/evaluation/runs/{id}",
    ]:
        assert _has(ep), f"{ep} missing"
    # i18n keys exist in both locales. 'nav.*' also appears in NAV_LABELS
    # and the sidebar (4 total); 'cat.ops' backs four NAV_LABELS entries
    # plus two locales (6); 'nav.Operations' is sidebar + two locales (3).
    for key in ["nav.queue", "nav.reindex", "nav.consistency", "nav.eval"]:
        assert app_js.count(f"'{key}'") == 4
    assert app_js.count("'cat.ops'") == 6
    assert app_js.count("'nav.Operations'") == 3
    # Shared panel keys land in both locales (exactly one entry each).
    for key in [
        "ops.queue.status", "ops.reindex.submit", "ops.consistency.scan",
        "ops.eval.tab_sets", "common.prev", "common.next",
    ]:
        assert app_js.count(f"'{key}'") == 2

    # Status words and shared field labels exist in both locales, and
    # panels consume them through statusLabel/$t instead of raw words.
    for key in [
        "ops.status.done", "ops.status.failed", "common.job_id",
        "common.doc_id", "common.scope", "ops.consistency.milvus_rows",
        "ops.eval.mean_recall", "common.question_id",
    ]:
        assert app_js.count(f"'{key}'") == 2
    assert "statusLabel" in _COMP_JS
    # No raw English template strings survive in the ops modules.
    # (The words may legitimately appear inside en dictionary values,
    # so pin the old template wrappers rather than bare words.)
    assert "physical index(es)" not in _COMP_JS
    assert ">source of truth</span>" not in _COMP_JS
    assert "total · offset" not in _COMP_JS
