"""Interactive dashboard page for debugging the service.

Mounts a single self-contained HTML page at ``GET /dashboard``. The page
talks to the existing public endpoints only (healthz, /v1/models,
/v1/embeddings, /v1/databases family, etc.) — it does not call any
debug-only or hidden routes.

UI is **light-themed, minimalist**, Chinese-localised. The default
landing is an **overview** that summarises service health, model-load
state. From the left sidebar the operator
can drill into the same endpoints the debug page used to expose: model
list, text/image/multimodal embeddings, rerank, model load/unload,
databases, collections, vectors, and search.

The font stack is unified on **Microsoft YaHei (微软雅黑)** with system
fallbacks; a separate mono stack handles only code panes and HTTP
headers.

The signature visual is a row of dim-dots in the topbar — the dot
count equals the dimensions of the currently-loaded embedder, and a
subtle pulse ripples along the row when a request completes.

The HTML / CSS / JS are split across three real files for editability
and hot-reload:

- ``templates/dashboard.html``  — HTML skeleton + panels + modals
- ``static/dashboard/dashboard.css`` — all styles
- ``static/dashboard/dashboard.js``  — all client-side scripts
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["dashboard"])

# Resolve the three resource paths once at import time. Using
# ``__file__`` keeps the layout correct regardless of the cwd
# ``python -m vector_service`` is launched from.
_HERE = Path(__file__).resolve().parent
_TPL_DIR = _HERE.parent / "templates"
_STATIC_DIR = _HERE.parent / "static"

templates = Jinja2Templates(directory=str(_TPL_DIR))


@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Interactive dashboard",
    description=(
        "Single-page operator dashboard. The page is rendered from "
        "``templates/dashboard.html`` with CSS / JS served as static "
        "assets from ``/static/dashboard/``."
    ),
)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the interactive dashboard page."""
    return templates.TemplateResponse(request, "dashboard.html", {})


# ---------------------------------------------------------------------------
# Backwards-compatible test helper
# ---------------------------------------------------------------------------
# The unit-test suite (~200 assertions) reads
# ``vector_service.api.dashboard.DASHBOARD_HTML`` and substring-checks
# the rendered page. After splitting the page into three files, we
# rebuild the same flat string on demand by inlining the CSS and JS
# back into the template. This keeps tests untouched while letting
# production serve the page via Jinja2 + StaticFiles.
_CSS_PATH = _STATIC_DIR / "dashboard" / "dashboard.css"
_TPL_PATH = _TPL_DIR / "dashboard.html"
_COMPONENTS_DIR = _STATIC_DIR / "dashboard" / "components"
_VUE_PATH = _STATIC_DIR / "dashboard" / "vue.esm-browser.prod.js"
# Vue's ESM browser build is shipped in latin-1 (smaller than UTF-8
# BOM). Read it explicitly so ``_build_dashboard_html`` works
# regardless of platform default encoding.
_VUE_ENCODING = "latin-1"

_CSS_LINK = (
    '<link rel="stylesheet" '
    "href=\"{{ url_for('static', path='dashboard/dashboard.css') }}\" />"
)
_VUE_TAG = (
    '<script src="{{ url_for(\'static\', path=\'dashboard/vue.esm-browser.prod.js\') }}">'
    '</script>'
)
_APP_TAG = (
    '<script type="module" '
    "src=\"{{ url_for('static', path='dashboard/components/app.js') }}\">"
    '</script>'
)


# Concatenate the template + CSS + Vue + all components into one
# string for tests. The tests run hundreds of substring checks
# (e.g. ``"renderModelCard" in DASHBOARD_HTML``); reading once at
# import keeps the cost predictable. Production serves each file
# separately via Jinja2 + StaticFiles + ES module imports.
def _build_dashboard_html() -> str:
    template = _TPL_PATH.read_text(encoding="utf-8")
    css = _CSS_PATH.read_text(encoding="utf-8")
    vue = _VUE_PATH.read_text(encoding=_VUE_ENCODING)
    components = []
    for path in sorted(_COMPONENTS_DIR.glob("*.js")):
        components.append(path.read_text(encoding="utf-8"))
    components_js = "\n\n".join(components)

    if _CSS_LINK in template:
        template = template.replace(_CSS_LINK, f"<style>{css}</style>")
    if _VUE_TAG in template:
        template = template.replace(_VUE_TAG, f"<script>{vue}</script>")
    if _APP_TAG in template:
        template = template.replace(
            _APP_TAG,
            f"<script>\n{components_js}\n</script>",
        )
    return template


DASHBOARD_HTML: str = _build_dashboard_html()
