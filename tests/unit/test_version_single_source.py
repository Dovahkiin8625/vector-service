"""Version single-source-of-truth invariants.

``vector_service.__init__.__version__`` is the only place the service
version may be written as a literal. Everything else — packaging
metadata, the Prometheus ``vs_info`` gauge, ``GET /v1/system/status``,
and the dashboard statusbar — derives from that constant (see the
comment block in ``src/vector_service/__init__.py``).

These tests fail when a second literal creeps back in, when packaging
stops pointing at the attribute, or when the statusbar regresses to a
hardcoded string.
"""
from __future__ import annotations

import pathlib
import re
import tomllib

from fastapi.testclient import TestClient

from vector_service import __version__
from vector_service.core.metrics import REGISTRY
from vector_service.main import app

# Three-segment numeric literals ("0.2.0"). The lookarounds exclude
# longer runs like ``0.0.0.0`` (host defaults) and 4-part versions.
_VERSION_RE = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?![\d.])")

_SRC = pathlib.Path("src/vector_service")
_PKG = pathlib.Path("pyproject.toml")


def _scanned_sources() -> list[tuple[pathlib.Path, str]]:
    """All version-bearing sources except the single allowed literal.

    Covers ``src/**/*.py``, dashboard ``components/*.js`` and
    ``templates/*.html``. The vendored Vue ESM build is excluded (it
    carries its own upstream version string); ``__init__.py`` is held
    out separately so it can be asserted to be the *only* copy.
    """
    out: list[tuple[pathlib.Path, str]] = []
    for pattern in ("**/*.py", "static/dashboard/components/*.js", "templates/*.html"):
        for path in sorted(_SRC.glob(pattern)):
            if path.name == "__init__.py":
                continue
            if path.name == "vue.esm-browser.prod.js":
                continue
            out.append((path, path.read_text(encoding="utf-8")))
    return out


# ---------------------------------------------------------------------------
# 1. Exactly one version literal in the whole tree
# ---------------------------------------------------------------------------


def test_version_literal_appears_exactly_once():
    for path, text in _scanned_sources():
        hits = _VERSION_RE.findall(text)
        assert not hits, f"{path} contains version literal(s) {hits}; use vector_service.__version__"
    init_text = (_SRC / "__init__.py").read_text(encoding="utf-8")
    init_hits = _VERSION_RE.findall(init_text)
    assert init_hits == [__version__], f"__init__.py literals {init_hits} != [{__version__!r}]"


# ---------------------------------------------------------------------------
# 2. Packaging metadata reads the attribute dynamically
# ---------------------------------------------------------------------------


def test_pyproject_version_is_dynamic():
    data = tomllib.loads(_PKG.read_text(encoding="utf-8"))
    project = data["project"]
    assert "version" not in project, "pyproject.toml must not hardcode project.version"
    assert "version" in project.get("dynamic", []), "dynamic must list 'version'"
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "vector_service.__version__", f"unexpected version attr {attr!r}"


# ---------------------------------------------------------------------------
# 3. The constant itself is a renderable version string
# ---------------------------------------------------------------------------


def test_version_constant_is_well_formed():
    assert _VERSION_RE.fullmatch(__version__), f"__version__={__version__!r} is not X.Y.Z"


# ---------------------------------------------------------------------------
# 4. System status route serves the constant
# ---------------------------------------------------------------------------


def test_system_status_route_serves_package_version():
    with TestClient(app) as c:
        r = c.get("/v1/system/status")
    assert r.status_code == 200
    assert r.json()["service"]["version"] == __version__


# ---------------------------------------------------------------------------
# 5. Prometheus vs_info labels itself with the constant
# ---------------------------------------------------------------------------


def test_vs_info_metric_carries_package_version():
    with TestClient(app):
        pass  # lifespan sets the VS_INFO labels
    versions = set()
    for metric in REGISTRY.collect():
        if metric.name != "vs_info":
            continue
        for sample in metric.samples:
            versions.add(sample.labels["version"])
    assert versions == {__version__}, f"vs_info version labels {versions} != {{{__version__!r}}}"


# ---------------------------------------------------------------------------
# 6. Dashboard statusbar renders the value from the API payload
# ---------------------------------------------------------------------------


def test_statusbar_renders_store_service_fields():
    app_js = (_SRC / "static/dashboard/components/app.js").read_text(encoding="utf-8")
    # Bound, not literal: the statusbar reads store.service.*.
    assert "store.service.version" in app_js
    assert "store.service.embeddingBackend" in app_js
    assert "store.service.storeBackend" in app_js
    # Filled from the same /v1/system/status payload the route above serves.
    assert "/v1/system/status" in app_js
