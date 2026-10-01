"""Regression tests for the dashboard i18n layer.

The dashboard ships two flat dictionaries (``I18N.zh`` / ``I18N.en``)
inside ``components/app.js`` and a single ``t(key, vars)`` helper that
every panel is expected to route its user-visible prose through.

These tests pin three contracts:

1. **Key parity** — the two locales define exactly the same key set,
   with the same ``{placeholder}`` names, so switching language can
   never silently drop a label.
2. **``t()`` semantics** — unknown keys render as the key itself and
   warn once; ``vars`` substitute ``{name}`` placeholders.
3. **No hard-coded panel prose** — the strings that used to be baked
   into the panel modules stay gone.

Sources are read from disk (same style as ``test_dashboard_route.py``)
so editing a component reflects in the next assertion.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

_ST = pathlib.Path("src/vector_service")
_COMP_DIR = _ST / "static" / "dashboard" / "components"
_APP_JS = (_COMP_DIR / "app.js").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dictionary extraction
# ---------------------------------------------------------------------------
def _dict_region(name: str) -> str:
    """Slice the source between ``  <name>: {`` and the next top-level
    locale key (or the closing brace of ``I18N``)."""
    start = re.search(rf"^  {name}: \{{$", _APP_JS, re.M)
    assert start, f"I18N.{name} not found"
    rest = _APP_JS[start.end():]
    end = re.search(r"^  \},$|^\};$", rest, re.M)
    assert end, f"I18N.{name} unterminated"
    return rest[: end.start()]


def _entries(name: str) -> dict[str, str]:
    """Parse ``'key': 'value',`` lines, honouring escaped quotes."""
    body = _dict_region(name)
    return {
        m.group(1): m.group(2)
        for m in re.finditer(
            r"^    '([^']+)':\s*'((?:[^'\\]|\\.)*)',\s*$", body, re.M
        )
    }


ZH = _entries("zh")
EN = _entries("en")
_COMPONENTS = sorted(_COMP_DIR.glob("*.js"))
_COMP_JS = "\n\n".join(p.read_text(encoding="utf-8") for p in _COMPONENTS)
# Everything except app.js, which legitimately holds the English wording
# inside its dictionaries — the prose sweep only cares about panel code.
_PANEL_JS = "\n\n".join(
    p.read_text(encoding="utf-8") for p in _COMPONENTS if p.name != "app.js"
)


def _placeholders(value: str) -> list[str]:
    return sorted(re.findall(r"\{([a-zA-Z0-9_]+)\}", value))


# ---------------------------------------------------------------------------
# 1. Key parity
# ---------------------------------------------------------------------------
def test_dictionaries_are_non_trivial():
    """Guard the parser itself: a broken regex must not make the
    parity assertions below vacuously pass."""
    assert len(ZH) > 500, f"parsed only {len(ZH)} zh keys"
    assert len(EN) > 500, f"parsed only {len(EN)} en keys"


def test_locales_define_the_same_keys():
    zh_only = sorted(set(ZH) - set(EN))
    en_only = sorted(set(EN) - set(ZH))
    assert not zh_only, f"keys missing from en: {zh_only}"
    assert not en_only, f"keys missing from zh: {en_only}"


def test_locale_values_are_non_empty():
    empty = sorted(k for k, v in {**ZH, **EN}.items() if not v.strip())
    assert not empty, f"blank translations: {empty}"


def test_placeholders_match_across_locales():
    """``{n}`` in zh must be ``{n}`` in en — a renamed placeholder
    leaves a literal ``{total}`` on screen in one language."""
    mismatched = {
        k: (_placeholders(ZH[k]), _placeholders(EN[k]))
        for k in ZH
        if _placeholders(ZH[k]) != _placeholders(EN[k])
    }
    assert not mismatched, f"placeholder mismatch: {mismatched}"


def test_panel_key_groups_exist_in_both_locales():
    """The panel-grouped namespaces introduced by the i18n sweep."""
    for prefix in (
        "embeddings.", "rerank.", "similarity.", "search.", "browse.",
        "collections.", "records.", "modals.", "retrieval.", "kb.",
    ):
        assert any(k.startswith(prefix) for k in ZH), f"no zh keys for {prefix}"
        assert any(k.startswith(prefix) for k in EN), f"no en keys for {prefix}"


def test_retrieval_stage_keys_cover_the_pipeline():
    """Stage vocabulary comes from retrieval/pipeline.py's _Timer names;
    a new stage must not silently render as a raw key."""
    for stage in (
        "route", "rewrite", "recall", "fuse", "mmr", "rerank",
        "expand", "compress",
    ):
        assert f"retrieval.stage.{stage}" in ZH, f"missing zh stage {stage}"
        assert f"retrieval.stage.{stage}" in EN, f"missing en stage {stage}"


# ---------------------------------------------------------------------------
# 2. t() semantics
# ---------------------------------------------------------------------------
def test_t_warns_once_on_missing_key():
    src = _APP_JS
    assert "_warnedKeys" in src
    assert "console.warn('[i18n] missing key:'" in src
    # Fallback is the key itself, never an empty string.
    assert "s = key;" in src


def test_t_supports_placeholder_interpolation():
    src = _APP_JS
    assert "export function t(key, vars)" in src
    assert "s.split('{' + k + '}').join(String(vars[k]))" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_t_runtime_behaviour(tmp_path):
    """Execute the real ``t()`` under node against the real dictionary.

    The script goes through a file rather than ``node -e`` — the
    dictionaries are large enough to blow the Windows command-line
    limit.
    """
    literal = _APP_JS[_APP_JS.index("const I18N = {") + len("const I18N = "):]
    literal = literal[: literal.index("\n};") + 2]
    fn = _APP_JS[_APP_JS.index("const _warnedKeys"):]
    fn = fn[: fn.index("\n}\n") + 3]
    out_file = tmp_path / "out.json"
    script = f"""
import fs from 'node:fs';
const I18N = {literal};
const store = {{ locale: 'en' }};
{fn}
const out = {{
  known: t('common.reset'),
  missing: t('no.such.key'),
  interp: t('kb.chars_tokens', {{ chars: 12, tokens: 7 }}),
  zh: (store.locale = 'zh', t('common.reset')),
}};
// Written to a file, not stdout: the Windows console codepage would
// mangle the Chinese values on the way back.
fs.writeFileSync({json.dumps(str(out_file))}, JSON.stringify(out), 'utf8');
"""
    mod = tmp_path / "t_runtime.mjs"
    mod.write_text(script, encoding="utf-8")
    subprocess.run(
        ["node", str(mod)], capture_output=True, text=True, check=True
    )
    out = json.loads(out_file.read_text(encoding="utf-8"))
    assert out["known"] == EN["common.reset"]
    assert out["zh"] == ZH["common.reset"]
    assert out["missing"] == "no.such.key"
    assert out["interp"] == "12 chars · 7 tokens"


# ---------------------------------------------------------------------------
# 3. Panels route prose through t()
# ---------------------------------------------------------------------------
# Strings that were hard-coded in the components before the sweep. Each
# is pinned as a full literal (not a bare word) so legitimate uses inside
# an English dictionary value stay allowed.
_DEAD_LITERALS = [
    "no {{ kind }} models.",
    ">input mode<",
    "server-side embed (query_text)",
    "direct vector (query_vector)",
    "filter expression (Milvus native)",
    "click to toggle",
    "registered rerankers",
    "no reranker models.",
    ">cross-encoder rerank<",
    "JSON mode requires an array of strings.",
    "query_vector must be a JSON array.",
    "select database and collection.",
    "delete by filter",
    "delete selected",
    "primary key field",
    ">documents<",
    ">chunks<",
    ">markdown text<",
    ">embed model<",
    ">file</label>",
    ">strategy</label>",
    ">breakpoint %</label>",
    ">chunk size</label>",
    ">overlap</label>",
    ">database</label>",
    "download .md",
    "« first",
    "‹ prev",
    "next ›",
    "last »",
    "- no instance",
    "'开' : '关'",
    "确认删除数据库",
    "删除失败",
    "database name is required.",
    "collection name is required.",
    "vector dim must be >= 1.",
    "every field must have a name.",
    ">one per line<",
    ">pick several at once<",
    ">jump to<",
]


def test_no_hardcoded_panel_prose_survives():
    hits = [lit for lit in _DEAD_LITERALS if lit in _PANEL_JS]
    assert not hits, f"hard-coded UI prose still in components: {hits}"


def test_panels_use_the_translation_helper():
    """Every panel module renders through ``$t`` / ``t``."""
    for name in (
        "embeddings.js", "rerank.js", "similarity.js", "search.js",
        "browse.js", "collections.js", "records.js", "retrieval.js",
        "knowledge-base.js", "modals.js",
    ):
        src = (_COMP_DIR / name).read_text(encoding="utf-8")
        assert "$t(" in src or re.search(r"\bt\(", src), f"{name} does not call t()"


def test_every_referenced_key_is_defined():
    """No component may reference a key the dictionaries do not define.

    ``t()`` renders an unknown key as the key itself and logs
    ``[i18n] missing key``, so a typo'd key ships as visible
    ``some.panel.label`` text. Keys built by concatenation
    (``t('similarity.mode.' + mode.value)``) match the scan as a
    trailing-dot prefix and are skipped — their concrete forms are
    covered by the key-group assertions above.
    """
    # The lookbehind keeps `emit(` / `act(` / `format(` out: only a
    # standalone `t(` or `$t(` is a translation call.
    pattern = re.compile(r"""(?<![A-Za-z0-9_$])\$?t\(\s*'([A-Za-z0-9_.]+)'""")
    used = set()
    for path in _COMPONENTS:
        used |= set(pattern.findall(path.read_text(encoding="utf-8")))
    used = {k for k in used if not k.endswith(".")}
    missing = sorted(k for k in used if k not in ZH or k not in EN)
    assert not missing, f"keys referenced but not defined: {missing}"
