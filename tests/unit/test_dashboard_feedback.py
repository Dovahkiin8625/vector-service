"""S2 (operation-feedback) convention guards.

``docs/superpowers/plans/2026-10-01-dashboard-ui-review.md`` §1 S2 replaced
the native ``alert()`` / ``confirm()`` dialogs
with one inline convention shared by every panel:

- ``StatusBanner``  the outcome of a submit (error / success / info / warn)
- ``BusyButton``    a submit button that disables itself while in flight
- ``EmptyState``    the four list states (idle / loading / empty / error)
- ``askConfirm``    a modal that spells out the blast radius

These assertions keep a later edit from quietly reintroducing a blocking
native dialog or dropping the modal host. They read sources from disk,
so an edit in dev reflects in the next assertion.
"""
from __future__ import annotations

import pathlib

_COMP_DIR = pathlib.Path("src/vector_service/static/dashboard/components")
_PANELS = sorted(_COMP_DIR.glob("*.js"))


def _code(path):
    """Component source with full-line comments removed.

    Several files *mention* ``alert()`` / ``confirm()`` in the comments
    that explain why they were removed (and the markdown sanitiser keeps
    the literal ``javascript:alert(1)`` in a comment as an example);
    only real call sites are of interest here.
    """
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue
        kept.append(line)
    return "\n".join(kept)


def test_no_native_dialogs_remain():
    """Every panel speaks through the shared components.

    ``confirm(`` here is the lowercase native call: ``askConfirm(`` and
    ``settleConfirm(`` are spelled with a capital C and so do not match.
    """
    offenders = []
    for panel in _PANELS:
        code = _code(panel)
        if "alert(" in code or "confirm(" in code:
            offenders.append(panel.name)
    assert offenders == [], f"native dialog call sites remain in {offenders}"


def test_feedback_module_exports_the_convention():
    src = (_COMP_DIR / "feedback.js").read_text(encoding="utf-8")
    for name in ("StatusBanner", "BusyButton", "EmptyState", "NoticeBar", "ConfirmHost"):
        assert f"export const {name} = defineComponent(" in src, f"{name} not exported"
    for fn in ("askConfirm", "settleConfirm", "notify", "dismissNotice"):
        assert f"export function {fn}(" in src, f"{fn} not exported"


def test_feedback_module_has_no_app_import():
    """feedback.js is imported BY app.js (which mounts the two hosts), so
    an app.js import here would close a cycle and break module init."""
    src = (_COMP_DIR / "feedback.js").read_text(encoding="utf-8")
    assert "from './app.js'" not in src


def test_destructive_panels_confirm_through_the_modal():
    """Every panel that destroys or rewrites state asks first, and asks
    with the modal — a native confirm() cannot name what it is about to
    delete."""
    expected = {
        "browse.js", "collections.js", "databases.js", "records.js",
        "ops-queue.js", "ops-consistency.js", "ops-reindex.js",
    }
    for name in sorted(expected):
        src = (_COMP_DIR / name).read_text(encoding="utf-8")
        assert "askConfirm(" in src, f"{name} does not confirm via askConfirm()"


def test_app_mounts_the_global_feedback_hosts():
    src = (_COMP_DIR / "app.js").read_text(encoding="utf-8")
    assert "notice-bar" in src, "NoticeBar is not mounted"
    assert "confirm-host" in src, "ConfirmHost is not mounted"
