"""Translate a generic filter dict to a Milvus expression.

v1 scope: only top-level key/value equality, AND-composed.
Anything more complex raises FilterTranslationError.
"""
from __future__ import annotations

import json
from typing import Any


class FilterTranslationError(ValueError):
    """The filter cannot be translated to the backend expression."""


def translate_filter(filter: dict[str, Any] | None) -> str:
    """Translate a filter dict to a Milvus-style expression.

    Args:
        filter: dict of {field_name: value}, AND-composed. Values must be
            JSON-compatible scalars (str, int, float, bool, None).
        Empty/None returns "" (no filter).

    Returns:
        Milvus expression string like `metadata["k"] == "v"`.

    Raises:
        FilterTranslationError: if any value is a non-scalar (dict/list),
            or any key/value pair cannot be safely represented.
    """
    if not filter:
        return ""

    parts: list[str] = []
    for k, v in filter.items():
        _validate_key(k)
        parts.append(f'metadata["{k}"] {_op(v)} {_render_value(v)}')
    return " and ".join(parts)


def _validate_key(k: str) -> None:
    if not isinstance(k, str) or not k:
        raise FilterTranslationError(f"filter key must be non-empty str, got {k!r}")
    # Prevent injection of ] or similar characters that could break the expression
    if any(c in k for c in ['"', "\\", "\n", "\r"]):
        raise FilterTranslationError(f"filter key contains illegal chars: {k!r}")


def _op(v: Any) -> str:
    return "=="


def _render_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, str):
        # Use json.dumps to escape and add double quotes
        return json.dumps(v, ensure_ascii=False)
    raise FilterTranslationError(
        f"unsupported filter value type: {type(v).__name__} (v1 only supports scalars)"
    )
