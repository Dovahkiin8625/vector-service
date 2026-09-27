"""Runtime status of the Docling parser and its per-profile converters.

The dashboard needs the same honest resource picture for the *parser*
that :mod:`vector_service.core.model_info` gives the four inference
families — but Docling's shape is different:

- one :class:`~docling.document_converter.DocumentConverter` is cached
  per *profile* (``standard`` / ``native`` / ``vlm``) on the process
  singleton; a profile is "warm" iff its converter has been built;
- a converter holds its pipelines in ``initialized_pipelines``;
- in recent Docling the threaded ``StandardPdfPipeline`` keeps one
  model *stage* per attribute (``layout_model``, ``table_model``,
  ``ocr_model``, …), while ``VlmPipeline`` keeps its stages in the
  ``build_pipe`` list. Attribute layouts vary between Docling
  versions, so discovery here is a **generic bounded attribute
  traversal** — no hardcoded model-attribute contract;
- the RapidOCR stage runs entirely on ONNX Runtime: its session has
  no ``torch.nn.Module``, so its weights are *not* introspectable.
  Rather than silently omitting it we still report the component with
  ``resource_visible=False`` so the UI can mark the gap explicitly.

Never raises into the caller: status powers ``GET /v1/system/status``
which must stay fail-open.
"""
from __future__ import annotations

import importlib.util
import types
from collections import Counter
from typing import Any

from vector_service.core.config import get_settings
from vector_service.parsers.base import (
    PROFILE_NATIVE,
    PROFILE_STANDARD,
    PROFILE_VLM,
)

#: Profiles exposed in the status payload (concrete cached converters).
CONCRETE_PROFILES: tuple[str, ...] = (
    PROFILE_STANDARD,
    PROFILE_NATIVE,
    PROFILE_VLM,
)

#: Stage roots can sit deeper than the inference wrappers
#: (VlmConvertModel → engine → … → nn.Module), so the walk here is a
#: little deeper than ``model_info``'s wrapper-oriented cap.
_MAX_DEPTH = 6

#: Instance attributes that are pipeline bookkeeping, never models.
_SKIP_ATTRS: frozenset[str] = frozenset({
    "pipeline_options",
    "options",
    "artifacts_path",
    "force_backend_text",
    "keep_images",
    "enrichment_pipe",
})


def docling_available() -> bool:
    """``True`` when the optional ``docling`` package is importable."""
    try:
        return importlib.util.find_spec("docling") is not None
    except Exception:  # noqa: BLE001 — import machinery must never raise
        return False


def parser_status(parser: Any) -> dict[str, Any]:
    """Build the parser block for ``GET /v1/system/status``.

    Shape (every field is stable for the dashboard)::

        {
          "backend": "docling",
          "available": bool,
          "profiles": {
            "<profile>": {
              "warm": bool,
              "pipeline_classes": [...],
              "components": [...],
              "torch_param_count": int | None,
              "torch_memory_bytes": int | None,
              "device": str | None,
              "model_free": bool,
            },
          },
          "config": {device, ocr_langs, images_scale, vlm_preset, save_images},
        }
    """
    settings = get_settings().parser
    converters = getattr(parser, "_converters", {})

    try:
        profiles = {
            profile: _describe_profile(converters.get(profile))
            for profile in CONCRETE_PROFILES
        }
    except Exception:  # noqa: BLE001 — fail-open per module contract
        profiles = {
            profile: {"warm": False, "error": "status introspection failed"}
            for profile in CONCRETE_PROFILES
        }

    return {
        "backend": "docling",
        "available": docling_available(),
        "profiles": profiles,
        "config": {
            "device": settings.device,
            "ocr_langs": list(settings.ocr_langs),
            "images_scale": settings.images_scale,
            "vlm_preset": settings.vlm_preset,
            "save_images": bool(settings.save_images),
        },
    }


# ---------------------------------------------------------------------------
# Per-profile / per-pipeline introspection
# ---------------------------------------------------------------------------


def _describe_profile(converter: Any) -> dict[str, Any]:
    """Describe one cached converter (or the cold-profile placeholder)."""
    if converter is None:
        return {
            "warm": False,
            "pipeline_count": 0,
            "pipeline_classes": [],
            "components": [],
            "torch_param_count": None,
            "torch_memory_bytes": None,
            "device": None,
            "model_free": False,
        }

    pipelines = _converter_pipelines(converter)
    components: list[dict[str, Any]] = []
    seen_roots: set[int] = set()
    pipeline_classes: list[str] = []

    for pipeline in pipelines:
        cls_name = type(pipeline).__name__
        if cls_name not in pipeline_classes:
            pipeline_classes.append(cls_name)
        for source, root in _stage_roots(pipeline):
            if root is None or id(root) in seen_roots:
                continue
            seen_roots.add(id(root))
            component = _describe_component(source, root, pipeline)
            if component is not None:
                components.append(component)

    param_count = sum(
        c["param_count"]
        for c in components
        if c.get("torch") and c.get("param_count")
    )
    memory_bytes = sum(
        c["memory_bytes"]
        for c in components
        if c.get("torch") and c.get("memory_bytes")
    )
    devices = Counter(
        c["device"]
        for c in components
        if c.get("torch") and c.get("device")
    )

    return {
        "warm": True,
        "pipeline_count": len(pipelines),
        "pipeline_classes": pipeline_classes,
        "components": components,
        "torch_param_count": param_count or None,
        "torch_memory_bytes": memory_bytes or None,
        "device": devices.most_common(1)[0][0] if devices else None,
        # Native / Simple pipelines hold pipelines without torch weights
        # — state that plainly instead of showing "0 resources". A warm
        # converter whose pipelines have never been built (lazy Docling
        # without warm pre-init) is a different state: idle, not free.
        "model_free": bool(pipelines)
        and not any(c.get("torch") for c in components),
    }


def _converter_pipelines(converter: Any) -> list[Any]:
    """Return every pipeline instance a converter currently holds.

    ``initialized_pipelines`` is a dict keyed in recent Docling by
    ``(pipeline_cls, options_hash)``; accept any key shape and degrade
    to an empty list for test doubles / future versions.
    """
    initialized = getattr(converter, "initialized_pipelines", None)
    try:
        return list(initialized.values())
    except Exception:  # noqa: BLE001
        return []


def _stage_roots(pipeline: Any) -> list[tuple[str, Any]]:
    """Collect ``(source_name, stage_object)`` candidates from a pipeline.

    Generic on purpose: plain attributes plus list/tuple-style pipe
    holders (``build_pipe``). Non-model attributes are filtered later
    by :func:`_describe_component` (no torch weights, no known engine).
    """
    roots: list[tuple[str, Any]] = []
    try:
        attrs = vars(pipeline)
    except TypeError:
        return roots
    for name, value in attrs.items():
        if name.startswith("__") or name in _SKIP_ATTRS or value is None:
            continue
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, type) or item is None:
                    continue
                roots.append((f"{name}[{idx}]", item))
        elif isinstance(value, (dict, str, bytes, int, float, bool)):
            continue
        elif isinstance(value, type):
            continue
        else:
            roots.append((name, value))
    return roots


def _describe_component(
    source: str, root: Any, pipeline: Any
) -> dict[str, Any] | None:
    """Describe one model stage, or return ``None`` to skip it.

    Inclusion rules:

    - torch-bearing stages (layout / table / VLM / small helpers) are
      always included with their weight stats;
    - OCR stages are included even without torch weights and flagged
      ``resource_visible=False`` (ONNX Runtime sessions);
    - everything else model-free (assemblers, reading order, …) is
      dropped to keep the list honest about what actually costs
      resources.
    """
    kind = _classify(source, root, pipeline)
    info = _describe_torch(root)

    if info is None and kind != "ocr":
        return None

    component: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "class_name": type(root).__name__,
        "torch": info is not None,
    }
    if info is not None:
        component.update(
            device=info["device"],
            dtype=info["dtype"],
            param_count=info["param_count"],
            memory_bytes=info["memory_bytes"],
        )
    if kind == "ocr":
        component["engine"] = _ocr_engine(pipeline)
        component["resource_visible"] = info is not None
    if kind == "vlm":
        component["ref"] = get_settings().parser.vlm_preset
    return component


def _classify(source: str, root: Any, pipeline: Any) -> str:
    """Map a stage to a coarse ``kind`` used by the UI for labels."""
    haystack = "{} {} {}".format(
        source, type(root).__name__, type(pipeline).__name__
    ).lower()
    # Order matters: "layout_postprocessing_model" must not become layout.
    if "ocr" in haystack:
        return "ocr"
    if "postprocess" in haystack or "preprocess" in haystack:
        return "other"
    if "table" in haystack:
        return "table"
    if "layout" in haystack:
        return "layout"
    if "vlm" in haystack:
        return "vlm"
    return "other"


def _ocr_engine(pipeline: Any) -> str:
    """Best-effort OCR engine name (``onnxruntime`` / …) from options."""
    options = getattr(pipeline, "pipeline_options", None)
    ocr_options = getattr(options, "ocr_options", None)
    return str(getattr(ocr_options, "backend", "onnxruntime"))


# ---------------------------------------------------------------------------
# Bounded torch walk (stage-root oriented, mirrors model_info's approach)
# ---------------------------------------------------------------------------


def _describe_torch(root: Any) -> dict[str, Any] | None:
    """Weight stats for all torch modules reachable from ``root``.

    Returns ``None`` when torch is unavailable or no module is
    reachable. De-duplicates tensors by identity so shared submodules
    are counted once.
    """
    try:
        import torch
        from torch import nn
    except Exception:  # noqa: BLE001 — torch is optional
        return None

    modules = _find_modules(root, nn, torch.Tensor)
    if not modules:
        return None

    param_ids: set[int] = set()
    buffer_ids: set[int] = set()
    param_count = 0
    memory_bytes = 0
    devices: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()

    for module in modules:
        for tensor in module.parameters(recurse=True):
            if id(tensor) in param_ids:
                continue
            param_ids.add(id(tensor))
            param_count += int(tensor.numel())
            memory_bytes += int(tensor.numel()) * int(tensor.element_size())
            devices[str(tensor.device)] += 1
            dtypes[str(tensor.dtype).removeprefix("torch.")] += 1
        for tensor in module.buffers(recurse=True):
            if id(tensor) in buffer_ids:
                continue
            buffer_ids.add(id(tensor))
            memory_bytes += int(tensor.numel()) * int(tensor.element_size())

    return {
        "device": devices.most_common(1)[0][0] if devices else None,
        "dtype": dtypes.most_common(1)[0][0] if dtypes else None,
        "param_count": param_count or None,
        "memory_bytes": memory_bytes or None,
    }


def _find_modules(root: Any, nn: Any, tensor_cls: Any) -> list[Any]:
    """Collect every distinct ``torch.nn.Module`` reachable from ``root``."""
    found: list[Any] = []
    seen_objects: set[int] = set()

    def visit(obj: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or obj is None:
            return
        oid = id(obj)
        if oid in seen_objects:
            return
        seen_objects.add(oid)

        if isinstance(obj, nn.Module):
            found.append(obj)
            # Submodules are covered by parameters(recurse=True); don't
            # walk the module dict and risk back-reference cycles.
            return

        if isinstance(obj, (str, bytes, int, float, bool, bytearray)):
            return
        if isinstance(obj, tensor_cls):
            return
        # NOTE: do NOT skip every callable — Docling stages implement
        # ``__call__`` while holding the real engine/model in instance
        # attributes. Only prune actual functions/methods/classes.
        if isinstance(
            obj,
            (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodType,
                types.BuiltinMethodType,
                type,
            ),
        ):
            return

        children: list[Any] = []
        if isinstance(obj, dict):
            children.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children.extend(obj)
        else:
            attrs: dict[str, Any] = {}
            try:
                attrs.update(vars(obj))
            except TypeError:
                pass
            # vars() omits __slots__ attributes; collect them across
            # the MRO so slot-based wrappers stay visible.
            for base in type(obj).__mro__:
                slot_names = getattr(base, "__slots__", ())
                if isinstance(slot_names, str):
                    slot_names = (slot_names,)
                for name in slot_names:
                    if name.startswith("__") or name in attrs:
                        continue
                    try:
                        attrs[name] = getattr(obj, name)
                    except AttributeError:
                        pass
            for name, value in attrs.items():
                if name.startswith("__"):
                    continue
                children.append(value)
        for child in children:
            visit(child, depth + 1)

    visit(root, 0)

    unique: list[Any] = []
    module_ids: set[int] = set()
    for module in found:
        if id(module) not in module_ids:
            module_ids.add(id(module))
            unique.append(module)
    return unique
