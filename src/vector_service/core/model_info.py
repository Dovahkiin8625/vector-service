"""Runtime introspection of a loaded model instance.

``GET /v1/models`` reports not just *whether* a model is loaded but also
what the live instance costs to run: parameter count, weight footprint
(bytes), the device it sits on, and its parameter dtype. None of the
four concrete backend families (:mod:`vector_service.embeddings.*`,
:mod:`vector_service.rerankers.cross_encoder`) exposes those through a
common interface — the actual ``torch.nn.Module`` lives behind
different attribute names::

    BGEM3Embedder                       instance._impl._model.model (+ .colbert_linear / .sparse_linear)
    OpenCLIPVitL14ImageEmbedder         instance._model
    ChineseCLIPMultimodalEmbedder       instance._model
    CrossEncoderReranker                instance._impl.model

Rather than teach the API about each wrapper, :func:`describe_instance`
walks the instance's attributes (bounded depth + cycle-safe) and
collects every reachable :class:`torch.nn.Module`. Parameters are
de-duplicated by object identity — a child module's parameters appear
both under the parent's ``parameters()`` iterator and on their own when
the walk discovers the child directly, and BGE-M3's projection heads
are siblings (not descendants) of the main transformer.

The byte figure is the *weight footprint* (unique parameters +
buffers): it is the deterministic VRAM floor for a GPU model and the
resident-RAM cost for a CPU model. It deliberately does NOT include the
CUDA context, caching-allocator reserves, or inference activations —
``torch.cuda.memory_allocated`` reports those per *process* and cannot
be attributed to a single model once more than one is loaded.

Results are cached per instance in a :class:`weakref.WeakKeyDictionary`
so the dashboard's 2–5s polling of ``GET /v1/models`` doesn't re-walk a
500M-parameter object graph on every tick.
"""
from __future__ import annotations

import weakref
from collections import Counter
from typing import Any

# Attribute traversal is bounded — wrappers are at most ~3 levels deep
# (instance -> impl -> flag model -> nn.Module). The cap also stops us
# wandering into tokenizers / config objects that may hold wide object
# graphs.
_MAX_DEPTH = 4

# One introspection result per live instance. Keyed weakly so dropping
# the model on unload drops the entry too.
_cache: weakref.WeakKeyDictionary[Any, dict[str, Any] | None] = (
    weakref.WeakKeyDictionary()
)


def describe_instance(instance: Any) -> dict[str, Any] | None:
    """Return static runtime info for a loaded model wrapper.

    Returns a dict ``{"device", "dtype", "param_count",
    "memory_bytes"}`` (any field may be ``None`` when it cannot be
    determined), or ``None`` when the instance holds no discoverable
    ``torch.nn.Module`` (test doubles, pure-Python backends, or a host
    where torch is not installed). Never raises — introspection powers
    a listing endpoint and must not take a model card down with it.
    """
    if instance is None:
        return None
    # ``WeakKeyDictionary`` raises TypeError for non-weak-referenceable
    # keys (plain ``object()``, slotted classes without ``__weakref__``);
    # probe first and describe those without caching.
    try:
        weakref.ref(instance)
    except TypeError:
        return _build(instance)
    # ``None`` is a legitimate cached value, so guard with key
    # membership rather than truthiness.
    if instance in _cache:
        return _cache.get(instance)
    info = _build(instance)
    _cache[instance] = info
    return info


def _build(instance: Any) -> dict[str, Any] | None:
    try:
        import torch
        from torch import nn
    except Exception:  # noqa: BLE001 — torch is an optional/heavy import
        return None

    modules = _find_modules(instance, nn, torch.Tensor)
    if not modules:
        return None

    # De-duplicate Parameter / Buffer objects across all discovered
    # modules: a parent iterator yields the child's tensors, and some
    # wrappers expose the SAME module through more than one attribute.
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

    # Prefer the wrapper's own device record (all four concrete
    # backends set ``_device``) — it is the device the wrapper will
    # dispatch inference to even before every weight has migrated.
    device = getattr(instance, "_device", None)
    if not isinstance(device, str) or not device:
        device = devices.most_common(1)[0][0] if devices else None
    dtype = dtypes.most_common(1)[0][0] if dtypes else None

    return {
        "device": device,
        "dtype": dtype,
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
            # The module's own submodules are covered by
            # ``parameters(recurse=True)`` later; no need to walk its
            # __dict__ and risk crossing into tokenizer/back-refs.
            return

        # Only descend through plain instance attributes / containers.
        # Primitives, tensors, callables, types and modules are leaves.
        if isinstance(obj, (str, bytes, int, float, bool, bytearray)):
            return
        if tensor_cls is not None and isinstance(obj, tensor_cls):
            return
        if callable(obj) or isinstance(obj, type):
            return

        children: list[Any] = []
        if isinstance(obj, dict):
            children.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            children.extend(obj)
        else:
            try:
                attrs = vars(obj)
            except TypeError:
                return
            for name, value in attrs.items():
                # Skip dunder/private bookkeeping (locks, caches,
                # back-references); model attributes themselves are
                # single-underscore (_model, _impl), so keep those.
                if name.startswith("__"):
                    continue
                children.append(value)
        for child in children:
            visit(child, depth + 1)

    visit(root, 0)

    # Same module may surface via several attribute paths before its
    # parent/child relationship is considered; de-duplicate while
    # preserving discovery order.
    unique: list[Any] = []
    module_ids: set[int] = set()
    for module in found:
        if id(module) not in module_ids:
            module_ids.add(id(module))
            unique.append(module)
    return unique
