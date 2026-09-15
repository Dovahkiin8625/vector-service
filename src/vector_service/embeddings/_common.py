"""Shared helpers for backend implementations.

The text embedder, image embedder, multimodal embedder, and reranker
all share a small handful of cross-cutting concerns:

- detecting whether a model is already on disk (``dir_has_model``);
- flushing the CUDA allocator at unload time (``release_cuda_cache``);
- resolving the user-configured device string to a concrete
  ``"cuda"`` / ``"cpu"`` (``resolve_device``).

These three helpers are the canonical implementations used by every
backend. Earlier backend modules carried private copies of the same
logic; that drift is now consolidated here so behaviour stays
identical across families.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def dir_has_model(path: str | Path, *, markers: Iterable[str] = (
    "config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "model.onnx",
)) -> bool:
    """``True`` if ``path`` exists as a directory and contains any
    marker from ``markers``.

    Each marker can be either an exact filename (``"config.json"``)
    or a glob pattern (``"*.safetensors"``). The function returns
    ``True`` as soon as any marker resolves to a present file.

    Args:
        path: directory to inspect.
        markers: filenames (exact) or glob patterns whose presence
            in the directory is taken as "model is downloaded".
            Defaults to the union of the markers used by
            ``BGEM3Embedder`` and ``CrossEncoderReranker``.

    Returns:
        ``True`` if any marker resolves, ``False`` otherwise (also
        when ``path`` doesn't exist or isn't a directory).
    """
    if not os.path.isdir(path):
        return False
    try:
        present = set(os.listdir(path))
    except OSError:
        return False
    p = Path(path)
    for marker in markers:
        if any(c in marker for c in "*?["):
            # glob pattern (fnmatch-style; ``Path.glob`` handles it)
            if any(p.glob(marker)):
                return True
        elif marker in present:
            return True
    return False


def release_cuda_cache() -> None:
    """Best-effort CUDA cache flush; safe on CPU-only hosts.

    Catches every exception so a CUDA OOM during unload doesn't
    prevent the rest of the unload sequence from completing.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    """Map ``auto|cpu|cuda`` to a concrete device string.

    ``"cpu"`` returns ``"cpu"`` unconditionally. ``"cuda"`` returns
    ``"cuda"`` without probing — the backend's ``load()`` path is
    responsible for raising the family-specific error if CUDA really
    isn't available, so the constructor stays cheap and unit-testable.
    ``"auto"`` probes torch and falls back to ``"cpu"`` if CUDA is
    missing or torch can't be imported.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
