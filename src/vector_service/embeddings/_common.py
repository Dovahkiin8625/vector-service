"""Shared helpers for backend implementations.

These are intentionally additive — backend files may still carry
local copies of the same helpers for backwards compatibility. New
backends should import from here.

Why a separate module? The text embedder, image embedder, multimodal
embedder, and reranker all share a small handful of cross-cutting
concerns:

- detecting whether a model is already on disk (``_dir_has_model``);
- flushing the CUDA allocator at unload time
  (``_release_cuda_cache``);
- resolving the user-configured device string to a concrete
  ``"cuda"`` / ``"cpu"`` (``_resolve_device``).

The implementations differ slightly between backends (markers vary
by format, device options differ), so each backend keeps its own
copy. ``_common.py`` provides the canonical implementations used by
new backends; older modules remain unchanged.
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
    file from ``markers``.

    Args:
        path: directory to inspect.
        markers: file names whose presence in the directory is taken
            as "model is downloaded". Defaults to the union of the
            markers used by BGEM3Embedder and CrossEncoderReranker.

    Returns:
        ``True`` if any marker is present, ``False`` otherwise (also
        when ``path`` doesn't exist or isn't a directory).
    """
    if not os.path.isdir(path):
        return False
    try:
        present = set(os.listdir(path))
    except OSError:
        return False
    return bool(set(markers) & present)


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
