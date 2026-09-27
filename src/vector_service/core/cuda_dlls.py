"""Make pip-installed NVIDIA CUDA/cuDNN DLLs discoverable on Windows.

``onnxruntime-gpu`` loads its CUDA EP dependencies (cudart, cuBLAS,
cuBLASLt, cuFFT, cuRAND, cuDNN, NVRTC/NvJitLink) by bare DLL name.
The ``nvidia-*`` wheels pulled in by the ``onnxruntime-gpu[cuda]``
extra ship those DLLs under ``site-packages/nvidia/.../bin`` without
touching PATH or the loader configuration. On a host without a
system CUDA toolkit the EP then fails to initialise — ORT logs
something like ``Error loading onnxruntime_providers_cuda.dll which
depends on cublasLt64_13.dll which is missing``, silently drops
``CUDAExecutionProvider`` from the session, and RapidOCR responds by
falling back to CPU with its "CUDAExecutionProvider is not in
available providers" warning.

Older NVIDIA wheels use a per-library layout
(``nvidia/cublas/bin``); the CUDA 13 wheels consolidate everything
under ``nvidia/cu13/bin/x86_64``. We register every ``.../bin``
directory (and architecture subdirectory) that actually contains
DLLs, both via :func:`os.add_dll_directory` and by prepending PATH.

**torch/lib must come first.** The cu130 torch wheel ships its own
cuDNN 9 sub-library set under ``site-packages/torch/lib``. cuDNN is
split into several DLLs (``cudnn64_9.dll`` plus graph/engines/ops/…
sublibraries) whose versions must match; a process that loads the
slim ``cudnn64_9.dll`` from the ``nvidia-cudnn-cu13`` wheel first
makes every later ``LoadLibrary("cudnn64_9.dll")`` — including
torch's — resolve to that module, and torch then fails with
``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH``. We therefore do NOT
install the ORT ``cudnn`` extra and register torch/lib ahead of the
nvidia directories, so the single torch-bundled cuDNN set serves
both torch and ORT; the nvidia dirs only fill the gaps torch doesn't
ship (cuFFT, cuRAND).

:func:`enable_pip_cuda_dlls` must run before the first ONNX Runtime
session is created — main.py calls it at import time.
"""
from __future__ import annotations

import glob
import os
import site
import sys
import sysconfig


def enable_pip_cuda_dlls() -> list[str]:
    """Register pip-installed NVIDIA DLL directories with the loader.

    Returns the directories that were added. No-op (empty list) on
    non-Windows hosts or when no ``nvidia`` package tree is present.
    Idempotent — repeat calls are cheap and harmless.
    """
    if sys.platform != "win32":
        return []

    site_dirs: list[str] = [
        d
        for d in dict.fromkeys(
            (sysconfig.get_path("purelib"), *site.getsitepackages())
        )
        if d
    ]

    dll_dirs: list[str] = []

    # torch/lib first: one consistent cuDNN set for torch AND onnxruntime
    # (module docstring explains the sublibrary-mismatch failure).
    for base in site_dirs:
        torch_lib = os.path.join(base, "torch", "lib")
        if os.path.isdir(torch_lib) and glob.glob(
            os.path.join(torch_lib, "*.dll")
        ):
            dll_dirs.append(os.path.normpath(torch_lib))

    for base in site_dirs:
        nvidia_root = os.path.join(base, "nvidia")
        # nvidia/<lib>/bin (legacy per-library wheels) and
        # nvidia/cu13/bin/<arch> (consolidated CUDA 13 wheels).
        for bin_dir in glob.glob(os.path.join(nvidia_root, "*", "bin")):
            for d in (bin_dir, *glob.glob(os.path.join(bin_dir, "*"))):
                if os.path.isdir(d) and glob.glob(os.path.join(d, "*.dll")):
                    dll_dirs.append(os.path.normpath(d))

    dll_dirs = list(dict.fromkeys(dll_dirs))
    for d in dll_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            # Directory vanished between the glob and the call — PATH
            # below still covers it if it reappears; nothing else to do.
            pass
    if dll_dirs:
        os.environ["PATH"] = os.pathsep.join(
            [*dll_dirs, os.environ.get("PATH", "")]
        )
    return dll_dirs
