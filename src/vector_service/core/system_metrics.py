"""System metrics: CPU, memory, GPU, OS — cross-platform.

The dashboard's "总览" panel and the ``GET /v1/system/status`` endpoint
both depend on a single helper module so the shape of the metrics stays
consistent. ``psutil`` is the preferred source for CPU / memory /
per-process info on both Windows and Linux; we degrade gracefully when
it's not installed so the endpoint still works on a slim install.

GPU metrics are best-effort:

- ``pynvml`` (NVIDIA System Management Library) is tried first because
  it works without a full torch install and is the canonical way to
  query an NVIDIA card from a host process;
- ``torch.cuda`` is used as a fallback when pynvml is missing but torch
  is loaded anyway by an embedder — the embedder extras already pull
  torch in, so this is the common path in production;
- when neither is available we report ``gpus=[]`` with ``available=False``
  instead of raising — the dashboard still renders the rest of the
  panel.

All callers should treat every field as optional; ``system_snapshot``
returns the same dict shape regardless of platform or library
availability so the JS layer can render it without branching on
``psutil`` / ``pynvml`` presence.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from typing import Any

# ``psutil`` ships in uv.lock but we don't pin it as a hard dep yet — the
# import is best-effort and the helper degrades to ``os`` / ``shutil``
# when it's missing.
try:  # pragma: no cover — exercised on every supported platform
    import psutil  # type: ignore[import-not-found]

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


def _gather_cpu() -> dict[str, Any]:
    """Return CPU usage + core counts.

    ``psutil.cpu_percent`` is blocking for the first call; we keep the
    call short (~50ms) so the dashboard polling cycle doesn't stall.
    When psutil is missing we report ``percent=None`` and rely on
    ``os.cpu_count()`` for the logical-core count (always available).
    """
    logical = os.cpu_count() or 1
    if not _HAS_PSUTIL:
        return {"percent": None, "logical_cores": logical, "physical_cores": None}
    # ``interval=None`` returns the value since the previous call. The
    # first call always returns 0.0, which is fine — the dashboard
    # auto-refresh keeps the value moving within a couple of ticks.
    pct = psutil.cpu_percent(interval=None)  # type: ignore[attr-defined]
    phys = psutil.cpu_count(logical=False) or logical  # type: ignore[attr-defined]
    return {"percent": pct, "logical_cores": logical, "physical_cores": phys}


def _gather_memory() -> dict[str, Any]:
    """Total / used / percent for system RAM.

    Falls back to ``None`` placeholders when psutil isn't installed so
    the frontend can still render the row (with a "—" badge).
    """
    if not _HAS_PSUTIL:
        return {
            "total_bytes": None,
            "used_bytes": None,
            "available_bytes": None,
            "percent": None,
        }
    vm = psutil.virtual_memory()  # type: ignore[attr-defined]
    return {
        "total_bytes": int(vm.total),
        "used_bytes": int(vm.used),
        "available_bytes": int(vm.available),
        "percent": float(vm.percent),
    }


def _gather_process() -> dict[str, Any]:
    """Per-process RSS + CPU%. Optional but useful for capacity planning."""
    if not _HAS_PSUTIL:
        return {"rss_bytes": None, "cpu_percent": None}
    proc = psutil.Process(os.getpid())  # type: ignore[attr-defined]
    return {
        "rss_bytes": int(proc.memory_info().rss),
        "cpu_percent": float(proc.cpu_percent(interval=None)),
    }


def _gather_disk() -> dict[str, Any]:
    """Disk usage for the current working directory's filesystem.

    Cheap and stable across platforms. Useful for the dashboard so
    operators can spot a full disk before the embedder cache fill
    crashes a download.
    """
    try:
        path = os.getcwd()
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "percent": float(usage.used) * 100.0 / float(usage.total or 1),
        }
    except Exception as e:  # pragma: no cover — path may not exist on weird FS
        return {"path": None, "error": str(e)}


def _gather_gpu() -> list[dict[str, Any]]:
    """Best-effort GPU enumeration.

    Returns a list with one entry per visible device. ``available=False``
    is set when no usable library is importable so the UI can render a
    "no GPU detected" hint instead of an empty box.

    Detection chain (first hit wins):

    1. ``pynvml`` (from ``nvidia-ml-py``) — talks directly to the
       NVIDIA driver via libnvidia-ml. Doesn't require a CUDA-enabled
       torch; works on every host with the NVIDIA driver installed.
       This is the canonical way to query an NVIDIA GPU from a host
       process and the path that covers RTX 50-series (Blackwell,
       sm_120) when only a CPU-only torch wheel is installed.
    2. ``torch.cuda`` — used as a fallback when pynvml isn't installed
       but torch was. Requires torch built with the matching CUDA
       version (e.g. Blackwell needs CUDA 12.8+).
    3. ``torch.xpu`` — Intel discrete / iGPUs on Windows + Linux.

    The order matters: ``nvidia-ml-py`` ships a tiny wheel that
    talks to the driver directly, while ``torch.cuda`` is only useful
    when torch itself was built with CUDA support. We prefer the
    former because it's smaller, always up-to-date with the driver,
    and doesn't depend on the embedder extras being installed.
    """
    gpus: list[dict[str, Any]] = []
    # --- NVIDIA path: pynvml (re-exported by nvidia-ml-py) -----------
    try:  # pragma: no cover — depends on host
        # The ``nvidia-ml-py`` package re-exports the ``pynvml`` module
        # under its canonical name. ``import pynvml`` first so the
        # downstream API is the same regardless of which wheel is
        # actually installed.
        import pynvml  # type: ignore[import-not-found]

        try:
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                for idx in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpus.append(
                        {
                            "index": idx,
                            "vendor": "nvidia",
                            "name": name or f"GPU #{idx}",
                            "memory_total_bytes": int(mem.total),
                            "memory_used_bytes": int(mem.used),
                            "memory_percent": (
                                float(mem.used) / float(mem.total or 1) * 100.0
                            ),
                            "utilization_percent": float(util.gpu),
                        }
                    )
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            return [{"available": False, "error": f"pynvml: {e}"}]
    except Exception:
        pass
    if gpus:
        return gpus

    # --- torch.cuda fallback -----------------------------------------
    try:  # pragma: no cover — depends on host
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                free, total = torch.cuda.mem_get_info(idx)
                used = total - free
                gpus.append(
                    {
                        "index": idx,
                        "vendor": "nvidia",
                        "name": getattr(props, "name", f"GPU #{idx}"),
                        "memory_total_bytes": int(total),
                        "memory_used_bytes": int(used),
                        "memory_percent": (
                            float(used) / float(total or 1) * 100.0
                        ),
                        "utilization_percent": None,
                    }
                )
    except Exception:
        pass
    if gpus:
        return gpus

    # --- torch xpu (Intel / discrete GPUs on Windows + Linux) -------
    try:  # pragma: no cover
        import torch  # type: ignore[import-not-found]

        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            count = torch.xpu.device_count()  # type: ignore[attr-defined]
            for idx in range(count):
                name = torch.xpu.get_device_name(idx)  # type: ignore[attr-defined]
                gpus.append(
                    {
                        "index": idx,
                        "vendor": "intel",
                        "name": name or f"GPU #{idx}",
                        "memory_total_bytes": None,
                        "memory_used_bytes": None,
                        "memory_percent": None,
                        "utilization_percent": None,
                    }
                )
    except Exception:
        pass

    if gpus:
        return gpus

    # Build a diagnostic so the dashboard's "no GPU" hint points the
    # operator at the right missing dep instead of a vague message.
    diag: dict[str, Any] = {"available": False}
    try:
        import pynvml  # noqa: F401 — only checking presence
        diag["hint"] = "pynvml installed but returned 0 devices."
    except Exception:
        diag["hint"] = (
            "NVIDIA driver not reachable via pynvml/nvidia-ml-py. "
            "Install ``nvidia-ml-py`` (preferred) or "
            "``pip install nvidia-ml-py`` to query the driver directly."
        )
        diag["missing"] = "pynvml"
    try:
        import torch  # noqa: F401

        diag["torch_cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            cuda = getattr(torch.version, "cuda", None)
            diag["torch_cuda_built"] = cuda
            if cuda is None:
                diag.setdefault(
                    "hint",
                    "当前 torch 为 CPU-only 构建（``torch.version.cuda is None``）；"
                "新显卡（RTX 50 系列等）需 CUDA 12.8+，请安装匹配 CUDA 版本的 torch，"
                "或直接 ``pip install nvidia-ml-py`` 走 NVML 通道。",
                )
                diag["missing"] = "nvidia-ml-py / torch+CUDA"
    except Exception:
        pass
    return [diag]


def _gather_os() -> dict[str, Any]:
    """OS / kernel / arch — same shape on Windows and Linux."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "hostname": platform.node(),
    }


def system_snapshot() -> dict[str, Any]:
    """Build a single dict containing every metric the dashboard needs.

    Cheap to call: no I/O, just process-local introspection + a single
    nvml query. ``psutil``-backed sensors may return ``None`` for the
    very first call (psutil needs a baseline) — that's acceptable for
    the dashboard since it polls every few seconds.
    """
    return {
        "ts": time.time(),
        "os": _gather_os(),
        "cpu": _gather_cpu(),
        "memory": _gather_memory(),
        "process": _gather_process(),
        "disk": _gather_disk(),
        "gpus": _gather_gpu(),
        "psutil_available": _HAS_PSUTIL,
    }