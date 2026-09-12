"""Cross-encoder reranker backed by sentence-transformers ``CrossEncoder``."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import RerankerNotLoaded
from vector_service.rerankers.base import Reranker, ScoredHit

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder


def _dir_has_model(path: str) -> bool:
    """True if ``path`` already contains usable model files.

    Checks for any of ``config.json``, ``tokenizer_config.json``,
    ``model.safetensors``, ``pytorch_model.bin``, or ``model.onnx``.
    This matches the same convention used by ``BGEM3Embedder``.
    """
    import os

    if not os.path.isdir(path):
        return False
    markers = {
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model.onnx",
    }
    try:
        present = set(os.listdir(path))
    except OSError:
        return False
    return bool(markers & present)


def _resolve_device(device: str) -> str:
    """Map ``auto|cpu|cuda`` to a concrete torch device string.

    Mirrors ``BGEM3Embedder._resolve_device``: on ``auto``, try CUDA
    first; if import or probing fails, fall back to CPU.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder reranker."""

    model_name = "bge-reranker-v2-m3"

    def __init__(self, settings: Settings | None = None) -> None:
        s = (settings or get_settings()).reranker
        self._device: str = _resolve_device(s.device)
        self._batch_size: int = s.batch_size
        self._max_length: int = s.max_length
        self._model_dir: str = s.model_dir
        self._auto_download: bool = s.auto_download
        self._download_source: str = s.download_source
        self._hf_repo: str = s.hf_repo
        self._ms_repo: str = s.ms_repo
        self._impl: CrossEncoder | None = None
        self._lock = threading.Lock()

    # ---- lifecycle --------------------------------------------------

    def load(self) -> None:
        """Eagerly load weights. Idempotent."""
        with self._lock:
            if self._impl is not None:
                return
            self._ensure_model_dir()
            from sentence_transformers import CrossEncoder  # local import

            self._impl = CrossEncoder(
                self._model_dir,
                max_length=self._max_length,
                device=self._device,
            )
            # warmup so first request isn't a cold start
            try:
                self._impl.predict(
                    [("warmup", "")] * 5,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                )
            except Exception as exc:  # pragma: no cover — defensive
                self._impl = None
                raise RerankerNotLoaded(
                    f"warmup failed for {self.model_name}: {exc}"
                ) from exc

    def _ensure_model_dir(self) -> None:
        """Download weights if needed; otherwise raise ``RerankerNotLoaded``."""
        if _dir_has_model(self._model_dir):
            return
        if not self._auto_download:
            raise RerankerNotLoaded(
                f"model dir {self._model_dir!r} missing required files "
                f"and auto_download is disabled"
            )
        if self._download_source == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError as exc:  # pragma: no cover
                raise RerankerNotLoaded(
                    "modelscope is not installed; install the [embed] extra "
                    "or switch download_source to 'huggingface'"
                ) from exc
            snapshot_download(
                self._ms_repo, local_dir=self._model_dir
            )
        else:  # huggingface
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover
                raise RerankerNotLoaded(
                    "huggingface-hub is not installed"
                ) from exc
            snapshot_download(
                self._hf_repo, local_dir=self._model_dir
            )
        if not _dir_has_model(self._model_dir):  # pragma: no cover
            raise RerankerNotLoaded(
                f"download completed but {self._model_dir!r} still lacks "
                f"required files"
            )

    def _ensure_loaded(self) -> None:
        if self._impl is None:
            raise RerankerNotLoaded(
                f"reranker {self.model_name!r} is not loaded; "
                f"call load() first"
            )

    # ---- inference --------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[ScoredHit]:
        self._ensure_loaded()
        assert self._impl is not None  # for type checkers
        pairs = [(query, d) for d in documents]
        raw = self._impl.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        # raw may be numpy array or list — coerce to plain floats
        scores = [float(s) for s in raw]
        hits = sorted(
            [ScoredHit(index=i, score=s) for i, s in enumerate(scores)],
            key=lambda h: h.score,
            reverse=True,
        )
        if top_n is not None:
            hits = hits[:top_n]
        return hits


# Register self
from vector_service.rerankers.registry import RERANKER_REGISTRY

RERANKER_REGISTRY["bge-reranker-v2-m3"] = CrossEncoderReranker
