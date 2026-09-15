"""Cross-encoder reranker backed by sentence-transformers ``CrossEncoder``."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from vector_service.core.config import RerankerSettings, Settings, get_settings
from vector_service.core.errors import RerankerNotLoaded
from vector_service.embeddings import _common
from vector_service.rerankers.base import Reranker, ScoredHit

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder reranker."""

    model_name = "bge-reranker-v2-m3"

    def __init__(self, settings: Settings | RerankerSettings | None = None) -> None:
        # Two call sites feed us different shapes:
        #   1. Lifespan eager-load passes the full ``Settings`` (lifespan.py
        #      builds the reranker with the root config so it can also pass
        #      it to embedders/multimodal embedders).
        #   2. The hot-load route ``POST /v1/models/{id}/load`` passes the
        #      ``RerankerSettings`` block directly (the family table in
        #      ``api/models.py`` resolves nested blocks).
        # Unwrap the full Settings when needed so both paths work without
        # requiring the caller to know which form to send. We duck-type
        # on ``.reranker`` rather than ``isinstance`` so test stubs that
        # mimic ``RerankerSettings`` are accepted too.
        if settings is None:
            s = get_settings().reranker
        elif hasattr(settings, "reranker") and not hasattr(settings, "device"):
            # Likely the full Settings wrapper (has nested .reranker).
            s = settings.reranker
        else:
            # Already the reranker block.
            s = settings
        self._device: str = _common.resolve_device(s.device)
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

    def unload(self) -> None:
        """Release the CrossEncoder and free CUDA cache if applicable.

        Idempotent. The underlying ``CrossEncoder`` exposes a
        ``model`` attribute (the underlying transformer); we move it to
        CPU so the CUDA caching allocator can reclaim VRAM, then drop
        the reference. Subsequent ``rerank`` calls will lazily reload
        via ``_ensure_loaded``.
        """
        with self._lock:
            impl = self._impl
            self._impl = None
        if impl is not None:
            try:
                inner = getattr(impl, "model", None)
                if inner is not None:
                    inner.to("cpu")
            except Exception:
                pass
            try:
                impl.__dict__.clear()
            except Exception:
                pass
        _common.release_cuda_cache()

    def _ensure_model_dir(self) -> None:
        """Download weights if needed; otherwise raise ``RerankerNotLoaded``."""
        if _common.dir_has_model(self._model_dir):
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
        if not _common.dir_has_model(self._model_dir):  # pragma: no cover
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
