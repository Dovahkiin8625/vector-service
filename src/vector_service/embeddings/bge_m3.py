"""BGE-M3 embedder with GPU/CPU dispatch."""
from __future__ import annotations

from pathlib import Path

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import EmbedderError, ModelNotLoaded
from vector_service.embeddings import _common
from vector_service.embeddings.base import Embedder

_QUERY_PREFIX = "为这个句子生成表示以用于检索："


class BGEM3Embedder(Embedder):
    dim = 1024
    model_name = "bge-m3"

    def __init__(
        self,
        settings: Settings | None = None,
    ):
        s = settings or get_settings()
        self._settings = s
        self._device = self._resolve_device(s.embedding_device)
        self._batch_size = s.embedding_batch_size
        self._max_length = s.embedding_max_length
        self._model_dir = Path(s.embedding_model_dir)
        self._impl: _BGEBackend | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        return _common.resolve_device(requested)

    def _ensure_loaded(self) -> "_BGEBackend":
        if self._impl is not None:
            return self._impl
        self._load_internal()
        return self._impl  # type: ignore[return-value]

    def load(self) -> None:
        """Eagerly download (if needed), instantiate the backend, and warm up.

        Idempotent: a second call is a no-op. Raises `ModelNotLoaded` on
        failure so the lifespan handler can decide whether to stay up.
        """
        if self._impl is not None:
            return
        self._load_internal()

    def unload(self) -> None:
        """Release the loaded backend and free CUDA cache if applicable.

        Idempotent: safe to call before ``load`` or twice in a row.
        After this returns the next ``embed_documents`` / ``embed_query``
        call will lazily trigger a fresh ``load`` via ``_ensure_loaded``.
        """
        impl = self._impl
        self._impl = None
        if impl is not None:
            # Drop the inner model's reference first so GC + the CUDA
            # caching allocator can reclaim VRAM promptly.
            impl_attr = getattr(impl, "_model", None)
            if impl_attr is not None:
                try:
                    impl_attr.to("cpu")
                except Exception:
                    pass
            impl.__dict__.clear()
        _common.release_cuda_cache()

    def _load_internal(self) -> None:
        self._ensure_model_dir()
        try:
            if self._device == "cuda":
                self._impl = _TorchBackend(
                    self._model_dir, self._device, self._max_length, self._batch_size
                )
            else:
                # CPU：直接用 PyTorch 后端（ONNX int8 镜像在 ModelScope/HF 上经常
                # 缺失或不稳定，PyTorch 是通用可移植的最低公分母）。
                self._impl = _TorchBackend(
                    self._model_dir, "cpu", self._max_length, self._batch_size
                )
        except Exception as e:
            raise ModelNotLoaded(f"failed to load BGE-M3: {e}") from e

        # 预热
        try:
            self.embed_documents([""] * 5)
        except Exception as e:  # 预热失败不致命
            import structlog
            structlog.get_logger(__name__).warning("bge_m3_warmup_failed", error=str(e))

    def _ensure_model_dir(self) -> None:
        s = self._settings
        self._model_dir.mkdir(parents=True, exist_ok=True)
        # 判断是否已有必需文件（粗略）
        if _common.dir_has_model(self._model_dir):
            return
        if not s.embedding_auto_download:
            raise ModelNotLoaded(
                f"model dir {self._model_dir} has no BGE-M3 files and auto_download is off"
            )
        # 下载：按配置的 source 路由到 HF 或 ModelScope
        source = (s.embedding_download_source or "huggingface").lower()
        repo = s.embedding_ms_repo if source == "modelscope" else s.embedding_hf_repo
        if source == "modelscope":
            from modelscope import snapshot_download  # lazy
            snapshot_download(
                repo_id=repo,
                local_dir=str(self._model_dir),
            )
        else:
            from huggingface_hub import snapshot_download  # lazy
            snapshot_download(
                repo_id=repo,
                local_dir=str(self._model_dir),
                local_dir_use_symlinks=False,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        impl = self._ensure_loaded()
        return impl.encode(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        impl = self._ensure_loaded()
        return impl.encode([_QUERY_PREFIX + text], is_query=True)[0]


class _BGEBackend:
    def encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        raise NotImplementedError


class _TorchBackend(_BGEBackend):
    def __init__(self, model_dir: Path, device: str, max_length: int, batch_size: int):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=(device == "cuda"),
            device=device,
        )
        self._max_length = max_length
        self._batch_size = batch_size

    def encode(self, texts, is_query):
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [list(map(float, v)) for v in out["dense_vecs"]]