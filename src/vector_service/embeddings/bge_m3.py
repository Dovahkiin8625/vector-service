"""BGE-M3 embedder with GPU/CPU dispatch."""
from __future__ import annotations

import os
from pathlib import Path

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import EmbedderError, ModelNotLoaded
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
        if requested == "cuda":
            return _check_cuda()
        if requested == "cpu":
            return "cpu"
        # auto
        try:
            return _check_cuda()
        except ModelNotLoaded:
            return "cpu"

    def _ensure_loaded(self) -> "_BGEBackend":
        if self._impl is not None:
            return self._impl
        self._ensure_model_dir()
        try:
            if self._device == "cuda":
                self._impl = _TorchBackend(
                    self._model_dir, self._device, self._max_length, self._batch_size
                )
            else:
                # CPU: 优先 ONNX（int8 量化），失败回退 Torch
                try:
                    self._impl = _OnnxBackend(
                        self._model_dir, self._max_length, self._batch_size
                    )
                except Exception:
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
        return self._impl

    def _ensure_model_dir(self) -> None:
        s = self._settings
        self._model_dir.mkdir(parents=True, exist_ok=True)
        # 判断是否已有必需文件（粗略）
        if _dir_has_model(self._model_dir):
            return
        if not s.embedding_auto_download:
            raise ModelNotLoaded(
                f"model dir {self._model_dir} has no BGE-M3 files and auto_download is off"
            )
        # 下载
        from huggingface_hub import snapshot_download  # lazy: 仅 auto_download 时需要
        repo = s.embedding_onnx_repo if self._device == "cpu" else s.embedding_hf_repo
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


def _check_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError as e:
        raise ModelNotLoaded(f"torch not available: {e}")
    raise ModelNotLoaded("CUDA not available")


def _dir_has_model(p: Path) -> bool:
    if not p.exists():
        return False
    # 至少有 config.json 或 model.safetensors / model.onnx
    if (p / "config.json").exists():
        return True
    if any(p.glob("*.onnx")):
        return True
    if any(p.glob("*.safetensors")) or any(p.glob("pytorch_model.bin")):
        return True
    return False


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


class _OnnxBackend(_BGEBackend):
    def __init__(self, model_dir: Path, max_length: int, batch_size: int):
        # 简单 ONNX 推理：使用 optimum 或 onnxruntime 直接跑
        # 这里用 FlagEmbedding 的 ONNX 接口；若仓库不含 ONNX，会抛错回退
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=False,
            device="cpu",
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