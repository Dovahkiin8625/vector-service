"""OpenCLIP ViT-L/14 image embedder (openai pretrained weights, 768d)."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from vector_service.core.config import ImageEmbeddingSettings
from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


# Indirection so tests can monkeypatch without us importing open_clip at
# module load (we want a clean ImportError path if open_clip_torch isn't
# installed in production).
def _create_model_and_transforms(name: str, pretrained: str, **kwargs):
    import open_clip  # lazy

    return open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)


def _snapshot_download(repo_id: str, local_dir: str) -> str:
    from huggingface_hub import snapshot_download  # lazy

    return snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)


def _dir_has_model(p: Path) -> bool:
    if not p.exists():
        return False
    return (p / "config.json").exists() or any(p.glob("*.bin")) or any(p.glob("*.safetensors"))


class OpenCLIPVitL14ImageEmbedder(ImageEmbedder):
    dim = 768
    model_name = "openclip-vit-l-14"

    def __init__(self, settings: ImageEmbeddingSettings | None = None):
        if settings is None:
            # Lazy import to avoid pulling core.config when the embedder is
            # used in tests that pass their own settings stub.
            from vector_service.core.config import get_settings

            s = get_settings().image_embedding
        else:
            s = settings
        self._settings = s
        self._device = self._resolve_device(s.device)
        self._batch_size = s.batch_size
        self._model_dir = Path(s.model_dir)
        self._model = None
        self._preprocess = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cuda":
            return _check_cuda()
        if requested == "cpu":
            return "cpu"
        # auto
        try:
            return _check_cuda()
        except Exception:
            return "cpu"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._load_internal()

    def load(self) -> None:
        if self._model is not None:
            return
        self._load_internal()

    def _load_internal(self) -> None:
        self._ensure_model_dir()
        try:
            self._model, self._preprocess, _ = _create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=self._device
            )
        except Exception as e:
            raise ModelNotLoadedForImages(
                f"failed to load OpenCLIP ViT-L/14: {e}"
            ) from e

        # Warmup: single image to materialize kernels. Generate a tiny
        # valid PNG (1x1 black) — a bare PNG header is rejected by PIL.
        try:
            import io as _io

            tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
            buf = _io.BytesIO()
            tiny.save(buf, format="PNG")
            self.embed_images([ImageInput(data=buf.getvalue(), mime="image/png")])
        except Exception as e:  # warmup failures are non-fatal
            import structlog

            structlog.get_logger(__name__).warning("openclip_warmup_failed", error=str(e))

    def _ensure_model_dir(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if _dir_has_model(self._model_dir):
            return
        if not self._settings.auto_download:
            raise ModelNotLoadedForImages(
                f"model dir {self._model_dir} has no OpenCLIP weights and auto_download is off"
            )
        _snapshot_download(self._settings.hf_repo, str(self._model_dir))

    def _preprocess_one(self, image: ImageInput):
        pil = Image.open(io.BytesIO(image.data))
        return self._preprocess(pil)

    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        if not images:
            return []
        self._ensure_loaded()
        import torch  # local import; only needed at inference time

        batched = [self._preprocess_one(img) for img in images]
        tensor = torch.stack(batched).to(self._device)
        with torch.no_grad():
            out = self._model.encode_image(tensor)
        return [list(map(float, row)) for row in out.cpu().tolist()]

    def embed_query_image(self, image: ImageInput) -> list[float]:
        return self.embed_images([image])[0]


def _check_cuda() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError as e:
        raise ModelNotLoadedForImages(f"torch not available: {e}") from e
    raise ModelNotLoadedForImages("CUDA not available")
