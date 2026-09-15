"""OpenCLIP ViT-L/14 image embedder (openai pretrained weights, 768d)."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

from vector_service.core.config import ImageEmbeddingSettings
from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings import _common
from vector_service.embeddings.image_base import ImageEmbedder, ImageInput


# Indirection so tests can monkeypatch without us importing open_clip at
# module load (we want a clean ImportError path if open_clip_torch isn't
# installed in production).
def _create_model_and_transforms(name: str, pretrained: str, **kwargs):
    import open_clip  # lazy

    return open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)


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
        return _common.resolve_device(requested)

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._load_internal()

    def load(self) -> None:
        if self._model is not None:
            return
        self._load_internal()

    def unload(self) -> None:
        """Release the OpenCLIP model + preprocess transform.

        Idempotent. We null out the model and preprocess references so
        the next inference call triggers a fresh ``_load_internal``
        via ``_ensure_loaded`` (mirrors the text-embedder contract).
        """
        model = self._model
        self._model = None
        self._preprocess = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            # Drop the module's __dict__ so wrapped tensors become
            # unreachable and the CUDA caching allocator can reclaim
            # them. We swallow any errors: unload must never raise.
            try:
                model.__dict__.clear()
            except Exception:
                pass
        _common.release_cuda_cache()

    def _load_internal(self) -> None:
        # Ensure the local model_dir exists so open_clip can use it as a
        # download cache. open_clip's own downloader (called inside
        # create_model_and_transforms) handles fetching weights; we don't
        # talk to HuggingFace directly because OpenCLIP's manifest is
        # served from open_clip's CDN, not a normal HF repo.
        self._model_dir.mkdir(parents=True, exist_ok=True)
        # Point open_clip at our local cache so subsequent loads skip the
        # network call.
        os.environ.setdefault("OPEN_CLIP_DOWNLOAD_PATH", str(self._model_dir))

        if not _common.dir_has_model(self._model_dir) and not self._settings.auto_download:
            raise ModelNotLoadedForImages(
                f"model dir {self._model_dir} has no OpenCLIP weights and auto_download is off"
            )

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
            tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
            buf = io.BytesIO()
            tiny.save(buf, format="PNG")
            self.embed_images([ImageInput(data=buf.getvalue(), mime="image/png")])
        except Exception as e:  # warmup failures are non-fatal
            import structlog

            structlog.get_logger(__name__).warning("openclip_warmup_failed", error=str(e))

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
