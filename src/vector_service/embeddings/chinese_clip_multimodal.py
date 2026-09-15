"""Chinese-CLIP ViT-B/16 multimodal embedder (cross-modal text+image, 512d).

Loaded via ``transformers.ChineseCLIPModel`` + ``ChineseCLIPProcessor``.
The model's internal ``text_projection`` and ``visual_projection`` map the
768d tower outputs to a shared 512d embedding space, which we read from
``get_text_features`` / ``get_image_features`` via ``.pooler_output``
(the projection is applied in-place by the upstream implementation).

This single class backs both ``embed_text`` and ``embed_images`` so that
cosine similarity between text and image vectors is meaningful — the
basis of text-search-image and image-search-text retrieval.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings import _common
from vector_service.embeddings.image_base import ImageInput
from vector_service.embeddings.multimodal_base import MultimodalEmbedder


# Indirection so tests can monkeypatch without importing transformers at
# module load. Returns ``(model, processor, tokenizer)`` because we need
# the tokenizer separately for batched text encoding — the processor's
# ``__call__`` only handles ``images=`` (text path goes via
# ``processor.tokenizer``).
def _create_model_and_processor(repo_dir: str, **kwargs):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # lazy

    model = ChineseCLIPModel.from_pretrained(repo_dir, **kwargs)
    processor = ChineseCLIPProcessor.from_pretrained(repo_dir)
    return model, processor, processor.tokenizer


class ChineseCLIPMultimodalEmbedder(MultimodalEmbedder):
    dim = 512
    model_name = "chinese-clip-vit-base-patch16"

    def __init__(self, settings=None):
        if settings is None:
            # Lazy import to avoid pulling core.config when the embedder
            # is used in tests that pass their own settings stub.
            from vector_service.core.config import get_settings

            s = get_settings().multimodal_embedding
        else:
            s = settings
        self._settings = s
        self._device = self._resolve_device(s.device)
        self._batch_size = s.batch_size
        self._model_dir = Path(s.model_dir)
        self._model = None
        self._processor = None
        self._tokenizer = None

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
        """Release the Chinese-CLIP model + processor + tokenizer.

        Idempotent. The text and image towers share weights inside a
        single ``ChineseCLIPModel`` instance, so we only need to drop
        one reference. After this returns, the next ``embed_text`` or
        ``embed_images`` call will trigger a fresh ``_load_internal``
        via ``_ensure_loaded``.
        """
        model = self._model
        self._model = None
        self._processor = None
        self._tokenizer = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            try:
                model.__dict__.clear()
            except Exception:
                pass
        _common.release_cuda_cache()

    def _load_internal(self) -> None:
        self._ensure_model_dir()

        if not _common.dir_has_model(self._model_dir) and not self._settings.auto_download:
            raise ModelNotLoadedForImages(
                f"model dir {self._model_dir} has no Chinese-CLIP weights and auto_download is off"
            )

        try:
            model, processor, tokenizer = _create_model_and_processor(str(self._model_dir))
            self._model = model.to(self._device)
            self._processor = processor
            self._tokenizer = tokenizer
        except Exception as e:
            raise ModelNotLoadedForImages(
                f"failed to load Chinese-CLIP ViT-B/16: {e}"
            ) from e

        # Warmup: a single tiny image materializes kernels for both
        # towers' CUDA paths. Failures here are non-fatal.
        try:
            tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
            buf = io.BytesIO()
            tiny.save(buf, format="PNG")
            self.embed_images([ImageInput(data=buf.getvalue(), mime="image/png")])
        except Exception as e:
            import structlog

            structlog.get_logger(__name__).warning("chinese_clip_warmup_failed", error=str(e))

    def _ensure_model_dir(self) -> None:
        """Populate the local model dir via HF when empty.

        Mirrors ``bge_m3._ensure_model_dir``: if files already exist
        we're done; otherwise (with ``auto_download`` on) pull the
        configured ``hf_repo``. Offline / air-gapped deployments set
        ``auto_download=false`` and place weights manually.
        """
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if _common.dir_has_model(self._model_dir):
            return
        if not self._settings.auto_download:
            return
        repo = self._settings.hf_repo
        if not repo:
            raise ModelNotLoadedForImages(
                "no hf_repo configured and model dir is empty"
            )
        from huggingface_hub import snapshot_download  # lazy

        snapshot_download(
            repo_id=repo,
            local_dir=str(self._model_dir),
            local_dir_use_symlinks=False,
        )

    # ---- text tower --------------------------------------------------------

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        import torch  # local; only needed at inference time

        encoded = self._tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=52,  # chinese-clip default
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            features = self._model.get_text_features(**encoded)
        vec = features.pooler_output  # already projected to 512d
        return [list(map(float, row)) for row in vec.cpu().tolist()]

    # ---- image tower -------------------------------------------------------

    def _preprocess_one(self, image: ImageInput):
        pil = Image.open(io.BytesIO(image.data))
        return self._processor(images=[pil], return_tensors="pt")

    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        if not images:
            return []
        self._ensure_loaded()
        import torch  # local; only needed at inference time

        batched = []
        for img in images:
            out = self._preprocess_one(img)
            batched.append(out["pixel_values"])
        tensor = torch.cat(batched, dim=0).to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(pixel_values=tensor)
        vec = features.pooler_output  # already projected to 512d
        return [list(map(float, row)) for row in vec.cpu().tolist()]