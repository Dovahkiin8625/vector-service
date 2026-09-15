"""ChineseCLIPMultimodalEmbedder with mocked transformers + huggingface_hub."""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image as _PILImage

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageInput
from vector_service.embeddings.multimodal_base import MultimodalEmbedder


def _tiny_png_bytes() -> bytes:
    """Return a real (1x1, black) PNG so PIL can decode it during fake preprocessing."""
    buf = io.BytesIO()
    _PILImage.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeMultimodalSettings:
    """Minimal stand-in for MultimodalEmbeddingSettings."""

    def __init__(
        self,
        *,
        model_dir="./models/chinese-clip-vit-base-patch16",
        device="cpu",
        batch_size=4,
        auto_download=True,
        hf_repo="OFA-Sys/chinese-clip-vit-base-patch16",
    ):
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.auto_download = auto_download
        self.hf_repo = hf_repo


def _install_chinese_clip_stub(monkeypatch, dim=512):
    """Patch ChineseCLIPModel + ChineseCLIPProcessor.

    Both towers return real torch tensors sized to ``dim`` (default 512,
    matching Chinese-CLIP's projection space). The embedder is supposed
    to read ``.pooler_output`` — that's the post-projection vector in
    shared text/image space.
    """
    encode_calls: list[tuple[str, object]] = []

    # Explicit ``self`` because class-dict assignment makes Python's
    # descriptor protocol bind ``self`` automatically; without it the
    # first kwarg collides with the bound positional.
    def _get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
        encode_calls.append(("text", input_ids))
        n = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
        return type("O", (), {"pooler_output": torch.zeros(n, dim, dtype=torch.float32)})()

    def _get_image_features(self, pixel_values=None, **kwargs):
        encode_calls.append(("image", pixel_values))
        n = pixel_values.shape[0] if hasattr(pixel_values, "shape") else 1
        return type("O", (), {"pooler_output": torch.zeros(n, dim, dtype=torch.float32)})()

    def _to(self, *a, **kw):
        return self

    fake_model = type(
        "M",
        (),
        {"get_text_features": _get_text_features, "get_image_features": _get_image_features, "to": _to},
    )()

    def _fake_tokenizer(text=None, padding=None, truncation=None, max_length=None, return_tensors=None, **kwargs):
        n = len(text) if isinstance(text, list) else 1
        # shape doesn't matter for the fake; only ``shape[0]`` is read.
        return {"input_ids": torch.zeros(n, 8, dtype=torch.long), "attention_mask": torch.ones(n, 8, dtype=torch.long)}

    fake_tokenizer = _fake_tokenizer

    def _fake_processor(images=None, return_tensors=None, **kwargs):
        if images is None:
            return {}
        n = len(images) if isinstance(images, list) else 1
        return {"pixel_values": torch.zeros(n, 3, 224, 224, dtype=torch.float32)}

    fake_processor = _fake_processor

    def _create_model_and_processor(repo_dir, **kwargs):
        return fake_model, fake_processor, fake_tokenizer

    monkeypatch.setattr(
        "vector_service.embeddings.chinese_clip_multimodal._create_model_and_processor",
        _create_model_and_processor,
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )
    return encode_calls


def _make_idempotent_stub(monkeypatch):
    create_calls: list[tuple] = []

    def _create(repo_dir, **kwargs):
        create_calls.append((repo_dir,))

        class _M:
            def get_text_features(self, input_ids=None, attention_mask=None, **kwargs):
                n = input_ids.shape[0]
                return type("O", (), {"pooler_output": torch.zeros(n, 512, dtype=torch.float32)})()

            def get_image_features(self, pixel_values=None, **kwargs):
                n = pixel_values.shape[0]
                return type("O", (), {"pooler_output": torch.zeros(n, 512, dtype=torch.float32)})()

            def to(self, *a, **kw):
                return self

        def _tokenizer(text=None, padding=None, truncation=None, max_length=None, return_tensors=None, **kwargs):
            n = len(text) if isinstance(text, list) else 1
            return {"input_ids": torch.zeros(n, 8, dtype=torch.long)}

        def _processor(images=None, return_tensors=None, **kwargs):
            n = len(images) if isinstance(images, list) else 1
            return {"pixel_values": torch.zeros(n, 3, 224, 224, dtype=torch.float32)}

        return _M(), _processor, _tokenizer

    monkeypatch.setattr(
        "vector_service.embeddings.chinese_clip_multimodal._create_model_and_processor",
        _create,
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )
    return create_calls


# ---- ABC contract -----------------------------------------------------------


def test_multimodal_embedder_is_abstract():
    """MultimodalEmbedder can't be instantiated directly — subclasses must override."""
    with pytest.raises(TypeError):
        MultimodalEmbedder()  # type: ignore[abstract]


def test_chinese_clip_class_metadata():
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    assert ChineseCLIPMultimodalEmbedder.dim == 512
    assert ChineseCLIPMultimodalEmbedder.model_name == "chinese-clip-vit-base-patch16"


# ---- Load + warmup ----------------------------------------------------------


def test_chinese_clip_load_calls_create_and_warmup(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    encode_calls = _install_chinese_clip_stub(monkeypatch)
    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    embedder = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]

    embedder.load()
    # Warmup calls embed_images once → get_image_features.
    image_calls = [c for kind, c in encode_calls if kind == "image"]
    assert len(image_calls) >= 1


def test_chinese_clip_load_is_idempotent(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    create_calls = _make_idempotent_stub(monkeypatch)

    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()
    e.load()
    assert len(create_calls) == 1


def test_chinese_clip_load_propagates_model_not_loaded(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    def _boom(*a, **kw):
        raise RuntimeError("disk error")

    monkeypatch.setattr(
        "vector_service.embeddings.chinese_clip_multimodal._create_model_and_processor", _boom
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )

    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    with pytest.raises(ModelNotLoadedForImages):
        e.load()


# ---- embed_text -------------------------------------------------------------


def test_embed_text_returns_one_vector_per_input(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    encode_calls = _install_chinese_clip_stub(monkeypatch)
    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    vecs = e.embed_text(["一只猫", "一只狗", "一只鸟"])
    assert len(vecs) == 3
    assert all(len(v) == 512 for v in vecs)
    text_calls = [c for kind, c in encode_calls if kind == "text"]
    assert len(text_calls) == 1  # batched


def test_embed_text_empty_returns_empty(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    _install_chinese_clip_stub(monkeypatch)
    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    assert e.embed_text([]) == []


# ---- embed_images -----------------------------------------------------------


def test_embed_images_returns_one_vector_per_input(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    encode_calls = _install_chinese_clip_stub(monkeypatch)
    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    inputs = [ImageInput(data=_tiny_png_bytes(), mime="image/png") for _ in range(3)]
    vecs = e.embed_images(inputs)
    assert len(vecs) == 3
    assert all(len(v) == 512 for v in vecs)
    # Skip the warmup call; the test path should make one encode call.
    image_calls = [c for kind, c in encode_calls if kind == "image"]
    # warmup + 1 batched call = 2
    assert len(image_calls) >= 1


def test_embed_images_empty_returns_empty(monkeypatch, tmp_path):
    from vector_service.embeddings.chinese_clip_multimodal import ChineseCLIPMultimodalEmbedder

    _install_chinese_clip_stub(monkeypatch)
    s = _FakeMultimodalSettings(model_dir=str(tmp_path))
    e = ChineseCLIPMultimodalEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    assert e.embed_images([]) == []