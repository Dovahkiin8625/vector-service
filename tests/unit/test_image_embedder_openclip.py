"""OpenCLIPVitL14ImageEmbedder with mocked open_clip."""
from __future__ import annotations

import io

import pytest
import torch
from PIL import Image as _PILImage

from vector_service.core.errors import ModelNotLoadedForImages
from vector_service.embeddings.image_base import ImageInput


def _tiny_png_bytes() -> bytes:
    """Return a real (1x1, black) PNG so PIL can decode it during fake preprocessing."""
    buf = io.BytesIO()
    _PILImage.new("RGB", (1, 1), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeImageSettings:
    """Minimal stand-in for ImageEmbeddingSettings — only the attrs the embedder reads."""

    def __init__(
        self,
        *,
        model_dir="./models/openclip-vit-l-14",
        device="cpu",
        batch_size=4,
        auto_download=True,
        download_source="huggingface",
        hf_repo="",
    ):
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.auto_download = auto_download
        self.download_source = download_source
        self.hf_repo = hf_repo


def _install_open_clip_stub(monkeypatch, encode_return=None):
    """Patch open_clip + huggingface_hub so no real model loads.

    The fakes return real torch tensors so the implementation's
    ``torch.stack`` and ``out.cpu().tolist()`` calls work unchanged.
    """
    encode_return = encode_return if encode_return is not None else [[0.0] * 768]

    encode_calls: list[tuple] = []

    def _encode_image(tensor):
        encode_calls.append(tensor)
        # Mimic open_clip: one row per item in the batch.
        n = tensor.shape[0] if hasattr(tensor, "shape") else len(tensor)
        data = (
            encode_return * n
            if isinstance(encode_return, list) and len(encode_return) == n
            else encode_return[:n]
        )
        return torch.tensor(data, dtype=torch.float32)

    fake_model = type("M", (), {"encode_image": lambda self, t: _encode_image(t)})()
    # open_clip's preprocess returns a (C, H, W) tensor.
    fake_preprocess = lambda pil: torch.zeros(3, 224, 224, dtype=torch.float32)  # noqa: E731
    fake_tokenize = lambda texts: None  # noqa: E731

    def _create_model_and_transforms(name, pretrained, **kwargs):
        assert name == "ViT-L-14"
        assert pretrained == "openai"
        return fake_model, fake_preprocess, fake_tokenize

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms",
        _create_model_and_transforms,
    )
    # Avoid hitting HuggingFace: pretend the model dir already has weights.
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )
    return encode_calls


def _make_idempotent_stub(monkeypatch):
    """Variant: idempotency test only cares that _create runs once."""
    create_calls: list[tuple] = []

    def _create(name, pretrained, **kwargs):
        create_calls.append((name, pretrained))

        class _M:
            def encode_image(self, t):
                return torch.zeros(t.shape[0], 768, dtype=torch.float32)

        return _M(), lambda pil: torch.zeros(3, 224, 224, dtype=torch.float32), lambda texts: None

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms",
        _create,
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )
    return create_calls


def test_openclip_class_metadata():
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    assert OpenCLIPVitL14ImageEmbedder.dim == 768
    assert OpenCLIPVitL14ImageEmbedder.model_name == "openclip-vit-l-14"


def test_openclip_load_calls_create_model_and_warmup(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    encode_calls = _install_open_clip_stub(monkeypatch)
    s = _FakeImageSettings(model_dir=str(tmp_path))
    embedder = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]

    embedder.load()
    # Warmup issues exactly one encode_image call.
    assert len(encode_calls) >= 1


def test_openclip_load_is_idempotent(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    create_calls = _make_idempotent_stub(monkeypatch)

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()
    e.load()
    assert len(create_calls) == 1


def test_openclip_load_propagates_model_not_loaded_for_images(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    def _boom(*a, **kw):
        raise RuntimeError("disk error")

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _boom
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    with pytest.raises(ModelNotLoadedForImages):
        e.load()


def test_embed_images_returns_one_vector_per_input(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    encode_calls: list = []

    def _create(name, pretrained, **kwargs):
        class _M:
            def encode_image(self, t):
                encode_calls.append(t)
                n = t.shape[0]
                return torch.tensor(
                    [[0.1 * i] * 768 for i in range(n)], dtype=torch.float32
                )

        return _M(), lambda pil: torch.zeros(3, 224, 224, dtype=torch.float32), lambda texts: None

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    inputs = [ImageInput(data=_tiny_png_bytes(), mime="image/png") for _ in range(3)]
    vecs = e.embed_images(inputs)
    assert len(vecs) == 3
    assert all(len(v) == 768 for v in vecs)


def test_embed_query_image_equals_embed_images_of_one(monkeypatch, tmp_path):
    from vector_service.embeddings.openclip_vit_l14 import OpenCLIPVitL14ImageEmbedder

    def _create(name, pretrained, **kwargs):
        class _M:
            def encode_image(self, t):
                n = t.shape[0]
                return torch.tensor(
                    [[float(i)] * 768 for i in range(n)], dtype=torch.float32
                )

        return _M(), lambda pil: torch.zeros(3, 224, 224, dtype=torch.float32), lambda texts: None

    monkeypatch.setattr(
        "vector_service.embeddings.openclip_vit_l14._create_model_and_transforms", _create
    )
    monkeypatch.setattr(
        "vector_service.embeddings._common.dir_has_model", lambda p: True
    )

    s = _FakeImageSettings(model_dir=str(tmp_path))
    e = OpenCLIPVitL14ImageEmbedder(settings=s)  # type: ignore[arg-type]
    e.load()

    img = ImageInput(data=_tiny_png_bytes(), mime="image/png")
    q = e.embed_query_image(img)
    assert q == e.embed_images([img])[0]
