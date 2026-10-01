"""ImageEmbeddingSettings env-var parsing + nested Settings wiring."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.core.config import ImageEmbeddingSettings, Settings


def test_image_embedding_settings_defaults():
    s = ImageEmbeddingSettings()
    assert s.backend == "openclip-vit-l-14"
    assert s.device == "auto"
    assert s.batch_size == 16
    assert s.max_images_per_request == 64
    assert s.max_image_bytes == 10 * 1024 * 1024
    assert "image/jpeg" in s.allowed_mime
    assert "image/png" in s.allowed_mime
    assert "image/webp" in s.allowed_mime


def test_image_embedding_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BACKEND", "openclip-vit-l-14")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__DEVICE", "cpu")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BATCH_SIZE", "8")
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES", "5242880")
    s = ImageEmbeddingSettings()
    assert s.device == "cpu"
    assert s.batch_size == 8
    assert s.max_image_bytes == 5_242_880


def test_image_embedding_settings_batch_size_bounds():
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(batch_size=0)
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(batch_size=10_000)


def test_image_embedding_settings_max_image_bytes_bounds():
    with pytest.raises(ValidationError):
        ImageEmbeddingSettings(max_image_bytes=10)


def test_settings_exposes_image_embedding():
    s = Settings()
    assert hasattr(s, "image_embedding")
    assert isinstance(s.image_embedding, ImageEmbeddingSettings)


def test_settings_env_nested_image_embedding(monkeypatch):
    monkeypatch.setenv("VS_IMAGE_EMBEDDING__BATCH_SIZE", "4")
    s = Settings()
    assert s.image_embedding.batch_size == 4
