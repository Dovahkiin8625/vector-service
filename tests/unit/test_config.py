import pytest
from pydantic import ValidationError

from vector_service.core.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"
    assert s.port == 8080
    assert s.embedding_backend == "bge-m3"
    assert s.embedding_device == "auto"
    assert s.vector_store_backend == "milvus_lite"
    assert s.embedding_batch_size == 32
    assert s.embedding_max_length == 512


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("VS_PORT", "9090")
    monkeypatch.setenv("VS_EMBEDDING_DEVICE", "cuda")
    # Rebuild settings instance
    from vector_service.core import config as cfg
    cfg.get_settings.cache_clear()
    s = cfg.get_settings()
    assert s.port == 9090
    assert s.embedding_device == "cuda"


def test_invalid_device_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_device="tpu")


def test_paths_are_path_objects():
    s = Settings(_env_file=None)
    from pathlib import Path
    assert isinstance(s.embedding_model_dir, Path)
    assert isinstance(s.data_dir, Path)


def test_get_settings_singleton():
    from vector_service.core import config as cfg
    cfg.get_settings.cache_clear()
    a = cfg.get_settings()
    b = cfg.get_settings()
    assert a is b
