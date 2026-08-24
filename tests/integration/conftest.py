import uuid

import pytest
from fastapi.testclient import TestClient

from vector_service.main import create_app
from vector_service.core.config import get_settings


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Per-test settings: tmp data dir + tmp milvus URI, no auto-download."""
    monkeypatch.setenv("VS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VS_MILVUS_URI", str(tmp_path / f"milvus_{uuid.uuid4().hex[:8]}.db"))
    monkeypatch.setenv("VS_EMBEDDING_MODEL_DIR", str(tmp_path / "models" / "bge-m3"))
    monkeypatch.setenv("VS_EMBEDDING_AUTO_DOWNLOAD", "false")
    monkeypatch.setenv("VS_LOG_FORMAT", "console")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def fake_components(monkeypatch):
    """注入 FakeEmbedder + FakeStore，跳过模型加载。"""
    from vector_service.core import lifespan as lspan
    from vector_service.testing.fake_embedder import FakeEmbedder
    from vector_service.testing.fake_store import FakeStore

    monkeypatch.setattr(lspan, "build_embedder", lambda settings: FakeEmbedder(dim=4))
    monkeypatch.setattr(lspan, "build_store", lambda settings: FakeStore())


@pytest.fixture
def client(tmp_settings, fake_components):
    app = create_app()
    with TestClient(app) as c:
        yield c