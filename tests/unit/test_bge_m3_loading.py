import pytest

from vector_service.embeddings.registry import EMBEDDER_REGISTRY, get_embedder_class
from vector_service.embeddings.base import Embedder


def test_bge_m3_registered():
    assert "bge-m3" in EMBEDDER_REGISTRY
    cls = get_embedder_class("bge-m3")
    assert issubclass(cls, Embedder)


def test_unknown_embedder_raises():
    from vector_service.core.errors import EmbedderError
    with pytest.raises(EmbedderError):
        get_embedder_class("nonexistent-model")


def test_bge_m3_class_attrs():
    from vector_service.embeddings.bge_m3 import BGEM3Embedder
    assert BGEM3Embedder.dim == 1024
    assert BGEM3Embedder.model_name == "bge-m3"


@pytest.mark.slow
def test_bge_m3_real_load_cpu(tmp_path):
    """慢测试：真实加载 BGE-M3 (CPU)。需要模型已下载或网络可达。"""
    import os
    os.environ.setdefault("VS_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("VS_EMBEDDING_MODEL_DIR", str(tmp_path / "bge-m3"))
    os.environ.setdefault("VS_EMBEDDING_AUTO_DOWNLOAD", "true")

    from vector_service.core.config import get_settings
    get_settings.cache_clear()
    from vector_service.embeddings.bge_m3 import BGEM3Embedder

    emb = BGEM3Embedder()
    v = emb.embed_query("hello world")
    assert len(v) == 1024
    assert all(isinstance(x, float) for x in v)