"""GET /v1/models lists image embedders under type=image_embedder."""
from __future__ import annotations

import pytest

from vector_service.schemas.openai import Model


def test_model_type_accepts_image_embedder():
    m = Model(id="openclip-vit-l-14", type="image_embedder", dimensions=768)
    assert m.type == "image_embedder"
    assert m.dimensions == 768


def test_model_type_rejects_unknown():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Model(id="x", type="unknown_kind")  # type: ignore[arg-type]


def test_models_route_includes_image_embedder(monkeypatch):
    """The /v1/models route must surface the image embedder registry."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vector_service.api.models import router as models_router
    from vector_service.embeddings.image_base import ImageEmbedder

    class _StubImageEmbedder(ImageEmbedder):
        dim = 768
        model_name = "openclip-vit-l-14"

        def load(self):
            pass

        def embed_images(self, images):
            return [[0.0] * self.dim for _ in images]

        def embed_query_image(self, image):
            return [0.0] * self.dim

    class _StubTextEmbedder:
        model_name = "text-embed"
        dim = 4

    app = FastAPI()
    app.include_router(models_router)
    # The existing text embedder loop reads app.state.embedder; supply a stub
    # so the route doesn't AttributeError before reaching the image branch.
    app.state.embedder = _StubTextEmbedder()
    app.state.image_embedder = _StubImageEmbedder()
    client = TestClient(app)

    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    types = {entry["type"] for entry in body["data"]}
    assert "image_embedder" in types
    ids = {entry["id"]: entry for entry in body["data"]}
    assert "openclip-vit-l-14" in ids
    entry = ids["openclip-vit-l-14"]
    assert entry["type"] == "image_embedder"
    assert entry["dimensions"] == 768
