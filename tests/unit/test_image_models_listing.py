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


def test_get_model_image_dimensions_null_when_id_does_not_match_loaded(monkeypatch):
    """``GET /v1/models/{id}`` for an image embedder must report
    ``dimensions=null`` when the queried id differs from the live
    embedder on ``app.state.image_embedder``.

    The service only keeps one image embedder loaded at a time, so a
    registered-but-not-currently-loaded backend is genuinely "dim
    unknown" to this process — clients should see that as null, not as
    the dimension of whichever backend happens to be loaded.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vector_service.api.models import router as models_router
    from vector_service.embeddings.image_base import ImageEmbedder

    class _LoadedImageEmbedder(ImageEmbedder):
        dim = 768
        model_name = "openclip-vit-l-14"  # the backend that IS loaded

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
    app.state.embedder = _StubTextEmbedder()
    app.state.image_embedder = _LoadedImageEmbedder()
    client = TestClient(app)

    # Query the id that IS loaded — dimensions reported.
    r = client.get("/v1/models/openclip-vit-l-14")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "image_embedder"
    assert body["dimensions"] == 768

    # Stub a second registered-but-not-loaded id by monkeypatching the
    # image-embedder registry to include an extra name. The live embedder
    # still reports ``model_name = "openclip-vit-l-14"``, so the new id
    # must surface ``dimensions=null``.
    from vector_service.api import models as models_api
    from vector_service.embeddings import image_registry

    class _OtherImageEmbedder(ImageEmbedder):
        dim = 1024  # would be misleading if leaked
        model_name = "other-image-embedder"

        def load(self):
            pass

        def embed_images(self, images):
            return [[0.0] * self.dim for _ in images]

        def embed_query_image(self, image):
            return [0.0] * self.dim

    image_registry.IMAGE_EMBEDDER_REGISTRY["other-image-embedder"] = _OtherImageEmbedder
    try:
        r2 = client.get("/v1/models/other-image-embedder")
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["type"] == "image_embedder"
        # CRITICAL: must NOT leak the loaded embedder's dimension.
        assert body2["dimensions"] is None
    finally:
        image_registry.IMAGE_EMBEDDER_REGISTRY.pop("other-image-embedder", None)
