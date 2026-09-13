"""Model registry endpoints.

`GET /v1/models` returns every registered backend — embedders, rerankers,
and image embedders — under a single shape. `GET /v1/models/{id}` looks
up a single backend by id. Both discriminate the family via the `type`
field on each row.

Lives on its own router (tag `model`) so the OpenAPI surface groups
discovery endpoints together regardless of which subsystem (embeddings,
rerank, or image embeddings) actually serves the inference calls.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from vector_service.embeddings.image_registry import (
    IMAGE_EMBEDDER_REGISTRY,
    list_image_embedder_names,
)
from vector_service.embeddings.registry import (
    EMBEDDER_REGISTRY,
    list_embedder_names,
)
from vector_service.rerankers.registry import list_reranker_names
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.openai import Model, ModelList

router = APIRouter(prefix="/v1", tags=["model"])


@router.get(
    "/models",
    response_model=ModelList,
    summary="List available models",
    description=(
        "Return every registered backend — embedders, rerankers, and "
        "image embedders — under a single shape. Use `type` to "
        "discriminate them: `embedder` and `image_embedder` rows carry "
        "a `dimensions` value once loaded, `reranker` rows always have "
        "`dimensions=null`."
    ),
)
def list_models(request: Request):
    embedder = request.app.state.embedder
    models: list[Model] = []
    for name in list_embedder_names():
        try:
            dim = embedder.dim if (embedder and name == embedder.model_name) else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="embedder", dimensions=dim))
    for name in list_reranker_names():
        models.append(Model(id=name, type="reranker", dimensions=None))
    image_embedder = request.app.state.image_embedder
    for name in list_image_embedder_names():
        # Constraint: only the embedder that is actually loaded on
        # ``app.state.image_embedder`` can report a real dimension.
        # Querying a registered-but-unloaded model id must surface
        # ``dimensions=null`` so callers can tell apart "available"
        # from "currently loaded".
        effective = (
            image_embedder
            if (image_embedder and image_embedder.model_name == name)
            else None
        )
        try:
            dim = effective.dim if effective else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="image_embedder", dimensions=dim))
    return ModelList(data=models)


@router.get(
    "/models/{model_id}",
    response_model=Model,
    responses={404: {"model": ErrorEnvelope, "description": "Unknown model id."}},
    summary="Get a single model",
    description=(
        "Look up a registered embedder, reranker, or image embedder by "
        "id. `type` distinguishes the three families."
    ),
)
def get_model(model_id: str, request: Request):
    if model_id in EMBEDDER_REGISTRY:
        embedder = request.app.state.embedder
        dim = embedder.dim if (embedder and embedder.model_name == model_id) else None
        return Model(id=model_id, type="embedder", dimensions=dim)
    if model_id in list_reranker_names():
        return Model(id=model_id, type="reranker", dimensions=None)
    if model_id in IMAGE_EMBEDDER_REGISTRY:
        image_embedder = request.app.state.image_embedder
        # See note above in ``list_models``: only report ``dimensions``
        # when the queried id matches the live embedder on app.state.
        # Otherwise the model is registered but not loaded, and the
        # dimension is genuinely unknown to this process.
        dim = (
            image_embedder.dim
            if (image_embedder and image_embedder.model_name == model_id)
            else None
        )
        return Model(id=model_id, type="image_embedder", dimensions=dim)
    registered = (
        sorted(EMBEDDER_REGISTRY)
        + sorted(list_reranker_names())
        + sorted(IMAGE_EMBEDDER_REGISTRY)
    )
    raise HTTPException(status_code=404, detail={"error": {
        "code": "model_not_found",
        "message": f"unknown model {model_id!r}; registered: {registered}",
        "model": model_id,
    }})
