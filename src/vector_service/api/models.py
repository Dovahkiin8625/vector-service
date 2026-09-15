"""Model registry endpoints.

`GET /v1/models` returns every registered backend — embedders, rerankers,
and image embedders — under a single shape. `GET /v1/models/{id}` looks
up a single backend by id. Both discriminate the family via the `type`
field on each row.

`POST /v1/models/{id}/load` and `POST /v1/models/{id}/unload` provide
hot-reload controls over the single live instance of each family that
the service keeps on ``app.state``. They live on the same router so
discovery and lifecycle share one OpenAPI tag.

Lives on its own router (tag `model`) so the OpenAPI surface groups
discovery endpoints together regardless of which subsystem (embeddings,
rerank, or image embeddings) actually serves the inference calls.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    EmbedderError,
    ImageEmbedderError,
    MultimodalEmbedderError,
    RerankerError,
)
from vector_service.embeddings.image_registry import (
    IMAGE_EMBEDDER_REGISTRY,
    get_image_embedder_class,
    list_image_embedder_names,
)
from vector_service.embeddings.multimodal_registry import (
    MULTIMODAL_EMBEDDER_REGISTRY,
    get_multimodal_embedder_class,
    list_multimodal_embedder_names,
)
from vector_service.embeddings.registry import (
    EMBEDDER_REGISTRY,
    get_embedder_class,
    list_embedder_names,
)
from vector_service.core.model_lifecycle import (
    ConcurrentModelOperation,
    DifferentModelLoaded,
    ModelSlot,
)
from vector_service.rerankers.registry import (
    get_reranker_class,
    list_reranker_names,
)
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.openai import (
    Model,
    ModelList,
    ModelLoadResponse,
    ModelUnloadResponse,
)

router = APIRouter(prefix="/v1", tags=["model"])

# Type alias used internally for ``_resolve_family`` return values.
_FamilyLiteral = str  # one of: embedder | image_embedder | multimodal_embedder | reranker


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
    # Defensive ``getattr`` everywhere — the four family slots are
    # populated by lifespan in production, but test fixtures that build
    # an app without lifespan never set them, and ``request.app.state``
    # raises AttributeError on missing keys.
    embedder = getattr(request.app.state, "embedder", None)
    models: list[Model] = []
    for name in list_embedder_names():
        loaded = bool(embedder and embedder.model_name == name)
        try:
            dim = embedder.dim if loaded else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="embedder", dimensions=dim, loaded=loaded))
    for name in list_reranker_names():
        # Rerankers don't expose a dimension. ``loaded`` is the only signal
        # the dashboard (or any other consumer) has to tell apart a loaded
        # reranker from a registered-but-unloaded one.
        reranker = getattr(request.app.state, "reranker", None)
        loaded = bool(reranker and reranker.model_name == name)
        models.append(Model(id=name, type="reranker", dimensions=None, loaded=loaded))
    image_embedder = getattr(request.app.state, "image_embedder", None)
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
        loaded = effective is not None
        try:
            dim = effective.dim if loaded else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="image_embedder", dimensions=dim, loaded=loaded))
    multimodal_embedder = getattr(request.app.state, "multimodal_embedder", None)
    for name in list_multimodal_embedder_names():
        effective = (
            multimodal_embedder
            if (multimodal_embedder and multimodal_embedder.model_name == name)
            else None
        )
        loaded = effective is not None
        try:
            dim = effective.dim if loaded else None
        except Exception:
            dim = None
        models.append(Model(id=name, type="multimodal_embedder", dimensions=dim, loaded=loaded))
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
        loaded = bool(embedder and embedder.model_name == model_id)
        dim = embedder.dim if loaded else None
        return Model(id=model_id, type="embedder", dimensions=dim, loaded=loaded)
    if model_id in list_reranker_names():
        # Rerankers don't expose a dimension; ``loaded`` is the only
        # signal that the queried id is currently held on app.state.
        # ``getattr`` so test harnesses that skip lifespan (and therefore
        # never set ``app.state.reranker``) don't crash.
        reranker = getattr(request.app.state, "reranker", None)
        loaded = bool(reranker and reranker.model_name == model_id)
        return Model(id=model_id, type="reranker", dimensions=None, loaded=loaded)
    if model_id in IMAGE_EMBEDDER_REGISTRY:
        image_embedder = request.app.state.image_embedder
        # See note above in ``list_models``: only report ``dimensions``
        # when the queried id matches the live embedder on app.state.
        # Otherwise the model is registered but not loaded, and the
        # dimension is genuinely unknown to this process.
        loaded = bool(image_embedder and image_embedder.model_name == model_id)
        dim = image_embedder.dim if loaded else None
        return Model(id=model_id, type="image_embedder", dimensions=dim, loaded=loaded)
    if model_id in MULTIMODAL_EMBEDDER_REGISTRY:
        multimodal_embedder = getattr(request.app.state, "multimodal_embedder", None)
        loaded = bool(multimodal_embedder and multimodal_embedder.model_name == model_id)
        dim = multimodal_embedder.dim if loaded else None
        return Model(id=model_id, type="multimodal_embedder", dimensions=dim, loaded=loaded)
    registered = (
        sorted(EMBEDDER_REGISTRY)
        + sorted(list_reranker_names())
        + sorted(IMAGE_EMBEDDER_REGISTRY)
        + sorted(MULTIMODAL_EMBEDDER_REGISTRY)
    )
    raise HTTPException(status_code=404, detail={"error": {
        "code": "model_not_found",
        "message": f"unknown model {model_id!r}; registered: {registered}",
        "model": model_id,
    }})


# ---- hot load / unload ---------------------------------------------------
#
# The single live instance per family lives on ``app.state.<family>``
# after the lifespan eager-load. To support hot-swapping we mount a
# parallel :class:`ModelSlot` on ``app.state._slot_<family>`` (see
# ``core.model_lifecycle.attach_default_slots``). Each route resolves
# ``model_id`` to a (class, family, slot) triple, dispatches the slot
# operation to a thread executor so the event loop never blocks on the
# family lock, and maps the slot exceptions onto HTTP error envelopes.

# Lookup table: family name → (registry_dict, slot attr on app.state,
# settings block attribute, class lookup callable, app.state mirror
# attr). ``settings_attr=None`` means the family expects the full
# ``Settings`` instance (rather than a nested block) — that's the
# case for the text embedder, whose constructor signature is
# ``Embedder(settings: Settings)``. All other families take a nested
# block (``image_embedding``, ``multimodal_embedding``, ``reranker``).
# Order is the priority for ``_resolve_family`` — embedder beats
# reranker when the same id is registered in both. Each entry is
# ``(slot_attr, settings_attr, class_lookup, state_attr)``: the
# slot attribute on ``app.state``, the nested settings block (or
# ``None`` to pass the whole ``Settings``), the registry-lookup
# callable, and the legacy ``app.state.<family>`` mirror name.
_FAMILY_TABLE: dict[str, tuple[str, str | None, Callable[[str], Any], str]] = {
    "embedder": (
        "_slot_embedder",
        None,
        get_embedder_class,
        "embedder",
    ),
    "image_embedder": (
        "_slot_image",
        "image_embedding",
        get_image_embedder_class,
        "image_embedder",
    ),
    "multimodal_embedder": (
        "_slot_multimodal",
        "multimodal_embedding",
        get_multimodal_embedder_class,
        "multimodal_embedder",
    ),
    "reranker": (
        "_slot_reranker",
        "reranker",
        get_reranker_class,
        "reranker",
    ),
}

# Registry lookup is centralised here so we don't import four different
# registries inline. We pull the registry dict from the ``embeddings``
# / ``rerankers`` subpackages rather than from the module-level names
# in this file (those imports are kept for backwards compatibility).
def _family_registries() -> dict[str, dict[str, type]]:
    return {
        "embedder": EMBEDDER_REGISTRY,
        "image_embedder": IMAGE_EMBEDDER_REGISTRY,
        "multimodal_embedder": MULTIMODAL_EMBEDDER_REGISTRY,
        "reranker": _reranker_registry_dict(),
    }


def _reranker_registry_dict() -> dict[str, type]:
    # Imported lazily to keep this module import-light; the reranker
    # registry lives in a sub-module that has a side-effect import.
    from vector_service.rerankers.registry import RERANKER_REGISTRY
    return RERANKER_REGISTRY


def _resolve_family(request: Request, model_id: str) -> tuple[str, ModelSlot, type, Callable[[], Any], Any, str]:
    """Locate ``model_id`` in the four registries and return its family slot.

    Returns a 6-tuple ``(family, slot, cls, factory, settings_block, state_attr)``.
    ``cls`` is the registered class — the slot uses its ``model_name``
    class attribute to decide idempotent re-load vs. construction.
    ``state_attr`` is the ``app.state`` attribute name that mirrors the
    slot for the inference routes (``app.state.embedder`` etc.) — the
    load/unload handlers keep the two in sync.

    Order of resolution: embedder → image_embedder → multimodal_embedder
    → reranker. Documented in the load/unload route docstrings. The
    embedder wins over reranker on collision because most callers who
    ask for a ``bge-*`` id expect an embedder.

    Raises ``HTTPException`` 404 if the id is not registered anywhere.
    """
    registries = _family_registries()
    for family in ("embedder", "image_embedder", "multimodal_embedder", "reranker"):
        if model_id in registries[family]:
            slot_attr, settings_attr, class_lookup, state_attr = _FAMILY_TABLE[family]
            cls = class_lookup(model_id)
            slot = getattr(request.app.state, slot_attr, None)
            if slot is None:
                # Should be impossible in production: lifespan attaches
                # the slots. Surface a 503 if a custom test harness
                # forgets to call ``attach_default_slots``.
                raise HTTPException(503, detail={"error": {
                    "code": "model_slot_unavailable",
                    "message": f"{family} slot is not attached on app.state",
                }})
            settings_block = (
                request.app.state.settings
                if settings_attr is None
                else getattr(request.app.state.settings, settings_attr)
            )
            # Closure picks up the resolved class + settings block; the
            # route layer never needs to know the concrete type.
            def _factory(_cls=cls, _settings=settings_block):
                return _cls(settings=_settings)
            return family, slot, cls, _factory, settings_block, state_attr

    registered: list[str] = []
    for family in ("embedder", "image_embedder", "multimodal_embedder", "reranker"):
        registered.extend(sorted(registries[family]))
    raise HTTPException(status_code=404, detail={"error": {
        "code": "model_not_found",
        "message": f"unknown model {model_id!r}; registered: {sorted(set(registered))}",
        "model": model_id,
    }})


def _http_error(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    """Build an HTTPException with the canonical error envelope shape."""
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return HTTPException(status, detail={"error": detail})


@router.post(
    "/models/{model_id}/load",
    response_model=ModelLoadResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown model id."},
        409: {"model": ErrorEnvelope, "description": "Model is busy or a different id is loaded."},
        503: {"model": ErrorEnvelope, "description": "Backend failed to load."},
    },
    summary="Hot-load a model",
    description=(
        "Construct and load the registered backend for ``model_id``, "
        "replacing the current instance of that family on "
        "``app.state``. Idempotent for the same id; a *different* id "
        "in the same family returns 409 ``conflict_loaded`` (call "
        "``/unload`` first). Concurrent load/unload on the same "
        "family returns 409 ``model_busy``."
    ),
)
async def load_model(model_id: str, request: Request):
    family, slot, cls, factory, _settings, state_attr = _resolve_family(request, model_id)
    loop = asyncio.get_running_loop()
    try:
        instance = await loop.run_in_executor(None, slot.load, cls, factory)
    except ConcurrentModelOperation as exc:
        raise _http_error(409, "model_busy", str(exc), model=model_id, family=family)
    except DifferentModelLoaded as exc:
        raise _http_error(409, "conflict_loaded", str(exc), model=model_id, family=family)
    except (EmbedderError, ImageEmbedderError, MultimodalEmbedderError, RerankerError) as exc:
        raise _http_error(
            503, "model_load_failed",
            str(exc) or f"failed to load {model_id}",
            model=model_id, family=family,
            exception_type=type(exc).__name__,
        )
    # Mirror the slot into ``app.state.<family>`` so the inference routes
    # see the freshly loaded instance immediately. Without this the
    # routes would keep reading the previous (or stale) instance.
    setattr(request.app.state, state_attr, instance)
    return ModelLoadResponse(
        id=instance.model_name,
        type=family,  # type: ignore[arg-type]
        status="loaded",
        dimensions=getattr(instance, "dim", None),
    )


@router.post(
    "/models/{model_id}/unload",
    response_model=ModelUnloadResponse,
    responses={
        404: {"model": ErrorEnvelope, "description": "Unknown model id."},
        409: {"model": ErrorEnvelope, "description": "No model loaded or slot busy."},
    },
    summary="Hot-unload a model",
    description=(
        "Release the instance currently held on ``app.state`` for "
        "``model_id``'s family. The slot becomes empty and subsequent "
        "inference calls for that family return 503 until a new "
        "instance is loaded. Returns 409 ``not_loaded`` if the slot "
        "is already empty; 409 ``model_busy`` if another "
        "load/unload is in flight."
    ),
)
async def unload_model(model_id: str, request: Request):
    family, slot, _cls, _factory, _settings, state_attr = _resolve_family(request, model_id)
    loop = asyncio.get_running_loop()
    try:
        released = await loop.run_in_executor(None, slot.unload)
    except ConcurrentModelOperation as exc:
        raise _http_error(409, "model_busy", str(exc), model=model_id, family=family)
    if not released:
        raise _http_error(
            409, "not_loaded",
            f"no {family} instance is currently loaded",
            model=model_id, family=family,
        )
    # Drop the inference-side mirror so subsequent calls return
    # 503 (model not loaded) instead of silently re-loading via
    # ``_ensure_loaded``.
    setattr(request.app.state, state_attr, None)
    return ModelUnloadResponse(
        id=model_id,
        type=family,  # type: ignore[arg-type]
        status="unloaded",
    )
