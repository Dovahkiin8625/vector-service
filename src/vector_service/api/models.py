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

from fastapi import APIRouter, HTTPException, Request, Response

from vector_service.core.logging import get_logger
from vector_service.core.metrics import MODEL_LOADED
from vector_service.core.model_info import describe_instance
from vector_service.core.model_lifecycle import (
    ConcurrentModelOperation,
    DifferentModelLoaded,
    ModelSlot,
)
from vector_service.core.threadpools import run_in_model
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
from vector_service.rerankers.registry import (
    get_reranker_class,
    list_reranker_names,
)
from vector_service.schemas.errors import ErrorEnvelope
from vector_service.schemas.openai import (
    Model,
    ModelInfo,
    ModelList,
    ModelLoadResponse,
    ModelUnloadResponse,
)

router = APIRouter(prefix="/v1", tags=["model"])

log = get_logger(__name__)

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
    # Lifespan always initialises the four family mirrors (empty slot →
    # None), so read them straight off app.state.
    app_state = request.app.state
    embedder = app_state.embedder
    embedder_slot = _slot_for(app_state, "embedder")
    models: list[Model] = []
    for name in list_embedder_names():
        loaded = bool(embedder and embedder.model_name == name)
        try:
            dim = embedder.dim if loaded else None
        except Exception:
            dim = None
        load_status, load_error = _row_load_state(embedder_slot, name)
        models.append(Model(
            id=name, type="embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(embedder if loaded else None, embedder_slot),
        ))
    for name in list_reranker_names():
        # Rerankers don't expose a dimension. ``loaded`` is the only signal
        # the dashboard (or any other consumer) has to tell apart a loaded
        # reranker from a registered-but-unloaded one.
        reranker = app_state.reranker
        reranker_slot = _slot_for(app_state, "reranker")
        loaded = bool(reranker and reranker.model_name == name)
        effective_reranker = reranker if loaded else None
        load_status, load_error = _row_load_state(reranker_slot, name)
        models.append(Model(
            id=name, type="reranker", dimensions=None, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(effective_reranker, reranker_slot),
        ))
    image_embedder = app_state.image_embedder
    image_slot = _slot_for(app_state, "image_embedder")
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
        load_status, load_error = _row_load_state(image_slot, name)
        models.append(Model(
            id=name, type="image_embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(effective, image_slot),
        ))
    multimodal_embedder = app_state.multimodal_embedder
    multimodal_slot = _slot_for(app_state, "multimodal_embedder")
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
        load_status, load_error = _row_load_state(multimodal_slot, name)
        models.append(Model(
            id=name, type="multimodal_embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(effective, multimodal_slot),
        ))
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
    app_state = request.app.state
    if model_id in EMBEDDER_REGISTRY:
        embedder = app_state.embedder
        loaded = bool(embedder and embedder.model_name == model_id)
        dim = embedder.dim if loaded else None
        embedder_slot = _slot_for(app_state, "embedder")
        load_status, load_error = _row_load_state(embedder_slot, model_id)
        return Model(
            id=model_id, type="embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(embedder if loaded else None, embedder_slot),
        )
    if model_id in list_reranker_names():
        # Rerankers don't expose a dimension; ``loaded`` is the only
        # signal that the queried id is currently held on app.state.
        reranker = app_state.reranker
        reranker_slot = _slot_for(app_state, "reranker")
        loaded = bool(reranker and reranker.model_name == model_id)
        load_status, load_error = _row_load_state(reranker_slot, model_id)
        return Model(
            id=model_id, type="reranker", dimensions=None, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(reranker if loaded else None, reranker_slot),
        )
    if model_id in IMAGE_EMBEDDER_REGISTRY:
        image_embedder = app_state.image_embedder
        # See note above in ``list_models``: only report ``dimensions``
        # when the queried id matches the live embedder on app.state.
        # Otherwise the model is registered but not loaded, and the
        # dimension is genuinely unknown to this process.
        loaded = bool(image_embedder and image_embedder.model_name == model_id)
        dim = image_embedder.dim if loaded else None
        image_slot = _slot_for(app_state, "image_embedder")
        load_status, load_error = _row_load_state(image_slot, model_id)
        return Model(
            id=model_id, type="image_embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(image_embedder if loaded else None, image_slot),
        )
    if model_id in MULTIMODAL_EMBEDDER_REGISTRY:
        multimodal_embedder = app_state.multimodal_embedder
        loaded = bool(multimodal_embedder and multimodal_embedder.model_name == model_id)
        dim = multimodal_embedder.dim if loaded else None
        multimodal_slot = _slot_for(app_state, "multimodal_embedder")
        load_status, load_error = _row_load_state(multimodal_slot, model_id)
        return Model(
            id=model_id, type="multimodal_embedder", dimensions=dim, loaded=loaded,
            load_status=load_status, load_error=load_error,
            model_info=_model_info(
                multimodal_embedder if loaded else None, multimodal_slot
            ),
        )
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
# ``model_id`` to a (class, family, slot) triple and maps the slot
# exceptions onto HTTP error envelopes. Load is asynchronous: the route
# runs only the slot's non-blocking ``begin_load`` phase, then schedules
# the slow ``finish_load`` (factory + ``load()``) on a thread executor
# and returns HTTP 202; clients poll GET /v1/models for ``load_status``.
# Unload stays synchronous — it only releases local resources.

# Lookup table: family name → (slot attr on app.state, settings block
# attribute, class lookup callable, app.state mirror attr).
# ``settings_attr=None`` means the family expects the full
# ``Settings`` instance (rather than a nested block) — that's the
# case for the text embedder, whose constructor signature is
# ``Embedder(settings: Settings)``. All other families take a nested
# block (``image_embedding``, ``multimodal_embedding``, ``reranker``).
# Order is the priority for ``_resolve_family`` — embedder beats
# reranker when the same id is registered in both. Each entry is
# ``(slot_attr, settings_attr, class_lookup, state_attr)``: the
# slot attribute on ``app.state``, the nested settings block (or
# ``None`` to pass the whole ``Settings``), the registry-lookup
# callable, and the ``app.state.<family>`` mirror that inference
# routes read for the live instance.
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
# registries inline.
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


def _slot_for(app_state: Any, family: str) -> ModelSlot | None:
    """Return the family's :class:`ModelSlot`, or ``None`` for test apps
    that mount the router without ``attach_default_slots``."""
    slot_attr = _FAMILY_TABLE[family][0]
    return getattr(app_state, slot_attr, None)


def _model_info(instance: Any, slot: ModelSlot | None) -> ModelInfo | None:
    """Build the runtime ``model_info`` block for a loaded row.

    ``instance`` is the live backend for THIS registered id (or ``None``
    when the row is not the family's currently-loaded id). Static
    resource details come from :func:`describe_instance` (cached on the
    instance); the slot contributes the last load duration. Returns
    ``None`` for rows with no live instance; never raises — a failure to
    introspect must not take ``GET /v1/models`` down.
    """
    if instance is None:
        return None
    try:
        details = describe_instance(instance)
    except Exception:  # noqa: BLE001 — introspection is best-effort
        details = None
    details = details or {}
    duration = getattr(slot, "load_duration", None) if slot is not None else None
    if not details and duration is None:
        return None
    return ModelInfo(
        device=details.get("device"),
        dtype=details.get("dtype"),
        param_count=details.get("param_count"),
        memory_bytes=details.get("memory_bytes"),
        load_duration_seconds=duration,
    )


def _row_load_state(slot: ModelSlot | None, name: str) -> tuple[str, str | None]:
    """Map a slot's async state onto one registered model id.

    ``loaded`` is derived separately (from the live ``app.state.<family>``
    mirror). Here we only resolve the finer ``load_status``: only the id
    the slot is currently loading (or last failed to load) reports
    ``loading`` / ``failed``; every other registered id of the same
    family stays ``unloaded``.
    """
    if slot is None:
        return "unloaded", None
    if slot.loaded_id == name:
        return "loaded", None
    if slot.loading_id == name and slot.load_state in ("loading", "failed"):
        return slot.load_state, slot.load_error
    return "unloaded", None


def _http_error(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    """Build an HTTPException with the canonical error envelope shape."""
    detail: dict[str, Any] = {"code": code, "message": message}
    detail.update(extra)
    return HTTPException(status, detail={"error": detail})


@router.post(
    "/models/{model_id}/load",
    response_model=ModelLoadResponse,
    status_code=202,
    responses={
        200: {"model": ModelLoadResponse, "description": "Idempotent re-load: the id was already held."},
        202: {"model": ModelLoadResponse, "description": "Load accepted; running in the background."},
        404: {"model": ErrorEnvelope, "description": "Unknown model id."},
        409: {"model": ErrorEnvelope, "description": "Model is busy or a different id is loaded."},
        503: {"model": ErrorEnvelope, "description": "Model slot is not attached."},
    },
    summary="Hot-load a model (asynchronous)",
    description=(
        "Start loading the registered backend for ``model_id``. The call "
        "is asynchronous: it returns **202** ``{status: 'loading'}`` "
        "immediately while construction + ``load()`` (potentially a slow "
        "weight download / GPU warmup) continue in the background. Poll "
        "``GET /v1/models`` and watch this id's ``load_status`` — it ends "
        "at ``loaded`` (``dimensions`` populated) or ``failed`` with a "
        "``load_error`` message; there is no synchronous failure "
        "response. If the same id is already held, the request is "
        "idempotent and returns **200** ``{status: 'loaded'}`` without "
        "rebuilding. A *different* id in the same family returns 409 "
        "``conflict_loaded`` (call ``/unload`` first); concurrent "
        "load/unload on the same family returns 409 ``model_busy``."
    ),
)
async def load_model(model_id: str, request: Request, response: Response):
    family, slot, cls, factory, _settings, state_attr = _resolve_family(request, model_id)
    # Fast phase: non-blocking lock + idempotent/conflict settlement.
    # ``begin_load`` performs no I/O, so it is safe on the event loop.
    try:
        instance = slot.begin_load(cls)
    except ConcurrentModelOperation as exc:
        raise _http_error(409, "model_busy", str(exc), model=model_id, family=family)
    except DifferentModelLoaded as exc:
        raise _http_error(409, "conflict_loaded", str(exc), model=model_id, family=family)

    if instance is not None:
        # Idempotent hit — the slot already holds this id. 200, not 202.
        response.status_code = 200
        return ModelLoadResponse(
            id=instance.model_name,
            type=family,  # type: ignore[arg-type]
            status="loaded",
            dimensions=getattr(instance, "dim", None),
        )

    # Slow phase runs on a worker thread; settle bookkeeping when done.
    app = request.app
    model_id_local = model_id

    async def _settle() -> None:
        try:
            loaded = await run_in_model(slot.finish_load, factory)
        except asyncio.CancelledError:
            # Server shutting down mid-load. The executor thread is NOT
            # cancelled: it keeps running finish_load, which settles the
            # slot state and releases the lock on its own. Just propagate.
            raise
        except Exception as exc:  # noqa: BLE001 — slot already recorded ``failed``
            # Any failure (family error OR unexpected) is observable via
            # GET /v1/models (load_status=failed / load_error); nothing is
            # installed on app.state. Don't let it surface as an unretrieved
            # task exception.
            log.warning(
                "model_load_failed",
                model=model_id_local,
                family=family,
                error=str(exc) or f"failed to load {model_id_local}",
                exception_type=type(exc).__name__,
            )
            return
        # Mirror the slot into ``app.state.<family>`` so the inference
        # routes see the freshly loaded instance immediately.
        setattr(app.state, state_attr, loaded)
        MODEL_LOADED.labels(kind=family).set(1)
        log.info(
            "model_loaded",
            model=loaded.model_name,
            family=family,
            device=getattr(loaded, "_device", "unknown"),
            dim=getattr(loaded, "dim", None),
        )

    task = asyncio.create_task(_settle())
    tasks = app.state._model_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return ModelLoadResponse(
        id=model_id,
        type=family,  # type: ignore[arg-type]
        status="loading",
        dimensions=None,
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
    try:
        released = await run_in_model(slot.unload)
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
