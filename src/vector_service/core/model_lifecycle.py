"""Per-family model slot: single-instance holder + lifecycle lock.

Every model family the service exposes — text embedder, image embedder,
multimodal embedder, reranker — is a single-slot resource: at most one
loaded instance is attached to ``app.state`` at any time. ``ModelSlot``
encapsulates that invariant together with the non-blocking lock used to
serialise load/unload requests.

Why a dedicated holder instead of touching ``app.state.<family>`` from
the routes? Two reasons:

1. The slot owns the inflight bookkeeping. A second load/unload arriving
   while the first is still running must return 409 ``model_busy`` — the
   slot exposes that contract in one place rather than scattering
   ``threading.Lock`` across route handlers and lifespan.
2. ``unload`` semantics are family-agnostic. The slot calls
   ``instance.unload()`` on whatever concrete class the registry hands
   over; each concrete class knows how to release its own resources.

Concrete model classes (``Embedder`` / ``ImageEmbedder`` /
``MultimodalEmbedder`` / ``Reranker``) expose a symmetric ``load`` /
``unload`` pair. ``unload`` is idempotent and safe to call before
``load``.
"""
from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class ConcurrentModelOperation(Exception):
    """Raised when a load/unload is attempted while another is in flight.

    Maps to HTTP 409 ``model_busy``. Distinct from
    ``DifferentModelLoaded`` (which signals a *successful* prior load of
    a different id) — the slot has not yet finished a previous operation,
    so the second caller must retry.
    """


class DifferentModelLoaded(Exception):
    """Raised when ``load`` would replace a different model_id than the one
    the slot currently holds.

    Maps to HTTP 409 ``conflict_loaded``. Callers must ``unload`` first
    if they want to switch families.
    """


class ModelSlot(Generic[T]):
    """Single-instance slot for a model family with a non-blocking lock.

    Concurrency contract: ``load`` and ``unload`` use a ``threading.Lock``
    acquired with ``blocking=False``. If the lock is held by another
    thread, callers see ``ConcurrentModelOperation`` immediately rather
    than queueing — this keeps the FastAPI event loop responsive and
    surfaces a deterministic 409 ``model_busy`` to the client.

    The lock is held for the duration of the factory call (which may
    include a slow weight download or GPU warmup). Routes dispatch the
    call to a thread executor so the event loop is never blocked on the
    lock acquisition itself.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._instance: T | None = None
        self._lock = threading.Lock()

    # ---- introspection ------------------------------------------------

    @property
    def kind(self) -> str:
        """The model family this slot belongs to (``"embedder"`` etc.)."""
        return self._kind

    def get(self) -> T | None:
        """Return the currently loaded instance, or ``None`` if empty."""
        return self._instance

    def set_instance(self, instance: T) -> None:
        """Inject an already-loaded instance (lifespan path only).

        Used by ``lifespan`` after the eager load completes so the
        slot, the inference routes (``app.state.<family>``), and the
        hot-reload routes all agree on the same single source of
        truth.

        Unlike ``load``, this method does NOT call ``load()`` on the
        instance — the caller has already done that. It is also not
        intended to be called from the hot-reload routes; they go
        through ``load`` so they participate in the lock + inflight
        contract.
        """
        self._instance = instance

    @property
    def loaded_id(self) -> str | None:
        """``model_name`` of the loaded instance, or ``None`` if empty."""
        inst = self._instance
        return getattr(inst, "model_name", None)

    # ---- lifecycle ----------------------------------------------------

    def load(
        self,
        cls: type[T],
        factory: Callable[[], T],
    ) -> T:
        """Build and load a new instance, replacing any existing one.

        ``cls`` is the registered class whose ``model_name`` we use to
        decide whether the slot already holds an equivalent instance
        (idempotent re-load). ``factory`` constructs and returns a
        freshly-wired instance; ``load()`` is called on it inside the
        critical section.

        Order of operations:

        1. Acquire the lock non-blocking. Fail fast with
           ``ConcurrentModelOperation`` if busy.
        2. If the slot already holds an instance whose
           ``model_name`` matches ``cls.model_name``, return the
           existing instance without constructing — a no-op that
           keeps the eager-loaded model warm. Idempotent re-load
           semantics.
        3. If the slot holds an instance whose id *differs* from
           ``cls.model_name``, raise ``DifferentModelLoaded`` without
           constructing.
        4. Otherwise build the new instance, call ``load()`` on it,
           and install it. If the factory or ``load()`` raises,
           release the partial instance's resources and propagate.
        """
        if not self._lock.acquire(blocking=False):
            raise ConcurrentModelOperation(
                f"{self._kind}: another load/unload is already in progress"
            )
        try:
            new_id = getattr(cls, "model_name", None)
            current = self._instance
            if current is not None:
                current_id = getattr(current, "model_name", None)
                if current_id == new_id:
                    # Same id: idempotent — keep the existing instance
                    # warm. Skip both construction and load() entirely.
                    return current
                raise DifferentModelLoaded(
                    f"{self._kind}: a different model "
                    f"({current_id!r}) is already loaded; unload it "
                    f"before loading {new_id!r}"
                )

            new = factory()
            try:
                new.load()
            except BaseException:
                # Factory returned an instance but load failed: free
                # any internal state the constructor may have
                # allocated so we don't leak partial resources.
                try:
                    new.unload()
                except Exception:
                    pass
                raise

            self._instance = new
            return new
        finally:
            self._lock.release()

    def unload(self) -> bool:
        """Release the current instance and clear the slot.

        Returns ``True`` if an instance was released, ``False`` if the
        slot was already empty. Raises ``ConcurrentModelOperation`` if
        the lock is held by another thread.
        """
        if not self._lock.acquire(blocking=False):
            raise ConcurrentModelOperation(
                f"{self._kind}: another load/unload is already in progress"
            )
        try:
            current = self._instance
            if current is None:
                return False
            try:
                current.unload()
            finally:
                # Always clear the slot, even if unload raised — the
                # caller has explicitly asked us to release the resource,
                # and a stuck half-unloaded instance is worse than an
                # empty slot.
                self._instance = None
            return True
        finally:
            self._lock.release()


def attach_default_slots(app, *, settings) -> None:
    """Attach the four canonical ``ModelSlot`` objects to ``app.state``.

    Called from ``lifespan`` after the eager loads complete. Routes read
    these slots via ``request.app.state._slot_<kind>``. The slots start
    empty if the corresponding eager load failed; the route layer will
    populate them on the first ``POST /v1/models/{id}/load`` for that
    family.

    The function also publishes ``settings`` on ``app.state.settings``
    (overwriting whatever was there) so the hot-reload routes can read
    the per-family settings block when constructing a fresh instance.
    Lifespan sets this earlier, so the overwrite is a no-op in
    production; test fixtures that skip lifespan rely on this to
    bootstrap.

    Why expose slots on ``app.state`` instead of a registry singleton?
    The slots carry *process-local* state — the lock in particular is
    only meaningful inside one event loop. Keeping them on ``app.state``
    matches the convention used by ``app.state.embedder`` etc.
    """
    from vector_service.core.model_lifecycle import ModelSlot

    app.state._slot_embedder = ModelSlot("embedder")
    app.state._slot_image = ModelSlot("image_embedder")
    app.state._slot_multimodal = ModelSlot("multimodal_embedder")
    app.state._slot_reranker = ModelSlot("reranker")
    app.state.settings = settings
