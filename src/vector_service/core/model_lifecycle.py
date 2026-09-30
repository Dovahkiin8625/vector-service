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

Asynchronous loading
---------------------

``POST /v1/models/{id}/load`` must not block for the (potentially very
slow) factory + ``load()`` call. The hot-load route therefore uses a
two-phase API:

1. :meth:`ModelSlot.begin_load` — a fast, non-blocking phase run on the
   event loop: acquire the lock, settle idempotent/conflict cases, and
   flip the observable state to ``loading``. The lock is LEFT HELD.
2. :meth:`ModelSlot.finish_load` — the slow phase, dispatched to a
   thread executor: construct + ``load()`` the instance, install it, and
   release the lock in ``finally``.

``threading.Lock`` is not owned by the thread that acquired it, so
releasing it from the executor thread (after acquiring on the event-loop
thread) is legal and well-defined. Between the two phases every other
load/unload sees ``ConcurrentModelOperation`` exactly as before.

The observable state — ``unloaded`` / ``loading`` / ``loaded`` /
``failed`` plus the target id and last error — is what ``GET /v1/models``
reports per registered id, so clients poll the list while a background
load is in flight.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

# Observable load states. Writes happen under ``_lock``; lock-free reads
# of these references are safe under CPython's atomic attribute access.
STATE_UNLOADED = "unloaded"
STATE_LOADING = "loading"
STATE_LOADED = "loaded"
STATE_FAILED = "failed"


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
        # Async-observable load state. ``_loading_id`` is the target of
        # the current/most-recent load, so GET /v1/models can mark the
        # one card that is loading or last failed (the slot itself is
        # per-family while the listing is per-model-id).
        self._load_state: str = STATE_UNLOADED
        self._loading_id: str | None = None
        self._load_error: str | None = None
        self._load_error_type: str | None = None
        # Wall-clock seconds of the last successful build + ``load()``.
        # Surfaced on GET /v1/models (``model_info.load_duration_seconds``)
        # so the dashboard card can show how long the (possibly
        # download/warmup-bound) load actually took. ``None`` when unknown
        # or reset by a subsequent unload / failed load.
        self._load_duration: float | None = None

    # ---- introspection ------------------------------------------------

    @property
    def kind(self) -> str:
        """The model family this slot belongs to (``"embedder"`` etc.)."""
        return self._kind

    def get(self) -> T | None:
        """Return the currently loaded instance, or ``None`` if empty."""
        return self._instance

    @property
    def load_state(self) -> str:
        """One of ``unloaded`` / ``loading`` / ``loaded`` / ``failed``."""
        return self._load_state

    @property
    def loading_id(self) -> str | None:
        """Model id targeted by the current/most-recent load, if any."""
        return self._loading_id

    @property
    def load_error(self) -> str | None:
        """Error message from the last failed load, or ``None``."""
        return self._load_error

    @property
    def load_error_type(self) -> str | None:
        """Exception class name from the last failed load, if any."""
        return self._load_error_type

    @property
    def load_duration(self) -> float | None:
        """Wall-clock seconds the last successful load took, if measured."""
        return self._load_duration

    def set_instance(
        self, instance: T, *, load_duration: float | None = None
    ) -> None:
        """Inject an already-loaded instance (lifespan path only).

        Used by ``lifespan`` after the eager load completes so the
        slot, the inference routes (``app.state.<family>``), and the
        hot-reload routes all agree on the same single source of
        truth.

        Unlike ``load``, this method does NOT call ``load()`` on the
        instance — the caller has already done that. It is also not
        intended to be called from the hot-reload routes; they go
        through ``begin_load``/``finish_load`` so they participate in
        the lock + inflight contract.

        The observable state is flipped to ``loaded`` so GET /v1/models
        reports an eagerly-loaded instance exactly like a hot-loaded one.

        ``load_duration`` optionally records how long the lifespan's
        build + ``load()`` took; pass ``None`` (the default) when it was
        not measured.
        """
        self._instance = instance
        self._load_state = STATE_LOADED
        self._loading_id = getattr(instance, "model_name", None)
        self._load_error = None
        self._load_error_type = None
        self._load_duration = load_duration

    @property
    def loaded_id(self) -> str | None:
        """``model_name`` of the loaded instance, or ``None`` if empty."""
        inst = self._instance
        return getattr(inst, "model_name", None)

    # ---- lifecycle ----------------------------------------------------

    def _check_current(self, cls: type[T]) -> T | None:
        """Settle the idempotent / conflict cases against the instance.

        Callers MUST hold ``_lock``. Returns the existing instance when
        its id matches ``cls.model_name`` (idempotent re-load), raises
        :class:`DifferentModelLoaded` when another id is held, and
        returns ``None`` when the slot is empty.
        """
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
        return None

    def _mark_failed(self, exc: BaseException) -> None:
        """Record a failed load. Callers MUST hold ``_lock``."""
        self._load_state = STATE_FAILED
        self._load_error = str(exc) or f"failed to load {self._loading_id}"
        self._load_error_type = type(exc).__name__

    @staticmethod
    def _release_partial(new: T) -> None:
        """Best-effort cleanup of a half-constructed instance.

        ``load()`` failed after the factory returned the object — give
        the concrete class a chance to free whatever its constructor
        allocated (GPU handles, tokenizers, …). Cleanup itself must not
        mask the original load exception, so errors are swallowed.
        """
        try:
            new.unload()
        except Exception:  # noqa: BLE001,S110 - best-effort teardown, original exc must propagate
            pass

    def _build_and_install(self, cls: type[T], factory: Callable[[], T]) -> T:
        """Run factory + ``load()`` and install the result.

        Callers MUST hold ``_lock`` and have already ruled out the
        idempotent / conflict cases. On failure the partial instance's
        ``unload()`` is best-effort invoked and the observable state is
        flipped to ``failed`` before the exception is re-raised.
        """
        self._load_state = STATE_LOADING
        self._loading_id = getattr(cls, "model_name", None)
        self._load_error = None
        self._load_error_type = None
        self._load_duration = None

        t0 = time.perf_counter()
        try:
            new = factory()
        except BaseException as exc:
            self._mark_failed(exc)
            raise
        try:
            new.load()
        except BaseException as exc:
            # Factory returned an instance but load failed: free any
            # internal state the constructor may have allocated so we
            # don't leak partial resources.
            self._release_partial(new)
            self._mark_failed(exc)
            raise

        self._instance = new
        self._load_state = STATE_LOADED
        self._load_duration = time.perf_counter() - t0
        return new

    def load(
        self,
        cls: type[T],
        factory: Callable[[], T],
    ) -> T:
        """Build and load a new instance, replacing any existing one.

        Synchronous, lock-wrapping variant used by the lifespan eager
        load and by direct callers/tests. The hot-load route uses the
        split :meth:`begin_load` / :meth:`finish_load` pair instead so
        the HTTP request returns before the slow phase.

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
            current = self._check_current(cls)
            if current is not None:
                return current
            return self._build_and_install(cls, factory)
        finally:
            self._lock.release()

    # ---- asynchronous two-phase load ----------------------------------

    def begin_load(self, cls: type[T]) -> T | None:
        """Fast phase of an asynchronous load (run on the event loop).

        Performs only non-blocking work:

        1. Acquire the lock non-blocking; raise
           :class:`ConcurrentModelOperation` if another load/unload is
           in flight.
        2. Idempotent hit (same id already loaded): release the lock
           and return the existing instance.
        3. Conflict (a different id is held): release the lock and
           raise :class:`DifferentModelLoaded`.
        4. Empty slot: flip state to ``loading`` and return ``None`` —
           **the lock is left held** until the caller runs
           :meth:`finish_load` (in a thread executor). ``threading.Lock``
           is not bound to the acquiring thread, so the executor thread
           may release it.
        """
        if not self._lock.acquire(blocking=False):
            raise ConcurrentModelOperation(
                f"{self._kind}: another load/unload is already in progress"
            )
        try:
            current = self._check_current(cls)
        except BaseException:
            self._lock.release()
            raise
        if current is not None:
            self._lock.release()
            return current
        self._load_state = STATE_LOADING
        self._loading_id = getattr(cls, "model_name", None)
        self._load_error = None
        self._load_error_type = None
        self._load_duration = None
        return None

    def finish_load(self, factory: Callable[[], T]) -> T:
        """Slow phase of an asynchronous load (run in a thread executor).

        Precondition: a prior :meth:`begin_load` returned ``None`` and
        left this slot's lock held. Runs the factory and the instance's
        ``load()``, installs the result (state ``loaded``) or records
        the failure (state ``failed`` + error), and ALWAYS releases the
        lock in ``finally`` — including when the awaiting coroutine is
        cancelled (the executor thread itself is not cancelled).
        """
        t0 = time.perf_counter()
        try:
            try:
                new = factory()
            except BaseException as exc:
                self._mark_failed(exc)
                raise
            try:
                new.load()
            except BaseException as exc:
                self._release_partial(new)
                self._mark_failed(exc)
                raise
            self._instance = new
            self._load_state = STATE_LOADED
            self._load_duration = time.perf_counter() - t0
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
                self._load_state = STATE_UNLOADED
                self._loading_id = None
                self._load_error = None
                self._load_error_type = None
                self._load_duration = None
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
    app.state._slot_embedder = ModelSlot("embedder")
    app.state._slot_image = ModelSlot("image_embedder")
    app.state._slot_multimodal = ModelSlot("multimodal_embedder")
    app.state._slot_reranker = ModelSlot("reranker")
    # Family mirrors start empty; lifespan and the hot-load routes replace
    # these as instances become available, and inference routes read them
    # directly — so every mirror must exist from the start.
    app.state.embedder = None
    app.state.image_embedder = None
    app.state.multimodal_embedder = None
    app.state.reranker = None
    app.state.parser = None
    # Strong refs to in-flight async-load tasks so the garbage collector
    # can't reap a task mid-load; each task removes itself on completion.
    app.state._model_tasks = set()
    app.state.settings = settings
