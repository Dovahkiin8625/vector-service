"""Isolated bounded thread pools (:mod:`vector_service.core.threadpools`).

Covers:

- :class:`PoolSettings` bounds and the ``max_pending >= workers`` rule;
- env overrides on :class:`RuntimeSettings`;
- unbound helpers falling back to ``asyncio.to_thread``;
- bound dispatch onto the named worker threads, with args/kwargs;
- admission: with one worker, inflight never exceeds one and surplus
  callers queue (``waiting``);
- isolation: a saturated ``store`` pool cannot stall the ``model`` pool;
- per-event-loop semaphore bookkeeping and ``describe`` / ``shutdown``.
"""
from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic import ValidationError

from vector_service.core.threadpools import (
    ThreadPools,
    bind_pools,
    run_in,
    run_in_model,
    run_in_sqlite,
    run_in_store,
    unbind_pools,
)

# ---- fixtures / helpers ------------------------------------------------


def _runtime(*, store=(2, 16), model=(2, 16), sqlite=(2, 16)):
    from vector_service.core.config import RuntimeSettings

    return RuntimeSettings(
        store_workers=store[0],
        store_max_pending=store[1],
        model_workers=model[0],
        model_max_pending=model[1],
        sqlite_workers=sqlite[0],
        sqlite_max_pending=sqlite[1],
    )


class _BoundPools:
    """Bind fresh pools for one test and guarantee teardown."""

    def __init__(self, **kwargs):
        self.pools = ThreadPools(_runtime(**kwargs))
        bind_pools(self.pools)

    def close(self):
        unbind_pools()
        self.pools.shutdown()


@pytest.fixture
def bound():
    unbind_pools()
    handle = _BoundPools()
    try:
        yield handle.pools
    finally:
        handle.close()


# ---- settings ----------------------------------------------------------


def test_pool_settings_accepts_valid_shape():
    from vector_service.core.config import PoolSettings

    settings = PoolSettings(workers=4, max_pending=32)
    assert settings.workers == 4
    assert settings.max_pending == 32


@pytest.mark.parametrize("workers", [0, -1, 65])
def test_pool_settings_rejects_worker_bounds(workers):
    from vector_service.core.config import PoolSettings

    with pytest.raises(ValidationError):
        PoolSettings(workers=workers, max_pending=32)


def test_pool_settings_rejects_pending_below_workers():
    from vector_service.core.config import PoolSettings

    with pytest.raises(ValidationError) as exc_info:
        PoolSettings(workers=8, max_pending=4)
    assert "max_pending" in str(exc_info.value)


def test_pool_settings_rejects_pending_bounds():
    from vector_service.core.config import PoolSettings

    with pytest.raises(ValidationError):
        PoolSettings(workers=1, max_pending=5000)


def test_runtime_settings_defaults():
    from vector_service.core.config import RuntimeSettings

    settings = RuntimeSettings()
    assert (settings.store.workers, settings.store.max_pending) == (8, 64)
    assert (settings.model.workers, settings.model.max_pending) == (2, 16)
    assert (settings.sqlite.workers, settings.sqlite.max_pending) == (4, 32)


def test_runtime_settings_env_override(monkeypatch):
    from vector_service.core.config import RuntimeSettings

    monkeypatch.setenv("VS_RUNTIME__STORE_WORKERS", "3")
    monkeypatch.setenv("VS_RUNTIME__SQLITE_MAX_PENDING", "128")
    settings = RuntimeSettings()
    assert settings.store.workers == 3
    assert settings.sqlite.max_pending == 128


# ---- unbound fallback --------------------------------------------------


def test_unbound_helpers_fall_back_to_default_executor():
    unbind_pools()

    async def scenario():
        value = await run_in_store(lambda x, *, y: (x, y, threading.current_thread().name), 7, y="z")
        return value

    x, y, thread_name = asyncio.run(scenario())
    assert (x, y) == (7, "z")
    assert thread_name != "MainThread"


def test_run_in_unbound_passes_through():
    unbind_pools()
    assert asyncio.run(run_in("model", lambda a, b: a + b, 2, b=3)) == 5


# ---- bound dispatch ----------------------------------------------------


def test_bound_dispatch_uses_named_worker(bound):
    async def scenario():
        names = await asyncio.gather(
            run_in_store(lambda: threading.current_thread().name),
            run_in_model(lambda: threading.current_thread().name),
            run_in_sqlite(lambda: threading.current_thread().name),
        )
        return names

    store_name, model_name, sqlite_name = asyncio.run(scenario())
    assert store_name.startswith("vs-store")
    assert model_name.startswith("vs-model")
    assert sqlite_name.startswith("vs-sqlite")


def test_bound_dispatch_returns_values(bound):
    def work(a, b, *, c):
        return a + b + c

    async def scenario():
        return await asyncio.gather(
            run_in_store(work, 1, 2, c=3),
            run_in_model(work, 4, 5, c=6),
            run_in_sqlite(work, 7, 8, c=9),
        )

    assert asyncio.run(scenario()) == [6, 15, 24]


# ---- admission / backpressure -----------------------------------------


def test_admission_queues_surplus_and_serializes():
    # One worker, two admissions: four submissions → two admitted to the
    # executor (one running, one queued there), two waiting at the
    # semaphore. All four work calls must still run one at a time.
    handle = _BoundPools(store=(1, 2))
    pools = handle.pools
    release = threading.Event()
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def work():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        release.wait(5.0)
        with state_lock:
            active -= 1
        return "ok"

    async def scenario():
        tasks = [asyncio.create_task(run_in_store(work)) for _ in range(4)]
        stats = pools.describe()["store"]
        while not (stats["inflight"] == 2 and stats["waiting"] == 2):
            await asyncio.sleep(0.005)
            stats = pools.describe()["store"]
        # Admitted-but-queued work must not start while the first runs.
        release.set()
        return await asyncio.gather(*tasks)

    try:
        assert asyncio.run(scenario()) == ["ok"] * 4
    finally:
        handle.close()
    assert max_active == 1
    stats = pools.describe()["store"]
    assert stats["scheduled"] == 4
    assert stats["inflight"] == 0
    assert stats["waiting"] == 0


def test_pools_are_isolated():
    # One store worker, blocked; the model pool must still make progress.
    handle = _BoundPools(store=(1, 4), model=(1, 4))
    pools = handle.pools
    block = threading.Event()

    def blocked_store_call():
        block.wait(5.0)
        return "store"

    def fast_model_call():
        return "model"

    async def scenario():
        store_task = asyncio.create_task(run_in_store(blocked_store_call))
        while pools.describe()["store"]["inflight"] < 1:
            await asyncio.sleep(0.005)
        model_result = await asyncio.wait_for(
            run_in_model(fast_model_call), timeout=3.0
        )
        block.set()
        store_result = await store_task
        return model_result, store_result

    try:
        assert asyncio.run(scenario()) == ("model", "store")
    finally:
        handle.close()


# ---- per-loop bookkeeping ---------------------------------------------


def test_semaphore_is_per_event_loop(bound):
    pool = bound._pools["sqlite"]

    async def one():
        await run_in_sqlite(lambda: None)

    asyncio.run(one())
    closed_loops = list(pool._semaphores)
    assert len(closed_loops) == 1
    assert closed_loops[0].is_closed()

    # Closed-loop entries are pruned lazily, on the next lookup — a new
    # loop gets a fresh full-value semaphore and never inherits waiters.
    loop_b = asyncio.new_event_loop()
    try:
        sem_b = pool.semaphore(loop_b)
        assert closed_loops[0] not in pool._semaphores
        assert list(pool._semaphores) == [loop_b]
        assert sem_b._value == bound.describe()["sqlite"]["max_pending"]
        result = loop_b.run_until_complete(
            run_in_sqlite(lambda: 2)
        )
    finally:
        loop_b.close()
    assert result == 2


# ---- describe ----------------------------------------------------------


def test_describe_shape(bound):
    stats = bound.describe()
    assert set(stats) == {"store", "model", "sqlite"}
    for pool_stats in stats.values():
        assert set(pool_stats) == {
            "workers", "max_pending", "inflight", "waiting", "scheduled",
        }
        assert pool_stats["inflight"] == 0
        assert pool_stats["waiting"] == 0
    assert stats["store"]["workers"] == 2
    assert stats["model"]["max_pending"] == 16
