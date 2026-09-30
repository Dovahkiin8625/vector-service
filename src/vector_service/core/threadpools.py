"""Isolated bounded thread pools for blocking calls.

Three pools, each with its own ``ThreadPoolExecutor`` and an admission
semaphore:

- ``store``  — vector-store RPC (network I/O waits)
- ``model``  — local model / CPU inference (embedder, reranker, BM25,
  document parsing)
- ``sqlite`` — corpus repository calls

Call sites use the module-level :func:`run_in_store` / :func:`run_in_model`
/ :func:`run_in_sqlite` helpers. The live :class:`ThreadPools` is bound in
the lifespan; with nothing bound (unit tests, scripts) the helpers fall
back to ``asyncio.to_thread`` on the default executor.
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vector_service.core.config import PoolSettings

PoolName = Literal["store", "model", "sqlite"]
POOL_NAMES: tuple[PoolName, ...] = ("store", "model", "sqlite")


class _Pool:
    def __init__(self, name: PoolName, settings: PoolSettings):
        self.max_pending = settings.max_pending
        self.executor = ThreadPoolExecutor(
            max_workers=settings.workers, thread_name_prefix=f"vs-{name}"
        )
        # One admission semaphore per event loop (loops come and go in
        # the test suite).
        self._semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
        self.inflight = 0
        self.waiting = 0
        self.scheduled = 0

    def semaphore(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        for old_loop in list(self._semaphores):
            if old_loop.is_closed():
                del self._semaphores[old_loop]
        sem = self._semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self.max_pending)
            self._semaphores[loop] = sem
        return sem


class ThreadPools:
    """Owned by the application lifespan; one executor per blocking class."""

    def __init__(self, runtime: Any):
        self._pools = {
            name: _Pool(name, getattr(runtime, name)) for name in POOL_NAMES
        }

    async def run(
        self,
        name: PoolName,
        func,
        /,
        *args: Any,
        **kwargs: Any,
    ):
        """Run ``func(*args, **kwargs)`` on pool ``name`` with admission."""
        pool = self._pools[name]
        loop = asyncio.get_running_loop()
        sem = pool.semaphore(loop)
        pool.waiting += 1
        try:
            await sem.acquire()
        finally:
            pool.waiting -= 1
        pool.inflight += 1
        pool.scheduled += 1
        try:
            return await loop.run_in_executor(
                pool.executor,
                functools.partial(func, *args, **kwargs),
            )
        finally:
            pool.inflight -= 1
            sem.release()

    def describe(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "workers": pool.executor._max_workers,
                "max_pending": pool.max_pending,
                "inflight": pool.inflight,
                "waiting": pool.waiting,
                "scheduled": pool.scheduled,
            }
            for name, pool in self._pools.items()
        }

    def shutdown(self) -> None:
        for pool in self._pools.values():
            pool.executor.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------
# Module-level binding + ergonomic helpers
# ---------------------------------------------------------------------------

_bound: ThreadPools | None = None


def bind_pools(pools: ThreadPools) -> None:
    global _bound
    _bound = pools


def unbind_pools() -> None:
    global _bound
    _bound = None


async def run_in(
    name: PoolName,
    func,
    /,
    *args: Any,
    **kwargs: Any,
):
    if _bound is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    return await _bound.run(name, func, *args, **kwargs)


async def run_in_store(func, /, *args: Any, **kwargs: Any):
    return await run_in("store", func, *args, **kwargs)


async def run_in_model(func, /, *args: Any, **kwargs: Any):
    return await run_in("model", func, *args, **kwargs)


async def run_in_sqlite(func, /, *args: Any, **kwargs: Any):
    return await run_in("sqlite", func, *args, **kwargs)
