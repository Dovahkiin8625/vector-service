"""In-process pub/sub bus for job progress — the fan-out point for SSE.

Single process by contract (``workers=1``): subscribers are SSE
response tasks, publishers are the worker loop and the cancel route,
all running on the same event loop, so no locking is involved. A nudge
carries only a coarse ``kind`` label — subscribers reload the
authoritative ``ingest_jobs`` row themselves, so the bus never holds
job state and can never disagree with SQLite.

Slow consumers are isolated: :meth:`JobEventBus.publish` never blocks.
A subscriber whose queue is full loses that nudge; the SSE endpoint's
periodic resync tick repairs the gap.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class JobEventBus:
    """Per-job nudge fan-out for live progress consumers."""

    def __init__(self, maxsize: int = 64) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(
            set
        )
        self._maxsize = maxsize

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Start collecting nudges for ``job_id``.

        Pair every subscription with an :meth:`unsubscribe` (e.g. in the
        streaming response's ``finally``).
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers[job_id].add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber; drops the empty job bucket outright."""
        subs = self._subscribers.get(job_id)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            del self._subscribers[job_id]

    def publish(self, job_id: str, kind: str) -> int:
        """Nudge every subscriber without blocking.

        Returns the number of subscribers that got the nudge. Full
        queues are skipped (their resync tick catches up).
        """
        subs = self._subscribers.get(job_id)
        if not subs:
            return 0
        payload: dict[str, Any] = {"kind": kind}
        delivered = 0
        for queue in tuple(subs):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                continue
            delivered += 1
        return delivered

    def subscriber_count(self, job_id: str) -> int:
        """How many live SSE subscriptions a job currently has."""
        return len(self._subscribers.get(job_id, ()))
