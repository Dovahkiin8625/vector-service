"""In-process background worker, event bus and startup recovery."""

from vector_service.jobs.bus import JobEventBus
from vector_service.jobs.recovery import JOB_INTERRUPTED, recover_interrupted
from vector_service.jobs.worker import IngestWorker

__all__ = [
    "JOB_INTERRUPTED",
    "IngestWorker",
    "JobEventBus",
    "recover_interrupted",
]
