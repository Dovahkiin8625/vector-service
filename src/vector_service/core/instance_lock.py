"""Process-wide single-instance lock.

The SQLite corpus is the system of record and the ingest / maintenance
workers run in-process, so two service processes against the same
corpus directory are unsupported: duplicate job workers, no
cross-process row locking for model lifecycle, split-brain derived
state. Until the repository moves to Postgres (see TODO §6), startup
takes an exclusive OS byte-range lock on a file in the corpus
directory. A second instance fails fast with
:class:`InstanceAlreadyRunning` instead of corrupting shared state.

The lock is an OS lock, not a sentinel file: the operating system
releases it automatically if the process crashes, so no stale-lock
cleanup is ever needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType
from typing import Self

from vector_service.core.logging import get_logger

log = get_logger(__name__)


class InstanceAlreadyRunning(RuntimeError):
    """Another service process holds the corpus instance lock."""


class InstanceLock:
    """Exclusive file lock held for the service process lifetime."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: object | None = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def acquire(self) -> None:
        """Take the exclusive lock; raise if another process holds it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The byte range must cover a real byte: locks past EOF do not
        # conflict across processes. Then open ``r+b`` rather than
        # ``a+b`` — append-mode fds do not lock from the expected offset.
        self.path.touch(exist_ok=True)
        if self.path.stat().st_size == 0:
            self.path.write_bytes(b"x")
        fh = self.path.open("r+b")
        try:
            fh.seek(0)
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise InstanceAlreadyRunning(self._message()) from exc
            else:
                import fcntl

                try:
                    fcntl.flock(
                        fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except OSError as exc:
                    raise InstanceAlreadyRunning(self._message()) from exc
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        log.info("instance_lock_acquired", path=str(self.path))

    def release(self) -> None:
        """Release the lock; no-op when not held."""
        fh = self._fh
        if fh is None:
            return
        self._fh = None
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                log.warning("instance_lock_unlock_failed", path=str(self.path))
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                log.warning("instance_lock_unlock_failed", path=str(self.path))
        fh.close()

    def _message(self) -> str:
        return (
            f"another vector-service process already holds {self.path}; "
            "single-instance is required until the corpus moves to "
            "Postgres — stop the other process or point "
            "VS_INSTANCE_LOCK_PATH elsewhere"
        )
