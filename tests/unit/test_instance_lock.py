"""Single-instance OS byte-range lock (TODO §6).

Pins:

- acquire/release round trip on a path that did not exist;
- a second lock while the first is held fails fast with
  InstanceAlreadyRunning (same process, second handle — the OS lock
  conflict the second service process would hit);
- after release the lock is reacquirable;
- the context manager releases on exit;
- the lock file is created with a real byte (range locks past EOF do
  not conflict).
"""
from __future__ import annotations

import pytest

from vector_service.core.instance_lock import (
    InstanceAlreadyRunning,
    InstanceLock,
)


def test_lock_creates_file_and_round_trip(tmp_path):
    lock_path = tmp_path / "nested" / "dir" / "instance.lock"
    lock = InstanceLock(lock_path)
    lock.acquire()
    try:
        assert lock_path.is_file()
        assert lock_path.stat().st_size >= 1
    finally:
        lock.release()


def test_second_lock_while_held_raises(tmp_path):
    lock_path = tmp_path / "instance.lock"
    first = InstanceLock(lock_path)
    first.acquire()
    try:
        second = InstanceLock(lock_path)
        with pytest.raises(InstanceAlreadyRunning):
            second.acquire()
        # The failed acquire must not leave its handle open.
        assert second._fh is None
    finally:
        first.release()


def test_lock_reacquirable_after_release(tmp_path):
    lock_path = tmp_path / "instance.lock"
    first = InstanceLock(lock_path)
    first.acquire()
    first.release()

    second = InstanceLock(lock_path)
    second.acquire()
    second.release()


def test_context_manager_releases_on_exit(tmp_path):
    lock_path = tmp_path / "instance.lock"
    with InstanceLock(lock_path) as lock:
        assert lock._fh is not None
    # A fresh lock can take over immediately.
    with InstanceLock(lock_path):
        pass


def test_release_without_acquire_is_noop(tmp_path):
    InstanceLock(tmp_path / "instance.lock").release()
