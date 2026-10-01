"""Content-addressed blob store (:mod:`vector_service.corpus.blobs`)."""
from __future__ import annotations

from pathlib import Path

import pytest

from vector_service.corpus.blobs import BlobStore

#: Fixed 64-char hex digest used so paths are deterministic.
DIGEST = "a" * 64
DIGEST2 = "b" * 64


@pytest.fixture
def store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "originals")


def test_path_uses_two_char_shard(store: BlobStore) -> None:
    path = store.path_for(DIGEST)
    assert path.name == DIGEST
    assert path.parent.name == "aa"


def test_put_bytes_writes_blob_idempotently(store: BlobStore) -> None:
    store.put_bytes(b"hello", DIGEST)
    target = store.path_for(DIGEST)
    assert target.read_bytes() == b"hello"
    assert store.has(DIGEST)
    # A second write must not overwrite or raise — identical hash means
    # identical content.
    store.put_bytes(b"hello", DIGEST)
    assert target.read_bytes() == b"hello"


def test_promote_moves_spool_and_removes_dir(
    store: BlobStore, tmp_path: Path
) -> None:
    spool_dir = tmp_path / "jobs" / "job1"
    spool_dir.mkdir(parents=True)
    upload = spool_dir / "upload"
    upload.write_bytes(b"payload")

    store.promote(upload, DIGEST)

    assert store.path_for(DIGEST).read_bytes() == b"payload"
    assert not upload.exists()
    assert not spool_dir.exists()


def test_promote_duplicate_hash_discards_source(
    store: BlobStore, tmp_path: Path
) -> None:
    store.put_bytes(b"payload", DIGEST)
    spool_dir = tmp_path / "jobs" / "job2"
    spool_dir.mkdir(parents=True)
    upload = spool_dir / "upload"
    upload.write_bytes(b"payload")

    store.promote(upload, DIGEST)

    assert not upload.exists()
    assert not spool_dir.exists()


def test_promote_falls_back_to_copy_when_replace_fails(
    store: BlobStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vector_service.corpus.blobs as blobs_mod

    def _boom(src, dst):
        raise OSError("cross-device link")

    monkeypatch.setattr(blobs_mod.os, "replace", _boom)
    spool_dir = tmp_path / "jobs" / "job3"
    spool_dir.mkdir(parents=True)
    upload = spool_dir / "upload"
    upload.write_bytes(b"payload")

    store.promote(upload, DIGEST2)

    assert store.path_for(DIGEST2).read_bytes() == b"payload"
    assert not upload.exists()


def test_delete_removes_blob_and_prunes_shard(store: BlobStore) -> None:
    store.put_bytes(b"hello", DIGEST)
    assert store.delete(DIGEST)
    assert not store.has(DIGEST)
    assert not (store.root / "aa").exists()
    # Deleting a missing blob is a no-op False, not an error.
    assert not store.delete(DIGEST)


def test_iter_digests_lists_every_blob(store: BlobStore) -> None:
    assert list(store.iter_digests()) == []
    store.put_bytes(b"one", DIGEST)
    store.put_bytes(b"two", DIGEST2)
    assert sorted(store.iter_digests()) == sorted([DIGEST, DIGEST2])
