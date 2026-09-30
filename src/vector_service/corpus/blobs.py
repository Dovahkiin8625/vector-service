"""Content-addressed store for original upload binaries.

Raw document bytes never enter SQLite (they would bloat every backup and
every page read). They live here as immutable, SHA-256 addressed files and
the corpus references them through ``documents.content_hash`` — the on-disk
path is derivable from the hash, so no extra columns are needed:

    <root>/<hash[:2]>/<hash>

Identical uploads share one blob. A blob outlives a single document and is
removed only when *zero* documents reference its hash — either eagerly by
the delete routes (the repository returns orphaned hashes) or by the
maintenance worker's sweep, which also catches anything left behind by a
crash mid-delete.

Every method here is synchronous filesystem I/O; async callers run them
through ``asyncio.to_thread`` (default executor), never on the event loop.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path


class BlobStore:
    """Filesystem layout + promotion/GC primitives for original bytes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, digest: str) -> Path:
        """Absolute blob path for ``digest`` (file may not exist yet)."""
        return self.root / digest[:2] / digest

    def has(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    def put_bytes(self, data: bytes, digest: str) -> None:
        """Write ``data`` as the blob for ``digest`` unless it exists.

        Used when the source is in memory rather than a spool file.
        """
        target = self.path_for(digest)
        if target.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{digest}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    def promote(self, src: str | Path, digest: str) -> None:
        """Move a finished spool upload into the store under ``digest``.

        A blob with the same hash (identical content) already stored means
        the source is simply discarded. Afterwards the spool directory is
        removed — it only ever held this one upload.
        """
        src = Path(src)
        target = self.path_for(digest)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(src, target)
            except OSError:
                # Different volumes (EXDEV / Windows cross-drive) — copy.
                shutil.copy2(src, target)
                src.unlink()
        elif src.is_file():
            src.unlink()
        shutil.rmtree(src.parent, ignore_errors=True)

    def delete(self, digest: str) -> bool:
        """Remove one blob; return whether a file was actually deleted.

        Empty shard directories are pruned up to the store root.
        """
        target = self.path_for(digest)
        if not target.is_file():
            return False
        target.unlink()
        shard = target.parent
        try:
            shard.rmdir()
        except OSError:
            pass
        return True

    def iter_digests(self) -> Iterator[str]:
        """Yield every stored digest (one blob filename per shard)."""
        if not self.root.is_dir():
            return
        for shard in self.root.iterdir():
            if not shard.is_dir():
                continue
            for blob in shard.iterdir():
                if blob.is_file() and not blob.name.startswith("."):
                    yield blob.name
