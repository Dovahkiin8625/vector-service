"""Client-side BM25 sparse encoding for the thin vector index.

The vector store holds only sparse *vectors* (no text, no analyzer) with a
``SPARSE_INVERTED_INDEX`` / ``IP`` index. This service produces them:

- documents are encoded after fitting BM25 statistics over the corpus;
- queries are encoded to IDF-weighted sparse vectors;
- the fitted statistics are persisted per ``(database, collection)`` and are
  themselves **derived**: a missing state file is rebuilt from the corpus.

The analyzer is built explicitly (jieba tokenizer + CJK punctuation filter)
to avoid the NLTK stopword-data dependency that the bundled ``zh`` default
analyzer would require. Chinese stopword removal adds little for BM25 and
would need an offline NLTK corpus.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from milvus_model.sparse.bm25.bm25 import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import (
    Analyzer,
    JiebaTokenizer,
    PunctuationFilter,
)

# CJK/full-width punctuation on top of ``string.punctuation`` (which the
# PunctuationFilter already includes).
_CJK_PUNCT = "，。、；：？！「」『』《》〈〉【】〖〗（）〔〕—…·～％＃＊"


def _chinese_analyzer() -> Analyzer:
    return Analyzer(
        name="corpus-zh",
        tokenizer=JiebaTokenizer(),
        preprocessors=[],
        filters=[PunctuationFilter(extras=_CJK_PUNCT)],
    )


def _sparse_to_dict(csr_row: Any) -> dict[int, float]:
    """Convert one scipy CSR row to Milvus' sparse-vector dict shape."""
    coo = csr_row.tocoo()
    return {int(col): float(val) for col, val in zip(coo.col, coo.data)}


class SparseBM25:
    """Per-collection BM25 statistics and encoders."""

    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._efs: dict[tuple[str, str], BM25EmbeddingFunction] = {}
        self._locks: dict[tuple[str, str], threading.RLock] = {}
        self._guards_lock = threading.Lock()

    # ---- internals ------------------------------------------------------

    def _lock_for(self, key: tuple[str, str]) -> threading.RLock:
        with self._guards_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    def _state_file(self, database: str, collection: str) -> Path:
        safe_db = re.sub(r"[^A-Za-z0-9_.-]", "_", database)
        safe_coll = re.sub(r"[^A-Za-z0-9_.-]", "_", collection)
        return self.state_dir / f"{safe_db}__{safe_coll}.bm25.json"

    @staticmethod
    def _new_ef() -> BM25EmbeddingFunction:
        return BM25EmbeddingFunction(analyzer=_chinese_analyzer())

    def _load_or_rebuild(
        self,
        repo: Any,
        key: tuple[str, str],
        state_file: Path,
        logical_collection: str,
    ) -> BM25EmbeddingFunction:
        """Return a ready EF: cache → state file → rebuild from the corpus.

        Stats describe the logical collection's leaves even when they
        are cached/saved under a rebuilt physical collection name.
        Caller must hold the per-key lock.
        """
        ef = self._efs.get(key)
        if ef is not None:
            return ef
        ef = self._new_ef()
        if state_file.is_file():
            ef.load(str(state_file))
        else:
            database, collection = key
            texts = repo.leaf_chunk_texts(database, logical_collection)
            if not texts:
                raise RuntimeError(
                    f"cannot build BM25 stats: no chunks in {database!r}/"
                    f"{logical_collection!r} (physical {collection!r})"
                )
            ef.fit(texts)
            ef.save(str(state_file))
        self._efs[key] = ef
        return ef

    # ---- public API -----------------------------------------------------

    def fit_for_ingest(self, repo: Any, database: str, collection: str) -> None:
        """Refit stats over the whole corpus and persist them.

        Called by ingest AFTER the new chunks have been stored, so the
        statistics include them. Previously encoded document vectors keep
        their values (a full sparse-index rebuild is a separate job).
        """
        key = (database, collection)
        with self._lock_for(key):
            texts = repo.leaf_chunk_texts(database, collection)
            ef = self._new_ef()
            ef.fit(texts)
            ef.save(str(self._state_file(database, collection)))
            self._efs[key] = ef

    def fit_for_rebuild(
        self,
        repo: Any,
        database: str,
        logical_collection: str,
        physical_collection: str,
    ) -> None:
        """Fit fresh stats for a rebuilt index and persist them.

        Stats come from the logical collection's current leaves but are
        cached/saved under the new physical name: the rebuilt sparse
        vectors are all encoded against one fresh statistics set,
        independently of the old index still serving traffic.
        """
        key = (database, physical_collection)
        with self._lock_for(key):
            texts = repo.leaf_chunk_texts(database, logical_collection)
            ef = self._new_ef()
            ef.fit(texts)
            ef.save(str(self._state_file(database, physical_collection)))
            self._efs[key] = ef

    def ensure_ready(
        self,
        repo: Any,
        database: str,
        logical_collection: str,
        physical_collection: str,
    ) -> None:
        """Make sure a physical collection's EF is loaded before encoding.

        Cache → physical state file → rebuild from the logical
        collection's leaves. Used by repair and query encoding.
        """
        key = (database, physical_collection)
        with self._lock_for(key):
            self._load_or_rebuild(
                repo,
                key,
                self._state_file(database, physical_collection),
                logical_collection,
            )

    def encode_documents(
        self, database: str, collection: str, texts: list[str]
    ) -> list[dict[int, float]]:
        """Encode chunk texts to sparse vectors (post fit_for_ingest)."""
        key = (database, collection)
        with self._lock_for(key):
            ef = self._efs[key]
            matrix = ef.encode_documents(texts)
            return [_sparse_to_dict(matrix[i]) for i in range(matrix.shape[0])]

    def discard(self, database: str, collection: str) -> None:
        """Drop derived BM25 state for one collection (cache + stats file).

        Called when the collection is removed; a future collection with
        the same name rebuilds stats from the corpus on first query.
        """
        key = (database, collection)
        with self._lock_for(key):
            self._efs.pop(key, None)
            state_file = self._state_file(database, collection)
            if state_file.is_file():
                state_file.unlink()

    def encode_query(
        self,
        repo: Any,
        database: str,
        logical_collection: str,
        physical_collection: str,
        query: str,
    ) -> dict[int, float]:
        """Encode one query against a physical index's stats.

        Missing stats rebuild from the logical collection's leaves
        (e.g. after a delete invalidated them).
        """
        key = (database, physical_collection)
        with self._lock_for(key):
            ef = self._load_or_rebuild(
                repo,
                key,
                self._state_file(database, physical_collection),
                logical_collection,
            )
            matrix = ef.encode_queries([query])
            return _sparse_to_dict(matrix[0])
