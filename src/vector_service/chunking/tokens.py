"""Token counting / token-space helpers.

All chunkers size chunks in ``cl100k_base`` tokens — the same
encoding OpenAI's ``text-embedding-3-*`` models use — so configured
``chunk_size`` values are directly comparable to those models'
advertised limits.

``tiktoken`` is imported lazily and the encoding cached, so modules
that only need the :class:`~vector_service.chunking.base.Chunk` type
don't pay the import cost.
"""
from __future__ import annotations

# Lazy-loaded; ``_get_encoding`` caches the ``Encoding`` instance.
_ENCODING: object | None = None


def _get_encoding() -> object:
    """Lazy-load and cache the ``cl100k_base`` tiktoken encoding."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Return the token count for ``text`` under ``cl100k_base``."""
    enc = _get_encoding()
    return len(enc.encode(text))  # type: ignore[union-attr]


def encode_ids(text: str) -> list[int]:
    """Encode ``text`` to its ``cl100k_base`` token id sequence."""
    enc = _get_encoding()
    return list(enc.encode(text))  # type: ignore[union-attr]


def decode_ids(ids: list[int]) -> str:
    """Decode a ``cl100k_base`` token id sequence back to text."""
    enc = _get_encoding()
    return enc.decode(ids)  # type: ignore[union-attr]


def tail_tokens(text: str, n_tokens: int) -> str:
    """Return up to ``n_tokens`` from the tail of ``text``.

    Walks backwards in token-space (encodes the whole string, takes
    the last ``n_tokens`` ids, decodes back to text). Not perfectly
    lossless for multi-byte unicode, but sufficient for cross-chunk
    context propagation — tiktoken's BPE boundaries make this
    idempotent enough for prose.
    """
    if n_tokens <= 0 or not text:
        return ""
    enc = _get_encoding()
    ids = enc.encode(text)  # type: ignore[union-attr]
    if len(ids) <= n_tokens:
        return text
    return enc.decode(ids[-n_tokens:])  # type: ignore[union-attr]
