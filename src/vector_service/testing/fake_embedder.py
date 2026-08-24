"""Deterministic fake embedder for tests."""
from __future__ import annotations

import hashlib

from vector_service.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    """Deterministic embedder using SHA-256-derived vectors.

    Useful in tests where you don't want to load a real model.
    Same input always yields the same vector.
    """

    def __init__(self, dim: int = 4, model_name: str = "fake-embedder"):
        self.dim = dim
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector_for(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Same space as embed_documents so retrieval end-to-end tests work.
        # Real models (e.g. BGE-M3) apply their own prefixes internally.
        return _vector_for(text, self.dim)


def _vector_for(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Each byte produces 4 float values, expanded from byte array
    out: list[float] = []
    i = 0
    while len(out) < dim:
        b = digest[i % len(digest)]
        # Split into two nibbles to avoid floating point退点
        out.append(((b >> 4) / 15.0) - 0.5)
        out.append(((b & 0xF) / 15.0) - 0.5)
        i += 1
    return out[:dim]
