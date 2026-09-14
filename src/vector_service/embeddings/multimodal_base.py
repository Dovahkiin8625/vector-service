"""Multimodal embedder abstract base class.

A ``MultimodalEmbedder`` produces vectors for **both** text and image
inputs, and — unlike the separate ``Embedder`` / ``ImageEmbedder``
hierarchies — both towers must agree on ``dim`` so cross-modal similarity
(text↔image) is meaningful. Concrete examples: Chinese-CLIP, BLIP,
CoCa. The text and image encoders typically share weights inside a
single model and are loaded once.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from vector_service.embeddings.image_base import ImageInput


class MultimodalEmbedder(ABC):
    """Abstract base for cross-modal embedding models.

    Concrete subclasses must:
    - Set ``dim`` (int) and ``model_name`` (str).
    - Implement ``load``, ``embed_text``, and ``embed_images``.

    ``embed_text`` and ``embed_images`` must produce vectors in the same
    ``dim``-dimensional space so cosine similarity between text and image
    vectors is meaningful. All methods are synchronous; async dispatch
    is the caller's job (``loop.run_in_executor``).
    """

    dim: int
    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Eagerly load the model into memory and run any one-time warmup.

        Called once during application startup (see ``core.lifespan``).
        Subclasses that defer heavy work to first use must implement
        this so load happens up front and readiness probes can observe
        it. May raise ``ModelNotLoadedForImages`` (reused to keep the
        fail-open path uniform across all sub-models).
        """

    def unload(self) -> None:
        """Release the loaded model and any associated resources.

        Mirrors ``Embedder.unload``: idempotent, default no-op.
        Multimodal models typically hold a single underlying
        network that services both text and image towers — the
        default implementation leaves those alone; subclasses with
        native resources should override.
        """

    @abstractmethod
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Return one vector per input, in order."""

    @abstractmethod
    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        """Embed a batch of images. Return one vector per input, in order.

        Must produce vectors in the same ``dim``-dimensional space as
        ``embed_text``.
        """