"""Image embedder abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageInput:
    """A single image ready to embed.

    ``data`` is raw decoded bytes (NOT base64). Per-model preprocessing
    (PIL decode, resize, normalize) happens inside the embedder.
    """

    data: bytes
    mime: str


class ImageEmbedder(ABC):
    """Abstract base for image embedding models.

    Concrete subclasses must:
    - Set ``dim`` (int) and ``model_name`` (str).
    - Implement ``load``, ``embed_images``, and ``embed_query_image``.

    All methods are synchronous. Async dispatch is the caller's job
    (use ``loop.run_in_executor`` from FastAPI routes).
    """

    dim: int
    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Eagerly load the model into memory and run any one-time warmup.

        Called once during application startup (see ``core.lifespan``).
        Subclasses that lazily defer heavy work to first use must
        implement this so the load happens up front and readiness
        probes can observe it. May raise ``ModelNotLoadedForImages``.
        """

    def unload(self) -> None:
        """Release the loaded model and any associated resources.

        Mirrors ``Embedder.unload``: idempotent, default no-op.
        Subclasses that hold native resources (GPU weights, native
        handles) should override to release them.
        """

    @abstractmethod
    def embed_images(self, images: list[ImageInput]) -> list[list[float]]:
        """Embed a batch of images. Return one vector per input, in order."""

    @abstractmethod
    def embed_query_image(self, image: ImageInput) -> list[float]:
        """Embed a single query image. May differ from ``embed_images``.

        For models without a distinct query mode (e.g. OpenCLIP), this
        is identical to ``embed_images([image])[0]``.
        """
