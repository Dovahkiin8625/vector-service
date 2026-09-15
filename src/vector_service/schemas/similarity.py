"""Request / response schemas for the similarity endpoints.

All three endpoints share the same response envelope
(``SimilarityResponse``) — the only difference is the shape of
``query`` / ``documents`` per family. Keeping a single envelope keeps the
dashboard renderer simple (one ``result-row`` layout, one set of JS
handlers) and lets the client code treat all three endpoints uniformly.

The metric literal mirrors the one in ``schemas/management.py`` so the
similarity score a caller sees here matches what the vector store returns
for the same vectors.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vector_service.embeddings.similarity import validate_metric  # noqa: F401  (kept for callers that import from schema)
from vector_service.schemas.multimodal_embeddings import MultimodalImagePayload


# Re-export so route code can validate against the same set without reaching
# into the implementation module.
SimilarityMetric = Literal["cosine", "ip", "l2"]


# ---------------------------------------------------------------------------
# Request shapes
# ---------------------------------------------------------------------------


class TextSimilarityRequest(BaseModel):
    """Score a query string against N candidate documents using a text embedder."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "bge-m3",
                "query": "无线鼠标",
                "documents": ["蓝牙鼠标", "机械键盘"],
                "metric": "cosine",
            }
        }
    )

    model: str = Field(..., description="Registered text-embedder model id.")
    query: str = Field(..., min_length=1, description="Reference text to score candidates against.")
    documents: list[str] = Field(
        ..., min_length=1, description="Candidate texts to score; order is preserved in results."
    )
    metric: SimilarityMetric = Field(
        default="cosine",
        description="Distance / similarity metric. Cosine and ip sort descending; l2 sorts ascending.",
    )

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("documents")
    @classmethod
    def _documents_not_empty(cls, v: list[str]) -> list[str]:
        for i, d in enumerate(v):
            if not isinstance(d, str) or not d.strip():
                raise ValueError(f"documents[{i}] must be non-empty string")
        return v


class ImageSimilarityItem(BaseModel):
    """One base64-encoded image used as either query or candidate."""

    data: str = Field(description="Base64-encoded image bytes (no data: URI prefix).")
    mime: Annotated[str, Field(min_length=1, examples=["image/png", "image/jpeg", "image/webp"])]


class ImageSimilarityRequest(BaseModel):
    """Score a query image against N candidate images using an image embedder."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "openclip-vit-l-14",
                "query": {"data": "<base64>", "mime": "image/png"},
                "documents": [
                    {"data": "<base64>", "mime": "image/png"},
                    {"data": "<base64>", "mime": "image/jpeg"},
                ],
                "metric": "cosine",
            }
        }
    )

    model: str = Field(..., description="Registered image-embedder model id.")
    query: ImageSimilarityItem = Field(..., description="Reference image to score candidates against.")
    documents: list[ImageSimilarityItem] = Field(
        ..., min_length=1, description="Candidate images; order is preserved in results."
    )
    metric: SimilarityMetric = Field(default="cosine", description="Distance / similarity metric.")


class MultimodalSimilarityItem(BaseModel):
    """Either a Chinese text or a base64 image — mirrors the multimodal embedding
    request shape (``text`` xor ``image``)."""

    text: str | None = Field(default=None, description="Chinese text (text-tower input).")
    image: MultimodalImagePayload | None = Field(default=None, description="Image payload (image-tower input).")

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("text must not be empty when provided")
        return v


class MultimodalSimilarityRequest(BaseModel):
    """Score a query against N candidates where each side can be text or image.

    Uses the multimodal embedder's shared space (Chinese-CLIP etc.) —
    cosine similarity is meaningful across modalities.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "chinese-clip-vit-base-patch16",
                "query": {"text": "一只猫"},
                "documents": [
                    {"text": "狗"},
                    {"image": {"data": "<base64>", "mime": "image/png"}},
                ],
                "metric": "cosine",
            }
        }
    )

    model: str = Field(..., description="Registered multimodal-embedder model id.")
    query: MultimodalSimilarityItem = Field(..., description="Reference item (text xor image).")
    documents: list[MultimodalSimilarityItem] = Field(
        ..., min_length=1, description="Candidate items; each is text xor image."
    )
    metric: SimilarityMetric = Field(default="cosine", description="Distance / similarity metric.")

    @field_validator("documents")
    @classmethod
    def _docs_well_formed(cls, v: list[MultimodalSimilarityItem]) -> list[MultimodalSimilarityItem]:
        for i, item in enumerate(v):
            if (item.text is None) == (item.image is None):
                raise ValueError(
                    f"documents[{i}] must contain exactly one of 'text' or 'image'"
                )
        return v

    @field_validator("query")
    @classmethod
    def _query_well_formed(cls, v: MultimodalSimilarityItem) -> MultimodalSimilarityItem:
        if (v.text is None) == (v.image is None):
            raise ValueError("query must contain exactly one of 'text' or 'image'")
        return v


# ---------------------------------------------------------------------------
# Response shapes — shared
# ---------------------------------------------------------------------------


class SimilarityResultItem(BaseModel):
    """One scored candidate.

    ``score`` semantics depend on ``metric``: cosine/ip higher = more similar;
    l2 lower = more similar. The route sorts so the most-similar item is
    always ``index=0`` of the response ``results`` list.
    """

    index: int = Field(..., description="0-based index into the original input list.")
    score: float = Field(..., description="Similarity (cosine/ip) or distance (l2).")


class SimilarityResponse(BaseModel):
    """Common response shape for all three similarity endpoints."""

    model: str = Field(..., description="Embedder id used.")
    metric: SimilarityMetric = Field(..., description="Metric used to score.")
    results: list[SimilarityResultItem] = Field(
        ...,
        description="Scored candidates, sorted most-similar-first (cosine/ip) or closest-first (l2).",
    )
    request_id: str | None = Field(default=None, description="Server-assigned request id.")