"""Pydantic schemas for ``POST /v1/multimodal_embeddings``.

Each item in ``input`` is either ``{"text": "..."}`` or
``{"image": {"data": "<base64>", "mime": "..."}}`` — exactly one of the
two. Empty ``input`` is rejected by the route (422). The endpoint
batches by modality internally so each encoder is called once.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vector_service.schemas.openai import EmbeddingUsage


class MultimodalImagePayload(BaseModel):
    """A single base64-encoded image inside a multimodal input item."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"data": "<base64>", "mime": "image/png"}}
    )

    data: str = Field(description="Base64-encoded image bytes (no data: URI prefix).")
    mime: str = Field(
        min_length=1,
        description="Image MIME type; must be one of allowed_mime on the server.",
        examples=["image/png", "image/jpeg", "image/webp"],
    )


class MultimodalInputItem(BaseModel):
    """A single item — either text or image, never both, never neither."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "一只猫"}}
    )

    text: str | None = Field(
        default=None,
        description="Chinese text to embed via the text tower.",
    )
    image: MultimodalImagePayload | None = Field(
        default=None,
        description="Image to embed via the image tower.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "MultimodalInputItem":
        has_text = self.text is not None
        has_image = self.image is not None
        if has_text == has_image:
            # Both or neither: invalid.
            raise ValueError(
                "each input item must contain exactly one of 'text' or 'image'"
            )
        return self


class MultimodalEmbeddingRequest(BaseModel):
    """Request body for ``POST /v1/multimodal_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "chinese-clip-vit-base-patch16",
                "input": [
                    {"text": "一只猫"},
                    {"image": {"data": "<base64>", "mime": "image/png"}},
                ],
            }
        }
    )

    model: str = Field(
        description="Multimodal embedder model id (see ``GET /v1/models`` filtered by type=multimodal_embedder).",
        examples=["chinese-clip-vit-base-patch16"],
    )
    input: MultimodalInputItem | list[MultimodalInputItem] = Field(
        description="One item or a list of items; each item is ``{text:...}`` or ``{image:{data,mime}}``.",
    )
    encoding_format: Literal["float"] = Field(
        default="float",
        description="Encoding format. Only ``float`` is supported today.",
    )
    user: str | None = None


class MultimodalEmbeddingData(BaseModel):
    """A single embedding result inside ``MultimodalEmbeddingResponse.data``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"object": "multimodal_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
        }
    )

    object: Literal["multimodal_embedding"] = "multimodal_embedding"
    index: int
    embedding: list[float]


class MultimodalEmbeddingResponse(BaseModel):
    """Response envelope for ``POST /v1/multimodal_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {"object": "multimodal_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
                ],
                "model": "chinese-clip-vit-base-patch16",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        }
    )

    object: Literal["list"] = "list"
    data: list[MultimodalEmbeddingData]
    model: str
    usage: EmbeddingUsage