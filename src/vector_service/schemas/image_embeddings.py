"""Pydantic schemas for ``POST /v1/image_embeddings``."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vector_service.schemas.openai import EmbeddingUsage


class ImageInputItem(BaseModel):
    """A single base64-encoded image inside an ``input`` payload."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"data": "<base64>", "mime": "image/png"}}
    )

    data: str = Field(
        description="Base64-encoded image bytes (no data: URI prefix).",
    )
    mime: str = Field(
        min_length=1,
        description="Image MIME type; must be one of allowed_mime on the server.",
        examples=["image/png", "image/jpeg", "image/webp"],
    )


class ImageEmbeddingRequest(BaseModel):
    """Request body for ``POST /v1/image_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "openclip-vit-l-14",
                "input": {"data": "<base64>", "mime": "image/png"},
            }
        }
    )

    model: str = Field(
        description="Image embedder model id (see ``GET /v1/models`` filtered by type).",
        examples=["openclip-vit-l-14"],
    )
    input: ImageInputItem | list[ImageInputItem] = Field(
        description="One image or a list of images to embed.",
    )
    encoding_format: Literal["float"] = Field(
        default="float",
        description="Encoding format. Only ``float`` is supported today.",
    )
    user: str | None = None


class ImageEmbeddingData(BaseModel):
    """A single embedding result inside ``ImageEmbeddingResponse.data``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
        }
    )

    object: Literal["image_embedding"] = "image_embedding"
    index: int
    embedding: list[float]


class ImageEmbeddingResponse(BaseModel):
    """Response envelope for ``POST /v1/image_embeddings``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456]}
                ],
                "model": "openclip-vit-l-14",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        }
    )

    object: Literal["list"] = "list"
    data: list[ImageEmbeddingData]
    model: str
    usage: EmbeddingUsage
