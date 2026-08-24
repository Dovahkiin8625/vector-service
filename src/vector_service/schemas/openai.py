"""OpenAI-compatible embedding API schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: Literal["float"] = "float"
    user: str | None = None

    @field_validator("input")
    @classmethod
    def _validate_input_not_empty(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input must not be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list must not be empty")
            for i, t in enumerate(v):
                if not isinstance(t, str) or not t.strip():
                    raise ValueError(f"input[{i}] must be non-empty string")
        return v


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "vector-service"
    created: int = 0
    dimensions: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]