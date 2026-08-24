"""Vector store management API schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---- Collection ----

class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    dim: int | None = None
    metric: Literal["cosine", "ip", "l2"] = "cosine"
    backend_opts: dict = Field(default_factory=dict)


class CollectionInfoResponse(BaseModel):
    name: str
    dim: int
    metric: str
    count: int


# ---- Vectors ----

class UpsertVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)
    texts: list[str] | None = None
    embeddings: list[list[float]] | None = None
    metadatas: list[dict] | None = None

    @model_validator(mode="after")
    def _xor_texts_embeddings(self):
        has_t = self.texts is not None
        has_e = self.embeddings is not None
        if has_t and has_e:
            raise ValueError("provide either texts or embeddings, not both")
        if not has_t and not has_e:
            raise ValueError("must provide either texts or embeddings")
        n = len(self.ids)
        if has_t and len(self.texts) != n:  # type: ignore[arg-type]
            raise ValueError("ids and texts must have same length")
        if has_e and len(self.embeddings) != n:  # type: ignore[arg-type]
            raise ValueError("ids and embeddings must have same length")
        if self.metadatas is not None and len(self.metadatas) != n:
            raise ValueError("ids and metadatas must have same length")
        return self


class DeleteVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class GetVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class GetVectorItem(BaseModel):
    id: str
    vector: list[float] | None = None
    metadata: dict


class GetVectorsResponse(BaseModel):
    items: list[GetVectorItem]


# ---- Search ----

class SearchRequest(BaseModel):
    query_text: str | None = None
    query_embedding: list[float] | None = None
    top_k: int = Field(10, ge=1, le=1000)
    filter: dict | None = None

    @model_validator(mode="after")
    def _xor_text_embedding(self):
        has_t = self.query_text is not None
        has_e = self.query_embedding is not None
        if has_t and has_e:
            raise ValueError("provide either query_text or query_embedding, not both")
        if not has_t and not has_e:
            raise ValueError("must provide either query_text or query_embedding")
        return self


class HitResponse(BaseModel):
    id: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    hits: list[HitResponse]


# ---- Backend escape hatch ----

class BackendInfo(BaseModel):
    backend: str
    info: dict