"""OpenAI-compatible embedding API schemas.

Pydantic models for the ``/v1/embeddings`` and ``/v1/models`` endpoints.
``Field`` descriptions and ``model_config`` examples feed straight into the
generated OpenAPI schema so Swagger UI / ReDoc render a useful reference.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request body."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": "The quick brown fox jumps over the lazy dog",
                "model": "bge-m3",
            }
        }
    )

    input: str | list[str] = Field(
        description=(
            "Input text to embed. Either a single string or an array of "
            "strings. Each entry must be non-empty and at most "
            "`VS_EMBEDDING_MAX_CHARS_PER_TEXT` characters. The total number "
            "of inputs in one request is capped by "
            "`VS_EMBEDDING_MAX_TEXTS_PER_REQUEST`."
        ),
        examples=[
            "hello world",
            ["first document", "second document", "third document"],
        ],
    )
    model: str = Field(
        description=(
            "Embedder model id to use (e.g. `bge-m3`). The id must be "
            "registered in the embedder registry; see `GET /v1/models`."
        ),
        examples=["bge-m3"],
    )
    encoding_format: Literal["float"] = Field(
        default="float",
        description=(
            "Encoding format for the returned vectors. Only `float` is "
            "supported today; the field is kept for OpenAI compatibility."
        ),
    )
    user: str | None = Field(
        default=None,
        description=(
            "Optional opaque identifier for the end-user making the request. "
            "Useful for usage tracking / abuse detection. Not logged by "
            "default."
        ),
    )

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
    """A single embedding result inside an `EmbeddingResponse.data` array."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "embedding",
                "index": 0,
                "embedding": [0.0123, -0.0456, 0.0789, 0.1011],
            }
        }
    )

    object: Literal["embedding"] = "embedding"
    index: int = Field(description="0-based position of this input in the request.")
    embedding: list[float] = Field(
        description="Dense embedding vector. Length equals the model's `dimensions`."
    )


class EmbeddingUsage(BaseModel):
    """Token accounting block — currently a rough char-based estimate."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"prompt_tokens": 11, "total_tokens": 11}
        }
    )

    prompt_tokens: int = Field(
        description=(
            "Estimated input token count (ceil of `len(text) / 4` per "
            "input, summed across the request)."
        )
    )
    total_tokens: int = Field(description="Same as `prompt_tokens`; reserved for future use.")


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response envelope."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.0123, -0.0456, 0.0789, 0.1011],
                    }
                ],
                "model": "bge-m3",
                "usage": {"prompt_tokens": 11, "total_tokens": 11},
            }
        }
    )

    object: Literal["list"] = "list"
    data: list[EmbeddingData] = Field(description="One entry per input, in input order.")
    model: str = Field(description="Echoes the `model` field from the request.")
    usage: EmbeddingUsage


class ModelInfo(BaseModel):
    """Runtime details of a *currently loaded* model instance.

    Populated on :class:`Model` rows (and single-model lookups) only
    while the instance is actually held by the process; ``model_info``
    stays ``null`` for registered-but-unloaded ids. Every field is
    best-effort: backends that don't expose a discoverable
    ``torch.nn.Module`` (test doubles, future non-torch backends)
    report ``null`` for the resource fields, while ``load_duration_seconds``
    is still recorded by the lifecycle slot.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device": "cuda",
                "dtype": "float16",
                "param_count": 568_000_000,
                "memory_bytes": 1_136_000_000,
                "load_duration_seconds": 12.4,
            }
        }
    )

    device: str | None = Field(
        default=None,
        description=(
            "Device the loaded instance runs inference on (``cuda`` / "
            "``cpu``, possibly indexed e.g. ``cuda:0``)."
        ),
    )
    dtype: str | None = Field(
        default=None,
        description=(
            "Dominant parameter dtype (e.g. ``float16``, ``float32``, "
            "``bfloat16``)."
        ),
    )
    param_count: int | None = Field(
        default=None,
        description=(
            "Total number of unique parameters across every discoverable "
            "``torch.nn.Module`` of the instance (shared/child tensors "
            "counted once)."
        ),
    )
    memory_bytes: int | None = Field(
        default=None,
        description=(
            "Weight footprint in bytes: unique parameters plus buffers. "
            "For a GPU model this is the deterministic VRAM floor; it "
            "excludes the CUDA context, allocator reserves, and "
            "inference activations. For a CPU model it is the resident "
            "weight cost in RAM."
        ),
    )
    load_duration_seconds: float | None = Field(
        default=None,
        description=(
            "Wall-clock seconds the last successful build + ``load()`` "
            "(including any download/warmup done inside it) took. ``null`` "
            "for an instance injected before the slot measured it."
        ),
    )


class Model(BaseModel):
    """A registered model descriptor.

    Covers both embedders and rerankers under a single shape so that
    `GET /v1/models` can return every registered backend. ``type``
    discriminates the two families and ``dimensions`` is meaningful
    only for embedders (always ``None`` for rerankers).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "bge-m3",
                "object": "model",
                "type": "embedder",
                "owned_by": "vector-service",
                "created": 0,
                "dimensions": 1024,
                "loaded": True,
                "load_status": "loaded",
                "load_error": None,
                "model_info": {
                    "device": "cuda",
                    "dtype": "float16",
                    "param_count": 568000000,
                    "memory_bytes": 1136000000,
                    "load_duration_seconds": 12.4,
                },
            }
        }
    )

    id: str = Field(description="Model identifier.")
    object: Literal["model"] = "model"
    type: Literal["embedder", "reranker", "image_embedder", "multimodal_embedder"] = Field(
        default="embedder",
        description=(
            "Backend family. `embedder` for text embedding models (carries a "
            "`dimensions` value once loaded); `reranker` for cross-encoder "
            "rerankers (always `dimensions=null`); `image_embedder` for image "
            "embedding models (carries `dimensions` once loaded, mirroring "
            "the text embedder); `multimodal_embedder` for cross-modal "
            "text+image models (carries `dimensions` once loaded — vectors "
            "from both towers share the same space, e.g. 512d for Chinese-CLIP)."
        ),
    )
    owned_by: str = Field(
        default="vector-service",
        description="Provider that owns / ships the model.",
    )
    created: int = Field(
        default=0,
        description="Unix timestamp (seconds) at which the model was registered. 0 if unknown.",
    )
    dimensions: int | None = Field(
        default=None,
        description=(
            "Embedding dimensionality. `null` for rerankers and for "
            "embedders (text or image) not currently loaded; populated "
            "once the corresponding embedder is initialized at startup."
        ),
    )
    loaded: bool = Field(
        default=False,
        description=(
            "`true` when an instance of this model is currently held on "
            "the running process (``app.state.<family>``) and ready to "
            "serve inference. Always populated for every model type — "
            "for rerankers, where ``dimensions`` is permanently `null`, "
            "this is the only signal that the backend has the model "
            "loaded. The dashboard's 已加载 / 卸载 buttons are keyed off "
            "this field."
        ),
    )
    load_status: Literal["unloaded", "loading", "loaded", "failed"] = Field(
        default="unloaded",
        description=(
            "Fine-grained lifecycle state of this model id: `loading` "
            "while a background `POST .../load` is constructing/warming "
            "the instance, `loaded` once it is held on the process, "
            "`failed` if the last background load raised (see "
            "`load_error`), `unloaded` otherwise. Poll this field after "
            "receiving HTTP 202 `loading` from the load endpoint. Only "
            "the targeted id reports `loading`/`failed`; other ids of "
            "the same family stay `unloaded`."
        ),
    )
    load_error: str | None = Field(
        default=None,
        description=(
            "Error message from the last failed background load for this "
            "model id, or `null`. Cleared on the next successful load or "
            "unload."
        ),
    )
    model_info: ModelInfo | None = Field(
        default=None,
        description=(
            "Runtime details of the live instance (device, dtype, "
            "parameter count, weight footprint in bytes, load duration) "
            "while this id is loaded; `null` for unloaded / loading / "
            "failed rows."
        ),
    )


class ModelList(BaseModel):
    """Response of `GET /v1/models`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "object": "list",
                "data": [
                    {
                        "id": "bge-m3",
                        "object": "model",
                        "type": "embedder",
                        "owned_by": "vector-service",
                        "created": 0,
                        "dimensions": 1024,
                        "loaded": True,
                        "load_status": "loaded",
                        "load_error": None,
                    },
                    {
                        "id": "bge-reranker-v2-m3",
                        "object": "model",
                        "type": "reranker",
                        "owned_by": "vector-service",
                        "created": 0,
                        "dimensions": None,
                        "loaded": False,
                        "load_status": "unloaded",
                        "load_error": None,
                    },
                ],
            }
        }
    )

    object: Literal["list"] = "list"
    data: list[Model]


# ---- Hot load / unload --------------------------------------------------
#
# Both endpoints share the same shape so clients can dispatch on
# ``status`` alone. ``dimensions`` mirrors the field on :class:`Model`:
# ``None`` for rerankers, populated once the corresponding embedder is
# actually loaded. A load request normally returns HTTP 202 with
# ``status="loading"`` — the work continues in the background and the
# client polls ``GET /v1/models`` for ``load_status``. An idempotent
# re-load of an already-held id returns HTTP 200 with ``status="loaded"``
# immediately.


class ModelLoadResponse(BaseModel):
    """Response of `POST /v1/models/{model_id}/load`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "bge-m3",
                    "type": "embedder",
                    "status": "loading",
                    "dimensions": None,
                },
                {
                    "id": "bge-m3",
                    "type": "embedder",
                    "status": "loaded",
                    "dimensions": 1024,
                },
            ]
        }
    )

    id: str = Field(description="Model id whose load was requested (or is already loaded).")
    type: Literal["embedder", "reranker", "image_embedder", "multimodal_embedder"] = Field(
        description="Model family the loaded instance belongs to."
    )
    status: Literal["loading", "loaded"] = Field(
        description=(
            "`loading` (HTTP 202): the load is running in the background — "
            "poll `GET /v1/models` and watch the row's `load_status` for "
            "`loaded` / `failed` (+ `load_error`). `loaded` (HTTP 200): the "
            "id was already held on the slot (idempotent re-load)."
        )
    )
    dimensions: int | None = Field(
        default=None,
        description=(
            "Embedding dimensionality once the model is initialised; "
            "`null` while loading and for rerankers. Idempotent re-loads "
            "of the same id return the existing dimensions."
        ),
    )


class ModelUnloadResponse(BaseModel):
    """Response of `POST /v1/models/{model_id}/unload`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "bge-m3",
                "type": "embedder",
                "status": "unloaded",
            }
        }
    )

    id: str = Field(description="Model id that was unloaded.")
    type: Literal["embedder", "reranker", "image_embedder", "multimodal_embedder"] = Field(
        description="Model family the unloaded instance belonged to."
    )
    status: Literal["unloaded"] = "unloaded"