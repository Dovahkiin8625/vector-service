"""Vector store management API schemas.

Pydantic request / response models for the ``/v1/databases`` and
``/v1/databases/{db}/collections`` families of endpoints.

Collection schema is **fully caller-defined**: callers supply their own
scalar fields, their own vector field, and their own index parameters.
The service does not inject any field (no ``metadata_json``, no default
``id`` name). The only invariants are:

- exactly one VARCHAR primary key field
- at least one FLOAT_VECTOR field
- at least one index covering the vector field
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---- Database ----

class CreateDatabaseRequest(BaseModel):
    """Request body for `POST /v1/databases`."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"name": "tenant-a"}}
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Unique database name (1-64 chars). Acts as an isolation namespace "
            "for collections — every collection lives inside exactly one "
            "database."
        ),
        examples=["tenant-a"],
    )


class DatabaseInfoResponse(BaseModel):
    """Response of `GET /v1/databases/{name}`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "tenant-a", "metadata": {"created_at": "2026-09-10"}}
        }
    )

    name: str = Field(description="Database name.")
    metadata: dict = Field(
        default_factory=dict,
        description="Backend-supplied metadata (free-form; may be empty).",
    )


class DatabaseListResponse(BaseModel):
    """Response of `GET /v1/databases`."""

    databases: list[str] = Field(description="Names of every known database.")


# ---- Collection ----

# Supported scalar types. Mirrors Milvus DataType values we accept for
# non-vector fields. (FLOAT_VECTOR is reserved for the vector field and
# cannot be declared via ``scalar_fields``.)
ScalarDataType = Literal[
    "bool",
    "int8", "int16", "int32", "int64",
    "float", "double",
    "varchar",
    "json",
]


class ScalarFieldSpec(BaseModel):
    """One scalar (non-vector) field in a collection schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "category", "dtype": "varchar", "max_length": 64}
        }
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Field name. Must be unique within the collection.",
        examples=["category"],
    )
    dtype: ScalarDataType = Field(
        description="Milvus scalar data type.",
        examples=["varchar"],
    )
    is_primary: bool = Field(
        default=False,
        description=(
            "Mark this field as the primary key. Exactly one VARCHAR "
            "primary key is required per collection."
        ),
    )
    max_length: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Required for VARCHAR; max stored bytes per value.",
    )
    nullable: bool = Field(
        default=False,
        description="Whether the field accepts NULL values.",
    )
    default_value: Any | None = Field(
        default=None,
        description="Optional default value used when a row omits this field.",
    )


class VectorFieldSpec(BaseModel):
    """The vector field of a collection.

    Exactly one is required per collection. The ``metric_type`` here is
    the **default** metric for the index on this field; it can be
    overridden per-index in ``index_params``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "vector", "dim": 1024, "metric_type": "cosine"}
        }
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Vector field name. Must be unique within the collection.",
        examples=["vector"],
    )
    dim: int = Field(
        ge=1,
        le=32768,
        description="Vector dimensionality (e.g. 1024 for BGE-M3 dense).",
        examples=[1024],
    )
    metric_type: Literal["cosine", "ip", "l2"] = Field(
        default="cosine",
        description="Default distance metric for indexes on this field.",
    )


class IndexParamSpec(BaseModel):
    """Index parameters for one vector field.

    Passed verbatim to pymilvus — see the
    `Milvus index docs <https://milvus.io/docs/index.md>`_ for the full
    matrix. The service does not validate ``index_type`` / ``params``
    beyond JSON-shape.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field_name": "vector",
                "metric_type": "cosine",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
        }
    )

    field_name: str = Field(
        description="Vector field name this index covers.",
    )
    metric_type: Literal["cosine", "ip", "l2"] = Field(
        default="cosine",
        description="Distance metric used at search time.",
    )
    index_type: str = Field(
        default="HNSW",
        description=(
            "Milvus index type, e.g. ``HNSW``, ``IVF_FLAT``, ``IVF_SQ8``, "
            "``DISKANN``, ``FLAT``. Case-sensitive per pymilvus."
        ),
    )
    params: dict = Field(
        default_factory=dict,
        description=(
            "Index-specific knobs forwarded verbatim (e.g. ``{\"M\": 16, "
            "\"efConstruction\": 200}`` for HNSW, ``{\"nlist\": 128}`` for "
            "IVF_FLAT)."
        ),
    )


class CreateCollectionRequest(BaseModel):
    """Request body for `POST /v1/databases/{db}/collections`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "products",
                "primary_field": "id",
                "scalar_fields": [
                    {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
                    {"name": "category", "dtype": "varchar", "max_length": 64},
                    {"name": "price", "dtype": "float"},
                ],
                "vector_field": {"name": "vector", "dim": 1024, "metric_type": "cosine"},
                "index_params": [
                    {"field_name": "vector", "metric_type": "cosine",
                     "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
                ],
            }
        }
    )

    name: str = Field(
        min_length=1,
        max_length=64,
        description="Unique collection name within the parent database.",
        examples=["products"],
    )
    primary_field: str = Field(
        description=(
            "Name of the VARCHAR primary key field. Must appear in "
            "``scalar_fields`` with ``is_primary=true``."
        ),
        examples=["id"],
    )
    scalar_fields: list[ScalarFieldSpec] = Field(
        min_length=1,
        description=(
            "All non-vector fields. Must include the primary key (one "
            "entry with ``is_primary=true``, ``dtype='varchar'``). Other "
            "fields are optional and used as filter targets."
        ),
    )
    vector_field: VectorFieldSpec = Field(
        description="The (single) vector field. ``dim`` is fixed at create time.",
    )
    index_params: list[IndexParamSpec] = Field(
        default_factory=list,
        description=(
            "Indexes to build at create time. At minimum, one index for "
            "the vector field. Defaults to HNSW with cosine if omitted."
        ),
    )

    @model_validator(mode="after")
    def _validate_schema(self):
        names = [f.name for f in self.scalar_fields]
        if len(set(names)) != len(names):
            raise ValueError("scalar field names must be unique")
        if self.vector_field.name in names:
            raise ValueError(
                f"vector field name {self.vector_field.name!r} collides with a scalar field"
            )

        primary = [f for f in self.scalar_fields if f.is_primary]
        if len(primary) != 1:
            raise ValueError("exactly one scalar field must have is_primary=true")
        if primary[0].name != self.primary_field:
            raise ValueError(
                f"primary_field={self.primary_field!r} does not match "
                f"the is_primary field {primary[0].name!r}"
            )
        if primary[0].dtype != "varchar":
            raise ValueError("primary key field must have dtype='varchar'")
        if primary[0].max_length is None or primary[0].max_length < 1:
            raise ValueError("primary key field must set max_length >= 1")

        for f in self.scalar_fields:
            if f.dtype == "varchar" and (f.max_length is None or f.max_length < 1):
                raise ValueError(f"varchar field {f.name!r} must set max_length >= 1")

        vector_field_names = {self.vector_field.name}
        idx_targets = {ip.field_name for ip in self.index_params}
        for ip in self.index_params:
            if ip.field_name not in vector_field_names:
                raise ValueError(
                    f"index_params field_name {ip.field_name!r} is not the vector field"
                )
        # Auto-add a default index if caller forgot — keeps behavior predictable.
        if not self.index_params:
            self.index_params = [
                IndexParamSpec(
                    field_name=self.vector_field.name,
                    metric_type=self.vector_field.metric_type,
                    index_type="HNSW",
                    params={"M": 16, "efConstruction": 200},
                )
            ]
        return self


class FieldSummary(BaseModel):
    """Compact view of one field, used by ``GET .../collections/{name}``."""

    name: str
    dtype: str
    is_primary: bool = False
    dim: int | None = None


class CollectionInfoResponse(BaseModel):
    """Response of `GET /v1/databases/{db}/collections/{name}`."""

    database: str = Field(description="Parent database name.")
    name: str = Field(description="Collection name.")
    dim: int = Field(description="Vector dimensionality of the vector field.")
    metric: str = Field(description="Distance metric of the primary index.")
    count: int = Field(description="Number of vectors currently stored.")
    primary_field: str = Field(description="Name of the primary key field.")
    vector_field: str = Field(description="Name of the vector field.")
    fields: list[FieldSummary] = Field(
        description="All fields in the collection (scalar + vector)."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Backend-supplied metadata (free-form; may be empty).",
    )


# ---- Vectors ----

class UpsertVectorsRequest(BaseModel):
    """Request body for `PUT /v1/databases/{db}/collections/{name}/vectors`.

    Per-row scalar data goes in ``fields``: a list aligned with ``ids``
    where each entry is a ``{field_name: value}`` dict. The service
    splits ``vectors`` into the vector column and the rest into the
    scalar columns.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_field": "id",
                "vector_field": "vector",
                "ids": ["sku-1", "sku-2"],
                "texts": ["a wireless mouse", "a mechanical keyboard"],
                "fields": [
                    {"category": "mouse", "price": 29.9},
                    {"category": "keyboard", "price": 119.0},
                ],
            }
        }
    )

    primary_field: str = Field(
        description="Primary key field name (must match collection schema).",
    )
    vector_field: str = Field(
        description="Vector field name (must match collection schema).",
    )
    ids: list[str] = Field(
        min_length=1,
        description=(
            "Primary key values. Length must equal ``len(texts)`` or "
            "``len(vectors)`` and ``len(fields)`` (if provided)."
        ),
    )
    texts: list[str] | None = Field(
        default=None,
        description="Raw text inputs to embed server-side. Mutually exclusive with ``vectors``.",
    )
    vectors: list[list[float]] | None = Field(
        default=None,
        description="Pre-computed vectors. Each inner list must have length equal to ``vector_field.dim``.",
    )
    images: list[str] | None = Field(
        default=None,
        description="Base64-encoded image bytes (parallel to ids).",
    )
    image_mimes: list[str] | None = Field(
        default=None,
        description="MIME per image (parallel to ids). Required when images is provided.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Embedder model id. Required when images is provided (selects the "
            "image embedder). Optional otherwise (uses the default text embedder)."
        ),
    )
    fields: list[dict] | None = Field(
        default=None,
        description=(
            "Per-row scalar field values aligned with ``ids``. Each entry "
            "is a ``{field_name: value}`` map. Keys must be valid scalar "
            "field names declared at create time (excluding the primary "
            "key, which goes in ``ids``)."
        ),
    )

    @model_validator(mode="after")
    def _xor_three_way(self):
        has_t = self.texts is not None
        has_v = self.vectors is not None
        has_i = self.images is not None
        if (has_t + has_v + has_i) != 1:
            raise ValueError("provide exactly one of texts, vectors, images")
        n = len(self.ids)
        if has_t and len(self.texts) != n:  # type: ignore[arg-type]
            raise ValueError("ids and texts must have same length")
        if has_v and len(self.vectors) != n:  # type: ignore[arg-type]
            raise ValueError("ids and vectors must have same length")
        if has_i:
            if self.image_mimes is None or len(self.image_mimes) != n:
                raise ValueError("image_mimes required, same length as ids")
            if self.model is None:
                raise ValueError("model is required when images is provided")
        if self.fields is not None and len(self.fields) != n:
            raise ValueError("ids and fields must have same length")
        return self


class DeleteVectorsRequest(BaseModel):
    """Request body for `POST .../vectors/delete`."""

    primary_field: str = Field(description="Primary key field name.")
    ids: list[str] = Field(
        min_length=1,
        description="Identifiers of vectors to delete.",
    )


class GetVectorsRequest(BaseModel):
    """Request body for `POST .../vectors/get`."""

    primary_field: str = Field(description="Primary key field name.")
    ids: list[str] = Field(
        min_length=1,
        description="Identifiers of vectors to fetch.",
    )


class GetVectorItem(BaseModel):
    """Single item returned by `POST .../vectors/get`."""

    id: str = Field(description="Primary key value.")
    vector: list[float] | None = Field(
        default=None,
        description="Raw vector if requested and available; otherwise ``null``.",
    )
    fields: dict = Field(
        default_factory=dict,
        description="Per-row scalar field values returned by Milvus.",
    )


class GetVectorsResponse(BaseModel):
    """Response of `POST .../vectors/get`."""

    items: list[GetVectorItem]


# ---- Search ----

class SearchRequest(BaseModel):
    """Request body for `POST .../search`."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_field": "id",
                "vector_field": "vector",
                "query_text": "computer accessory",
                "top_k": 5,
                "filter_expr": "category == 'mouse'",
            }
        }
    )

    primary_field: str = Field(description="Primary key field name.")
    vector_field: str = Field(description="Vector field name to search against.")
    query_text: str | None = Field(
        default=None,
        description="Free-form query. Embedded server-side. Mutually exclusive with ``query_vector``.",
    )
    query_vector: list[float] | None = Field(
        default=None,
        description="Pre-computed query vector. Length must equal ``vector_field.dim``.",
    )
    query_image: str | None = Field(
        default=None,
        description="Base64-encoded query image. Mutually exclusive with query_text / query_vector.",
    )
    query_image_mime: str | None = Field(
        default=None,
        description="MIME for query_image. Required when query_image is provided.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Embedder model id. Required when query_image is provided "
            "(selects the image embedder); optional for text/vector queries."
        ),
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of nearest neighbours to return.",
    )
    filter_expr: str | None = Field(
        default=None,
        description=(
            "Optional Milvus-native boolean expression. Forwarded verbatim "
            "to pymilvus — e.g. ``category == 'mouse' and price < 100``. "
            "Field names must match scalar fields declared at create time."
        ),
    )
    output_fields: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of scalar field names to return in each hit. "
            "Defaults to the primary key only."
        ),
    )

    @model_validator(mode="after")
    def _xor_three_way(self):
        has_t = self.query_text is not None
        has_v = self.query_vector is not None
        has_i = self.query_image is not None
        if (has_t + has_v + has_i) != 1:
            raise ValueError("provide exactly one of query_text, query_vector, query_image")
        if has_i:
            if self.query_image_mime is None:
                raise ValueError("query_image_mime required when query_image is provided")
            if self.model is None:
                raise ValueError("model is required when query_image is provided")
        return self


class HitResponse(BaseModel):
    """A single search hit."""

    id: str = Field(description="Primary key value of the matched vector.")
    score: float = Field(
        description=(
            "Similarity / distance score. Higher is better for ``cosine`` "
            "and ``ip``; lower is better for ``l2``."
        ),
    )
    fields: dict = Field(
        default_factory=dict,
        description="Per-row scalar field values requested via ``output_fields``.",
    )


class SearchResponse(BaseModel):
    """Response of `POST .../search`."""

    hits: list[HitResponse] = Field(
        description="Top-k hits sorted by score (best first)."
    )


# ---- Backend escape hatch ----

class BackendInfo(BaseModel):
    """Diagnostic info returned by `GET /backend/raw` (debug builds)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "milvus",
                "info": {"uri": "http://localhost:19530"},
            }
        }
    )

    backend: str = Field(description="Active vector-store backend name.")
    info: dict = Field(
        description=(
            "Backend-specific introspection (connection URI, etc.). Shape "
            "is backend-defined; intended for operators / debugging."
        )
    )
