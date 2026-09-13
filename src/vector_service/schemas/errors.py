"""Shared error envelope schema for the HTTP API.

Every non-2xx response on the vector-service uses the canonical shape:

    {
      "error": {
        "code": "<machine-readable code>",
        "message": "<human-readable message>",
        "request_id": "<uuid>",
        "extra": { ...arbitrary structured fields... }
      }
    }

This module declares the corresponding Pydantic models so route handlers can
reference them via ``responses={404: ErrorEnvelope, 422: ErrorEnvelope, ...}``
and the generated OpenAPI / Swagger UI shows a single, consistent error shape
across endpoints.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorBody(BaseModel):
    """Inner error object — what every endpoint places in ``error``."""

    code: str = Field(
        description=(
            "Machine-readable error code. Stable identifiers such as "
            "`model_not_found`, `database_not_found`, `database_exists`, "
            "`collection_not_found`, `collection_exists`, "
            "`dimension_mismatch`, `too_many_texts`, `text_too_long`, "
            "`embedder_unavailable`, `store_unavailable`, `shape_mismatch`, "
            "`invalid_request`, `internal`."
        ),
        examples=["collection_not_found"],
    )
    message: str = Field(
        description="Human-readable error message describing what went wrong.",
        examples=["collection 'products' does not exist"],
    )
    request_id: str = Field(
        description=(
            "Server-side request correlation id. Matches the `X-Request-ID` "
            "response header so operators can grep server logs."
        ),
        examples=["8f4e1c2a-9b1d-4f0e-9c1a-2b3c4d5e6f70"],
    )
    extra: dict = Field(
        default_factory=dict,
        description=(
            "Endpoint-specific structured details (e.g. `expected` vs `got` "
            "for dimension mismatches, `index` for text_too_long, `errors` "
            "for global validation failures)."
        ),
        examples=[{}],
    )


class ErrorEnvelope(BaseModel):
    """Top-level error envelope returned by all non-2xx responses."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "collection_not_found",
                    "message": "collection 'products' does not exist",
                    "request_id": "8f4e1c2a-9b1d-4f0e-9c1a-2b3c4d5e6f70",
                    "extra": {"name": "products"},
                }
            }
        }
    )

    error: ErrorBody