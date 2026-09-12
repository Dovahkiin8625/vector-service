"""Image embedding request/response schemas."""
from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from vector_service.schemas.image_embeddings import (
    ImageEmbeddingData,
    ImageEmbeddingRequest,
    ImageEmbeddingResponse,
    ImageInputItem,
)


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


def test_image_input_item_accepts_base64_and_mime():
    item = ImageInputItem(data=VALID_PNG_B64, mime="image/png")
    assert item.data == VALID_PNG_B64
    assert item.mime == "image/png"


def test_image_input_item_requires_mime():
    with pytest.raises(ValidationError):
        ImageInputItem(data=VALID_PNG_B64, mime="")  # type: ignore[arg-type]


def test_request_single_image():
    req = ImageEmbeddingRequest(model="openclip-vit-l-14", input=ImageInputItem(data=VALID_PNG_B64, mime="image/png"))
    assert req.model == "openclip-vit-l-14"
    assert isinstance(req.input, ImageInputItem)


def test_request_list_of_images():
    items = [ImageInputItem(data=VALID_PNG_B64, mime="image/png") for _ in range(3)]
    req = ImageEmbeddingRequest(model="openclip-vit-l-14", input=items)
    assert isinstance(req.input, list)
    assert len(req.input) == 3


def test_request_requires_model():
    with pytest.raises(ValidationError):
        ImageEmbeddingRequest(input=ImageInputItem(data=VALID_PNG_B64, mime="image/png"))  # type: ignore[call-arg]


def test_data_object_literal():
    d = ImageEmbeddingData(index=0, embedding=[0.1] * 768)
    assert d.object == "image_embedding"
    assert d.index == 0
    assert len(d.embedding) == 768


def test_response_envelope_shape():
    resp = ImageEmbeddingResponse(
        data=[ImageEmbeddingData(index=0, embedding=[0.0] * 768)],
        model="openclip-vit-l-14",
        usage={"prompt_tokens": 1, "total_tokens": 1},  # type: ignore[arg-type]
    )
    assert resp.object == "list"
    assert resp.model == "openclip-vit-l-14"
    assert resp.usage.prompt_tokens == 1
