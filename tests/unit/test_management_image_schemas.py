"""Extended UpsertVectorsRequest + SearchRequest accept image inputs."""
from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from vector_service.schemas.management import SearchRequest, UpsertVectorsRequest


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


# ---- upsert ----


def test_upsert_accepts_images():
    req = UpsertVectorsRequest(
        primary_field="id",
        vector_field="vector",
        ids=["a", "b"],
        images=[VALID_PNG_B64, VALID_PNG_B64],
        image_mimes=["image/png", "image/png"],
        model="openclip-vit-l-14",
    )
    assert req.images is not None and len(req.images) == 2


def test_upsert_requires_image_mimes_when_images_present():
    with pytest.raises(ValidationError) as ei:
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            images=[VALID_PNG_B64],
            model="openclip-vit-l-14",
        )
    assert "image_mimes" in str(ei.value)


def test_upsert_requires_model_when_images_present():
    with pytest.raises(ValidationError) as ei:
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            images=[VALID_PNG_B64],
            image_mimes=["image/png"],
        )
    assert "model" in str(ei.value)


def test_upsert_rejects_texts_plus_images():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a"],
            texts=["hello"],
            images=[VALID_PNG_B64],
            image_mimes=["image/png"],
            model="openclip-vit-l-14",
        )


def test_upsert_image_mime_length_mismatch():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(
            primary_field="id",
            vector_field="vector",
            ids=["a", "b"],
            images=[VALID_PNG_B64, VALID_PNG_B64],
            image_mimes=["image/png"],
            model="openclip-vit-l-14",
        )


# ---- search ----


def test_search_accepts_query_image():
    req = SearchRequest(
        primary_field="id",
        vector_field="vector",
        query_image=VALID_PNG_B64,
        query_image_mime="image/png",
        model="openclip-vit-l-14",
        top_k=5,
    )
    assert req.query_image is not None


def test_search_requires_query_image_mime():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_image=VALID_PNG_B64,
            model="openclip-vit-l-14",
        )


def test_search_requires_model_when_query_image_present():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_image=VALID_PNG_B64,
            query_image_mime="image/png",
        )


def test_search_rejects_query_text_plus_query_image():
    with pytest.raises(ValidationError):
        SearchRequest(
            primary_field="id",
            vector_field="vector",
            query_text="hello",
            query_image=VALID_PNG_B64,
            query_image_mime="image/png",
            model="openclip-vit-l-14",
        )