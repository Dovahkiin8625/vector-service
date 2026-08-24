import pytest
from pydantic import ValidationError

from vector_service.schemas.openai import EmbeddingRequest
from vector_service.schemas.management import (
    CreateCollectionRequest,
    UpsertVectorsRequest,
    SearchRequest,
)


def test_embedding_request_accepts_string():
    r = EmbeddingRequest(input="hi", model="bge-m3")
    assert r.input == "hi"


def test_embedding_request_accepts_list():
    r = EmbeddingRequest(input=["a", "b"], model="bge-m3")
    assert r.input == ["a", "b"]


def test_embedding_request_rejects_empty_string():
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="", model="bge-m3")


def test_embedding_request_rejects_unknown_encoding():
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="x", model="bge-m3", encoding_format="base64")


def test_create_collection_dim_optional():
    r = CreateCollectionRequest(name="c")
    assert r.dim is None
    assert r.metric == "cosine"


def test_upsert_texts_or_embeddings_xor():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a"], texts=["t"], embeddings=[[0.1] * 4])


def test_upsert_requires_one():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a"])


def test_upsert_length_mismatch():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a", "b"], texts=["one"])


def test_search_xor():
    with pytest.raises(ValidationError):
        SearchRequest(query_text="x", query_embedding=[0.1] * 4)


def test_search_requires_one():
    with pytest.raises(ValidationError):
        SearchRequest()