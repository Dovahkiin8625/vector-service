"""LLM query transforms with graceful fallback on bad LLM output."""
from __future__ import annotations

import math

import pytest

from vector_service.retrieval import transforms
from vector_service.retrieval.base import RecallSpec


class FakeChat:
    def __init__(self, text):
        self.text = text
        self.messages = []

    def __call__(self, messages):
        self.messages.extend(messages)
        return self.text


class FakeEmbedder:
    """Maps fixed texts to simple vectors."""

    def embed_query(self, query):
        if query == "q":
            return [1.0, 0.0]
        return [0.0, 1.0]  # hypothetical answer


# ---- JSON extraction ----

def test_parse_plain_object():
    assert transforms.parse_json_object('{"queries": ["a", "b"]}') == {
        "queries": ["a", "b"]
    }


def test_parse_fenced_object():
    raw = 'noise before\n```json\n{"query": "why?"}\n```\nafter'
    assert transforms.parse_json_object(raw) == {"query": "why?"}


def test_parse_object_with_trailing_text():
    raw = 'sure:\n{"sub_queries": ["s1"]}\nhope that helps'
    assert transforms.parse_json_object(raw) == {"sub_queries": ["s1"]}


def test_parse_bad_json_raises():
    with pytest.raises(ValueError):
        transforms.parse_json_object("not json at all")


# ---- multi-query / step-back / decompose ----

def test_multi_query_parses_and_truncates():
    chat = FakeChat('```json\n{"queries": ["v1", "v2", "v3", "v4"]}\n```')
    assert transforms.multi_query(chat, "q", n=3) == ["v1", "v2", "v3"]


def test_multi_query_falls_back_to_original():
    chat = FakeChat("garbage")
    assert transforms.multi_query(chat, "原始问题", n=3) == ["原始问题"]


def test_multi_query_empty_list_falls_back():
    chat = FakeChat('{"queries": []}')
    assert transforms.multi_query(chat, "q") == ["q"]


def test_step_back_parses():
    chat = FakeChat('{"query": "what drives quarterly revenue?"}')
    assert transforms.step_back(chat, "q3 营收如何") == (
        "what drives quarterly revenue?"
    )


def test_step_back_falls_back():
    assert transforms.step_back(FakeChat("??"), "原问题") == "原问题"


def test_decompose_parses():
    chat = FakeChat('{"sub_queries": ["s1", "s2"]}')
    assert transforms.decompose(chat, "q") == ["s1", "s2"]


def test_decompose_bad_output_is_empty():
    assert transforms.decompose(FakeChat("nope"), "q") == []


# ---- HyDE ----

def test_hyde_mixes_vectors():
    chat = FakeChat("This is the hypothetical answer.")
    spec = transforms.hyde(chat, "q", FakeEmbedder(), alpha=0.7)
    assert isinstance(spec, RecallSpec)
    assert spec.query == "q"
    assert spec.hypothetical == "This is the hypothetical answer."
    # mix = 0.3*(1,0) + 0.7*(0,1) = (0.3, 0.7); then L2-normalized
    n = math.hypot(0.3, 0.7)
    assert spec.vector == [pytest.approx(0.3 / n), pytest.approx(0.7 / n)]


def test_hyde_alpha_one_uses_answer_only():
    chat = FakeChat("doc")
    spec = transforms.hyde(chat, "q", FakeEmbedder(), alpha=1.0)
    assert spec.vector == [pytest.approx(0.0), pytest.approx(1.0)]
