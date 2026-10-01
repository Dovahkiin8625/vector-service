"""LLM graph extraction for one chunk: tolerant parsing, never raises."""
from __future__ import annotations

import json

from vector_service.graph.extraction import (
    MAX_NAME_CHARS,
    extract_for_chunk,
)


def _payload(*, entities=(), relationships=(), claims=()):
    return json.dumps({
        "entities": entities,
        "relationships": relationships,
        "claims": claims,
    })


def test_extraction_builds_all_three_sections():
    raw = _payload(
        entities=[
            {"name": "Alice", "type": "person", "description": "an analyst"},
            {"name": "Acme", "type": "org", "description": "a vendor"},
        ],
        relationships=[
            {"source": "Alice", "target": "Acme", "description": "works at"},
        ],
        claims=[
            {"subject": "Alice", "object": "Acme", "type": "employment",
             "status": "true", "statement": "Alice works at Acme"},
        ],
    )
    result = extract_for_chunk(lambda m: raw, "chunk text")
    assert [e.name for e in result.entities] == ["Alice", "Acme"]
    assert result.entities[0].entity_type == "person"
    assert result.edges[0].source == "Alice"
    assert result.edges[0].target == "Acme"
    assert result.claims[0].statement == "Alice works at Acme"
    assert result.claims[0].object == "Acme"


def test_chat_failure_returns_empty_extraction():
    def boom(messages):
        raise RuntimeError("llm down")

    result = extract_for_chunk(boom, "chunk text")
    assert result.entities == []
    assert result.edges == []
    assert result.claims == []


def test_unparseable_response_returns_empty_extraction():
    result = extract_for_chunk(lambda m: "no json whatsoever", "text")
    assert result.entities == []


def test_json_fence_and_prose_are_tolerated():
    raw = 'Here you go:\n```json\n' + _payload(
        entities=[{"name": "Bob", "type": "person"}]
    ) + '\n```\nthanks'
    result = extract_for_chunk(lambda m: raw, "text")
    assert [e.name for e in result.entities] == ["Bob"]


def test_duplicate_entity_names_dedup():
    raw = _payload(entities=[
        {"name": "Pat", "type": "person"},
        {"name": "Pat", "type": "role"},
    ])
    result = extract_for_chunk(lambda m: raw, "text")
    assert [e.name for e in result.entities] == ["Pat"]


def test_entity_without_name_skipped():
    raw = _payload(entities=[
        {"type": "person"},
        {"name": "  ", "type": "person"},
        {"name": "Dana", "type": "person"},
    ])
    result = extract_for_chunk(lambda m: raw, "text")
    assert [e.name for e in result.entities] == ["Dana"]


def test_long_entity_name_truncated():
    raw = _payload(entities=[{"name": "x" * 200}])
    result = extract_for_chunk(lambda m: raw, "text")
    assert len(result.entities[0].name) == MAX_NAME_CHARS


def test_claim_without_subject_or_statement_skipped():
    raw = _payload(claims=[
        {"statement": "no subject"},
        {"subject": "Alice"},
        {"subject": "Alice", "statement": "a real claim"},
    ])
    result = extract_for_chunk(lambda m: raw, "text")
    assert len(result.claims) == 1
    assert result.claims[0].object is None
