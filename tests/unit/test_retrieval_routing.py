"""Query-intent routing: heuristic surface signals + LLM classifier.

Pins the deterministic router output (intents, signals, channel
booleans, predicates and cleaned query) and the LLM router's
failure-to-None contract — callers degrade to the heuristic decision.
"""
from __future__ import annotations

from vector_service.retrieval.base import MetaPredicate
from vector_service.retrieval.routing import (
    heuristic_route,
    llm_route,
)

# ---- heuristic: keyword ------------------------------------------------


def test_error_code_routes_keyword_exclusive():
    d = heuristic_route("ESE-5021 startup crash")
    assert d.router == "heuristic"
    assert "keyword" in d.intents
    assert "error_code" in d.signals
    # Strong code-like signal: BM25 alone, dense muted.
    assert d.bm25 is True
    assert d.dense is False
    assert d.graph is False


def test_quoted_phrase_is_keyword_but_keeps_dense_safety_net():
    d = heuristic_route('"季度营收" 报告')
    assert "quoted_phrase" in d.signals
    assert "keyword" in d.intents
    # Weak keyword phrasing: hybrid, not exclusive.
    assert d.bm25 is True
    assert d.dense is True


def test_identifier_surfaces_are_exclusive():
    for query, signal in (
        ("deadbeefcafe0123 crash", "hex_hash"),
        ("src/vector/main.py missing", "path"),
        ("v2.4.1 upgrade notes", "version"),
        ("parseJsonObject returns null", "camel_case"),
        ("leaf_chunk_texts empty", "snake_case"),
    ):
        d = heuristic_route(query)
        assert signal in d.signals
        assert d.bm25 is True
        assert d.dense is False


# ---- heuristic: semantic / graph ---------------------------------------


def test_natural_question_routes_semantic():
    d = heuristic_route("季度营收情况如何？")
    assert d.intents == ["semantic"]
    assert "question_form" in d.signals
    assert d.dense is True
    assert d.summary is True
    assert d.bm25 is False


def test_long_cjk_text_routes_semantic_without_question_form():
    query = "季度营收" * 4  # 16 CJK chars
    d = heuristic_route(query)
    assert "semantic" in d.intents
    assert "long_cjk_text" in d.signals
    assert "question_form" not in d.signals


def test_relationship_question_enables_graph():
    d = heuristic_route("张三和李四是什么关系？")
    assert "graph" in d.intents
    assert "relation_phrase" in d.signals
    assert d.graph is True
    # Question form too — dense legs stay available.
    assert "semantic" in d.intents
    assert d.dense is True


# ---- heuristic: predicates ---------------------------------------------


def test_text_predicate_extracted_and_query_cleaned():
    d = heuristic_route("年报 author: 张三")
    assert d.predicates == [MetaPredicate("author", "like", "张三")]
    assert d.query == "年报"
    assert "metadata" in d.intents


def test_numeric_predicate_keeps_native_op():
    d = heuristic_route("季报 chunk >= 3")
    assert d.predicates == [MetaPredicate("chunk_index", ">=", "3")]
    assert d.query == "季报"


def test_unknown_field_predicate_is_left_in_query():
    d = heuristic_route("foo:bar 年报")
    assert d.predicates == []
    assert d.query == "foo:bar 年报"
    assert "metadata" not in d.intents


def test_numeric_field_rejects_non_numeric_value():
    d = heuristic_route("page: abc 年报")
    assert d.predicates == []
    assert d.query == "page: abc 年报"


def test_predicate_aliases_map_to_canonical_fields():
    d = heuristic_route("file=年报.pdf status: published pages>10")
    fields = {(p.field, p.op, p.value) for p in d.predicates}
    assert fields == {
        ("filename", "like", "年报.pdf"),
        ("status", "like", "published"),
        ("page_count", ">", "10"),
    }


# ---- LLM router --------------------------------------------------------


def test_llm_route_parses_classification():
    def chat(messages):
        return '{"intents": ["semantic"], "filters": [], "query": "季度营收"}'

    d = llm_route(chat, "ignored")
    assert d is not None
    assert d.router == "llm"
    assert d.intents == ["semantic"]
    assert d.dense is True
    assert d.summary is True
    assert d.query == "季度营收"


def test_llm_route_accepts_filters():
    def chat(messages):
        return (
            '{"intents": ["keyword", "metadata"],'
            ' "filters": [{"field": "author", "op": "like", "value": "张三"}],'
            ' "query": "年报"}'
        )

    d = llm_route(chat, "ignored")
    assert d is not None
    assert d.predicates == [MetaPredicate("author", "like", "张三")]
    assert "metadata" in d.intents
    assert d.query == "年报"


def test_llm_route_chat_exception_returns_none():
    def chat(messages):
        raise RuntimeError("backend down")

    assert llm_route(chat, "q") is None


def test_llm_route_bad_json_returns_none():
    assert llm_route(lambda m: "not json at all", "q") is None


def test_llm_route_empty_intents_returns_none():
    def chat(messages):
        return '{"intents": [], "filters": [], "query": "q"}'

    assert llm_route(chat, "q") is None


def test_llm_route_unknown_intent_value_returns_none():
    def chat(messages):
        return '{"intents": ["bogus"], "filters": [], "query": "q"}'

    assert llm_route(chat, "q") is None


def test_llm_route_bad_filter_field_returns_none():
    def chat(messages):
        return (
            '{"intents": ["metadata"],'
            ' "filters": [{"field": "sql_inject", "op": "==", "value": "x"}],'
            ' "query": "q"}'
        )

    assert llm_route(chat, "q") is None


def test_llm_route_non_numeric_page_count_returns_none():
    def chat(messages):
        return (
            '{"intents": ["metadata"],'
            ' "filters": [{"field": "page_count", "op": "==", "value": "many"}],'
            ' "query": "q"}'
        )

    assert llm_route(chat, "q") is None


def test_llm_route_missing_query_falls_back_to_original():
    def chat(messages):
        return '{"intents": ["semantic"], "filters": []}'

    d = llm_route(chat, "原始问题")
    assert d is not None
    assert d.query == "原始问题"
