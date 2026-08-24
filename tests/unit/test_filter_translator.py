import pytest

from vector_service.core.filter_translator import translate_filter, FilterTranslationError


def test_empty():
    assert translate_filter(None) == ""
    assert translate_filter({}) == ""


def test_single_equality():
    assert translate_filter({"source": "doc1"}) == 'metadata["source"] == "doc1"'


def test_multiple_keys_are_anded():
    out = translate_filter({"source": "doc1", "lang": "en"})
    assert out == 'metadata["source"] == "doc1" and metadata["lang"] == "en"'


def test_int_value():
    out = translate_filter({"count": 5})
    assert out == 'metadata["count"] == 5'


def test_float_value():
    out = translate_filter({"score": 0.5})
    assert out == 'metadata["score"] == 0.5'


def test_bool_value():
    out = translate_filter({"active": True})
    assert out == 'metadata["active"] == True'


def test_none_value():
    out = translate_filter({"tag": None})
    assert out == 'metadata["tag"] == null'


def test_nested_dict_rejected():
    with pytest.raises(FilterTranslationError):
        translate_filter({"meta": {"a": 1}})


def test_list_value_rejected():
    with pytest.raises(FilterTranslationError):
        translate_filter({"tags": ["a", "b"]})


def test_or_unsupported_in_v1():
    with pytest.raises(FilterTranslationError):
        # v1 doesn't support OR, using operator field forces OR should be rejected
        translate_filter({"__or__": [{"a": 1}, {"b": 2}]})


def test_quote_in_string_escaped():
    out = translate_filter({"name": 'a"b'})
    assert '\\"a\\"b\\"' in out or 'a\\"b' in out
