"""Unit tests for the pluggable chunking strategies.

Covers fixed / paragraph / semantic / llm chunkers, contextual
chunk enrichment, and the factory + registry. Token counting is
stubbed with the same deterministic char-based counter used by
``test_recursive_chunker.py``; the fixed chunker's token windows
are exercised with text whose tiktoken count is known ("word " is
one token).
"""
from __future__ import annotations

import pytest

from vector_service.chunking import build_chunker, list_strategies
from vector_service.chunking.base import Chunk
from vector_service.chunking.llm_chunker import contextualize_chunks


def _char_counter(text: str) -> int:
    return max(1, len(text) // 4)


ALL_STRATEGIES = ["fixed", "paragraph", "recursive", "semantic", "llm"]


# ---- registry / factory -----------------------------------------------


def test_registry_lists_every_strategy():
    assert list_strategies() == ALL_STRATEGIES


def test_factory_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        build_chunker("nope", chunk_size=100, chunk_overlap=10)


def test_factory_semantic_requires_embed_fn_at_chunk_time():
    chunker = build_chunker(
        "semantic", chunk_size=100, chunk_overlap=0,
        embed_fn=lambda _t: [],
    )
    assert chunker is not None
    with pytest.raises(ValueError):
        # Construction without embed fn fails.
        build_chunker("semantic", chunk_size=100, chunk_overlap=0)


def test_factory_llm_requires_chat_fn():
    with pytest.raises(ValueError):
        build_chunker("llm", chunk_size=100, chunk_overlap=0)
    chunker = build_chunker(
        "llm", chunk_size=100, chunk_overlap=0, chat_fn=lambda _m: "",
    )
    assert chunker is not None


def test_factory_ignores_unknown_options(caplog):
    chunker = build_chunker(
        "recursive", chunk_size=100, chunk_overlap=0,
        options={"definitely_not_a_key": 1},
    )
    assert chunker is not None


# ---- fixed -------------------------------------------------------------


def test_fixed_empty_input():
    c = build_chunker("fixed", chunk_size=50, chunk_overlap=0)
    assert c.chunk("") == []
    assert c.chunk("   \n ") == []


def test_fixed_window_sizes_and_counts():
    # "word " is exactly one cl100k_base token → 120 tokens total.
    text = "word " * 120
    c = build_chunker("fixed", chunk_size=50, chunk_overlap=10)
    chunks = c.chunk(text)
    # step = 40 → starts 0, 40, 80 → 50 / 50 / 40 tokens.
    assert [x.token_count for x in chunks] == [50, 50, 40]
    assert [x.chunk_index for x in chunks] == [0, 1, 2]


def test_fixed_no_overlap_partitions_all_tokens():
    text = "word " * 120
    c = build_chunker("fixed", chunk_size=50, chunk_overlap=0)
    chunks = c.chunk(text)
    assert [x.token_count for x in chunks] == [50, 50, 20]


def test_fixed_protects_code_block():
    code = "```python\n" + "x = 1\n" * 80 + "```"
    md = ("word " * 20) + "\n\n" + code + "\n\n" + ("word " * 20)
    c = build_chunker("fixed", chunk_size=50, chunk_overlap=0,
                       options={"protect_code": True})
    chunks = c.chunk(md)
    code_chunks = [ch for ch in chunks if "```python" in ch.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2
    inner = code_chunks[0].text.split("```python", 1)[1].rsplit("```", 1)[0]
    assert inner.count("x = 1") == 80


def test_fixed_protect_code_option_accepted():
    c = build_chunker("fixed", chunk_size=50, chunk_overlap=0,
                       options={"protect_code": False})
    assert c.chunk("word " * 60)  # doesn't error, windows produced


# ---- paragraph ---------------------------------------------------------


def test_paragraph_merges_short_paragraphs():
    md = "\n\n".join(f"Short paragraph number {i}." for i in range(3))
    c = build_chunker("paragraph", chunk_size=500, chunk_overlap=0,
                       token_counter=_char_counter)
    chunks = c.chunk(md)
    assert len(chunks) == 1
    for i in range(3):
        assert f"paragraph number {i}" in chunks[0].text


def test_paragraph_respects_size_cap():
    md = "\n\n".join(f"Paragraph {i} body " + ("word " * 20) for i in range(10))
    c = build_chunker("paragraph", chunk_size=40, chunk_overlap=0,
                       token_counter=_char_counter)
    chunks = c.chunk(md)
    assert len(chunks) >= 2
    # Units are pre-split to fit → every emitted chunk within cap.
    for ch in chunks:
        assert _char_counter(ch.text) <= 40


def test_paragraph_oversized_degrades_to_sentences():
    long_para = "This is sentence one. " * 30
    md = "Before.\n\n" + long_para + "\n\nAfter."
    c = build_chunker("paragraph", chunk_size=40, chunk_overlap=0,
                       token_counter=_char_counter)
    chunks = c.chunk(md)
    assert len(chunks) >= 2
    for ch in chunks:
        assert _char_counter(ch.text) <= 40


def test_paragraph_keeps_breadcrumb():
    md = "# Title\n\n## Section\n\n" + "Word. " * 40
    c = build_chunker("paragraph", chunk_size=40, chunk_overlap=0,
                       token_counter=_char_counter)
    chunks = c.chunk(md)
    assert any("Title" in ch.section_header for ch in chunks)
    # The literal header participates in the first chunk's embedding.
    assert chunks[0].text.startswith("# Title")


# ---- semantic ----------------------------------------------------------


def _topic_embedder(texts):
    # Two orthogonal one-hot topics keyed off the sentence content.
    return [[1.0, 0.0] if "Cats" in t else [0.0, 1.0] for t in texts]


def test_semantic_cuts_at_topic_shift():
    # 15 sentences per side keep each group above the default
    # min_chunk_size floor (10% of 500 = 50 tokens).
    sentences = ["Cats meow loudly. "] * 15 + ["Dogs bark loudly. "] * 15
    md = "".join(sentences)
    c = build_chunker("semantic", chunk_size=500, chunk_overlap=0,
                       options={"breakpoint_percentile": 95},
                       embed_fn=_topic_embedder)
    chunks = c.chunk(md)
    assert len(chunks) == 2
    assert "Cats" in chunks[0].text and "Dogs" not in chunks[0].text
    assert "Dogs" in chunks[1].text and "Cats" not in chunks[1].text


def test_semantic_uniform_topic_is_one_group():
    md = "Same topic text. " * 10

    def _uniform(_texts):
        return [[1.0, 1.0] for _ in _texts]

    c = build_chunker("semantic", chunk_size=500, chunk_overlap=0,
                       embed_fn=_uniform)
    assert len(c.chunk(md)) == 1


def test_semantic_size_cap_still_packs():
    md = "Same topic sentence text. " * 20

    def _uniform(_texts):
        return [[1.0, 1.0] for _ in _texts]

    c = build_chunker("semantic", chunk_size=10, chunk_overlap=0,
                       token_counter=_char_counter, embed_fn=_uniform)
    chunks = c.chunk(md)
    assert len(chunks) >= 3
    for ch in chunks:
        assert _char_counter(ch.text) <= 10


def test_semantic_embeds_all_sentences_in_one_batch():
    md = "Alpha sentence. " * 8
    seen: list[int] = []

    def _record(texts):
        seen.append(len(texts))
        return [[1.0, 0.5] for _ in texts]

    c = build_chunker("semantic", chunk_size=500, chunk_overlap=0,
                       embed_fn=_record)
    c.chunk(md)
    assert seen == [8]


# ---- llm ----------------------------------------------------------------


def test_llm_cuts_at_returned_positions():
    # 15 sentences per side keep both chunks above the 50-token floor.
    md = "Topic A sentence. " * 15 + "Topic B sentence. " * 15

    def _chat(_messages):
        return '{"break_after": [15]}'

    c = build_chunker("llm", chunk_size=500, chunk_overlap=0, chat_fn=_chat)
    chunks = c.chunk(md)
    assert len(chunks) == 2


def test_llm_regex_fallback_for_loose_answer():
    # 36 sentences split at 18 -> ~54 tokens per side, above the floor.
    md = "Sentence one. " * 36

    def _chat(_messages):
        return "I would break the text after position 18 please."

    c = build_chunker("llm", chunk_size=500, chunk_overlap=0, chat_fn=_chat)
    chunks = c.chunk(md)
    assert len(chunks) == 2


def test_llm_bad_answer_degrades_without_crash():
    md = "One. " * 5

    def _boom(_messages):
        raise RuntimeError("endpoint down")

    c = build_chunker("llm", chunk_size=500, chunk_overlap=0, chat_fn=_boom)
    chunks = c.chunk(md)
    assert len(chunks) == 1  # whole section, no cuts


def test_llm_clamps_out_of_range_positions():
    from vector_service.chunking.llm_chunker import _parse_break_indices
    positions = _parse_break_indices('{"break_after": [0, 3, 99]}', 10)
    assert positions == {3}


def test_llm_windows_long_sections():
    md = "Sentence about one specific topic here. " * 6
    calls = 0

    def _chat(_messages):
        nonlocal calls
        calls += 1
        return '{"break_after": [1]}'

    c = build_chunker(
        "llm", chunk_size=500, chunk_overlap=0,
        options={"input_token_budget": 20},
        token_counter=_char_counter, chat_fn=_chat,
    )
    c.chunk(md)
    # Each sentence ≈ 9 char-counter tokens; budget 20 → 2 per window, 3 windows.
    assert calls == 3


# ---- contextual enrichment ---------------------------------------------


def test_contextualize_sets_context_per_chunk():
    chunks = [
        Chunk(text="Alpha body.", chunk_index=0, token_count=2, section_header=""),
        Chunk(text="Beta body.", chunk_index=1, token_count=2, section_header=""),
    ]

    def _chat(_messages):
        return "This chunk is from the project documentation."

    contextualize_chunks(chunks, document="doc", chat_fn=_chat)
    assert all(ch.context == "This chunk is from the project documentation."
               for ch in chunks)


def test_contextualize_failure_leaves_none():
    chunks = [Chunk(text="Body.", chunk_index=0, token_count=1, section_header="")]

    def _boom(_messages):
        raise RuntimeError("timeout")

    contextualize_chunks(chunks, document="doc", chat_fn=_boom)
    assert chunks[0].context is None


def test_contextualize_empty_answer_leaves_none():
    chunks = [Chunk(text="Body.", chunk_index=0, token_count=1, section_header="")]
    contextualize_chunks(chunks, document="doc",
                         chat_fn=lambda _m: "   ")
    assert chunks[0].context is None
