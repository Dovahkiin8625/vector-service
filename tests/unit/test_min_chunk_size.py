"""Tests for the no-tiny-fragments guarantee.

Covers the shared ``merge_small_chunks`` post-pass, the
``min_chunk_size`` floor (default 10% of ``chunk_size``) and the
fixed chunker's final-window absorption. Tiny chunks (a few tokens)
carry too little signal to be useful retrieval hits, so undersized
pieces are merged into neighbours rather than emitted standalone.
"""
from __future__ import annotations

import pytest

from vector_service.chunking import Chunk, build_chunker
from vector_service.chunking.structure import merge_small_chunks
from vector_service.chunking.tokens import count_tokens


def _chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        chunk_index=0,
        token_count=len(text),
        section_header="",
        page_number=None,
    )


# ---- merge_small_chunks ------------------------------------------------


def test_merge_absorbs_leading_fragment_into_next():
    chunks = [_chunk("a" * 10), _chunk("b" * 70), _chunk("c" * 70)]
    out = merge_small_chunks(
        chunks, chunk_size=500, token_counter=len, min_size=50
    )
    assert len(out) == 2
    # The 10-token fragment rides along with the following chunk
    # (counts include the 2-char separator).
    assert out[0].text.startswith("a" * 10)
    assert out[0].token_count == 82
    assert out[1].token_count == 70


def test_merge_chained_small_pieces_pack_together():
    chunks = [_chunk("a" * 20), _chunk("b" * 20), _chunk("c" * 60)]
    out = merge_small_chunks(
        chunks, chunk_size=500, token_counter=len, min_size=50
    )
    # 20+20 still under the floor, so the 60-token piece joins too
    # (two separators add 4 chars).
    assert len(out) == 1
    assert out[0].token_count == 104


def test_merge_tiny_tail_absorbed_even_when_over_cap():
    chunks = [_chunk("a" * 495), _chunk("b" * 10)]
    out = merge_small_chunks(
        chunks, chunk_size=500, token_counter=len, min_size=50
    )
    # Forward merge would exceed the cap (505), but the tail is
    # absorbed unconditionally rather than left as a 10-token fragment.
    assert len(out) == 1
    assert out[0].token_count == 507


def test_merge_respects_chunks_already_at_floor():
    chunks = [_chunk("a" * 60), _chunk("b" * 60)]
    out = merge_small_chunks(
        chunks, chunk_size=500, token_counter=len, min_size=50
    )
    assert len(out) == 2


def test_merge_disabled_at_floor_one():
    chunks = [_chunk("a" * 3), _chunk("b" * 3)]
    out = merge_small_chunks(
        chunks, chunk_size=500, token_counter=len, min_size=1
    )
    assert len(out) == 2


# ---- min_chunk_size resolution -----------------------------------------


def test_default_floor_is_ten_percent():
    c = build_chunker("recursive", chunk_size=500, chunk_overlap=50)
    assert c.min_chunk_size == 50
    c = build_chunker("recursive", chunk_size=250, chunk_overlap=25)
    assert c.min_chunk_size == 25


def test_explicit_min_chunk_size_option():
    c = build_chunker(
        "recursive",
        chunk_size=500,
        chunk_overlap=50,
        options={"min_chunk_size": 100},
    )
    assert c.min_chunk_size == 100


def test_min_chunk_size_applies_to_every_strategy():
    for name in ("fixed", "paragraph", "recursive", "semantic", "llm"):
        opts = {"min_chunk_size": 42}
        kwargs = {}
        if name == "semantic":
            kwargs["embed_fn"] = lambda texts: [[1.0] for _ in texts]
        if name == "llm":
            kwargs["chat_fn"] = lambda _m: "{}"
        c = build_chunker(name, chunk_size=500, chunk_overlap=50,
                          options=opts, **kwargs)
        assert c.min_chunk_size == 42


def test_min_chunk_size_must_be_below_chunk_size():
    with pytest.raises(ValueError, match="min_chunk_size"):
        build_chunker(
            "recursive",
            chunk_size=100,
            chunk_overlap=10,
            options={"min_chunk_size": 100},
        )


# ---- fixed chunker tail-window absorption -------------------------------


def _repeat_tokens(word: str, n: int) -> str:
    text = f" {word}" * n
    assert count_tokens(text) == n
    return text


def test_fixed_absorbs_short_final_window():
    # Floor at chunk_size=100 is 10 tokens; an 8-token tail is below it.
    text = _repeat_tokens("apple", 108)
    c = build_chunker("fixed", chunk_size=100, chunk_overlap=0)
    chunks = c.chunk(text)
    # Without absorption: [100, 8]; the short tail becomes one
    # merged window instead.
    assert len(chunks) == 1
    assert chunks[0].token_count == 108


def test_fixed_keeps_final_window_at_or_above_floor():
    text = _repeat_tokens("apple", 160)
    c = build_chunker("fixed", chunk_size=100, chunk_overlap=0)
    chunks = c.chunk(text)
    # Tail is 60 tokens >= 10% floor, so two windows survive.
    assert [ch.token_count for ch in chunks] == [100, 60]


# ---- end-to-end: one-line paragraphs no longer fragment ------------------


def test_recursive_line_per_paragraph_has_no_fragments():
    # Docling-style output: blank line between every physical line.
    lines = [f"This is line number {i} with some words." for i in range(40)]
    markdown = "\n\n".join(lines)
    c = build_chunker("recursive", chunk_size=200, chunk_overlap=20)
    chunks = c.chunk(markdown)
    assert len(chunks) > 1
    for ch in chunks:
        assert ch.token_count >= c.min_chunk_size
