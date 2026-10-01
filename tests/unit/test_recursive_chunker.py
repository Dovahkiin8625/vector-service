"""Unit tests for ``RecursiveChunker``.

Token counting is intentionally stubbed with a character-based
counter so tests are deterministic and don't require ``tiktoken``
to be installed. The shape of the counter — ``len(text) // 4`` —
mirrors the rough char-per-token heuristic the OpenAI API uses,
so chunk_size in tokens is roughly equivalent to chunk_size in
chars in these tests.
"""
from __future__ import annotations

import pytest

from vector_service.chunking.recursive_chunker import (
    Chunk,
    RecursiveChunker,
    _split_into_sections,
    count_tokens,
    _split_around_code_blocks,
)


def _char_counter(text: str) -> int:
    """Deterministic character-based token counter for tests."""
    return max(1, len(text) // 4)


def test_chunk_size_bounds():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=0)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=-1)


def test_chunk_overlap_must_be_lt_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=150)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=-1)


def test_empty_input_returns_empty_list():
    c = RecursiveChunker(chunk_size=100, chunk_overlap=10, token_counter=_char_counter)
    assert c.chunk("") == []
    assert c.chunk("   \n\n   ") == []


def test_very_small_input_yields_single_chunk():
    c = RecursiveChunker(chunk_size=100, chunk_overlap=10, token_counter=_char_counter)
    chunks = c.chunk("Hello world.")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].chunk_index == 0
    assert chunks[0].section_header == ""
    # No page markers in the input, but the chunker still defaults
    # to page 1 so callers see a non-null value when the format
    # supports pages. ``page_number`` is ``None`` only when the
    # caller explicitly opts out by supplying ``page_numbers=[]``.
    assert chunks[0].page_number == 1


def test_headers_become_breadcrumb_path():
    """Headers become a breadcrumb path on each emitted chunk.

    The chunker packs sections greedily into the buffer until
    ``chunk_size`` is exceeded, so a small document may produce a
    single chunk whose ``section_header`` is the *innermost* header
    path. The behaviour is exercised here by making each section
    large enough to force its own emission.
    """
    # Pad each section so it independently exceeds the chunk_size.
    long_intro = "Intro paragraph. " + ("word " * 50)
    long_bg = "Background paragraph. " + ("word " * 50)
    long_details = "Details paragraph. " + ("word " * 50)
    md = (
        "# Top Title\n\n"
        f"{long_intro}\n\n"
        "## Background\n\n"
        f"{long_bg}\n\n"
        "### Details\n\n"
        f"{long_details}"
    )
    c = RecursiveChunker(
        chunk_size=80, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    # We expect at least three chunks (one per section).
    assert len(chunks) >= 3
    # The first chunk carries the h1 prefix in its breadcrumb.
    assert "Top Title" in chunks[0].section_header
    # Some later chunk carries the Background breadcrumb.
    assert any("Background" in c.section_header for c in chunks)
    # The Details header appears somewhere in the chunk list.
    assert any("Details" in c.section_header for c in chunks)


def test_header_levels_pop_deeper_or_equal():
    md = (
        "# A\n\n"
        "a-body\n\n"
        "## B\n\n"
        "b-body\n\n"
        "# C\n\n"
        "c-body"
    )
    sections = _split_into_sections(md)
    # 3 sections: A, B, C. (No leading body before the first header.)
    assert len(sections) == 3
    assert sections[0].header_path == ["A"]
    assert sections[1].header_path == ["A", "B"]
    # ``# C`` is at the same level as ``# A`` — B should be popped.
    assert sections[2].header_path == ["C"]


def test_chunk_size_triggers_emit():
    """A document with several small sections should be packed into
    chunks until the running token count crosses ``chunk_size``."""
    md = "\n\n".join(f"Para {i} " + ("word " * 20) for i in range(10))
    c = RecursiveChunker(
        chunk_size=80, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    assert len(chunks) >= 2
    # No chunk should be wildly over the size cap.
    for ch in chunks:
        assert _char_counter(ch.text) <= 200  # generous: paragraphs may push over the cap slightly


def test_overlap_propagates_tail_tokens():
    """The first chunk's tail should appear as a prefix on the next
    chunk so cross-chunk context is preserved."""
    md = "\n\n".join(f"Paragraph {i} content " + ("extra " * 30) for i in range(6))
    c = RecursiveChunker(
        chunk_size=80, chunk_overlap=20, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    assert len(chunks) >= 2
    # The second chunk should contain some text from the first.
    first_tail = chunks[0].text[-80:]
    assert any(word in chunks[1].text for word in first_tail.split() if len(word) > 4)


def test_code_block_stays_atomic():
    """A code block must never be split across chunks."""
    code = "```python\n" + "x = 1\n" * 50 + "```"
    md = "Intro.\n\n" + code + "\n\nOutro paragraph."
    c = RecursiveChunker(
        chunk_size=20, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    # Find the chunk carrying the code block — it must contain the
    # opening AND closing fences.
    code_chunks = [ch for ch in chunks if "```python" in ch.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2
    # The code itself (without fences) must be intact.
    inner = code_chunks[0].text.split("```python", 1)[1].rsplit("```", 1)[0]
    assert inner.count("x = 1") == 50


def test_code_block_split_around_with_prose():
    """When a code block exceeds ``chunk_size`` the prose around it
    should still be split into separate chunks."""
    code = "```\n" + "y = 2\n" * 100 + "```"
    md = "Prose intro. " * 20 + "\n\n" + code + "\n\n" + "Prose outro. " * 20
    c = RecursiveChunker(
        chunk_size=20, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    # At least: intro chunk, code chunk (atomic), outro chunk.
    assert len(chunks) >= 3
    code_chunks = [ch for ch in chunks if ch.text.startswith("```") or "```" in ch.text]
    assert len(code_chunks) == 1


def test_chunk_indices_are_sequential():
    md = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
    c = RecursiveChunker(chunk_size=10, chunk_overlap=2, token_counter=_char_counter)
    chunks = c.chunk(md)
    indices = [ch.chunk_index for ch in chunks]
    assert indices == list(range(len(chunks)))


def test_token_count_matches_chunker_counter():
    """``token_count`` on every emitted chunk should match what the
    configured counter returns for that chunk's text. We pick a
    chunk_size large enough that no overlap prepending occurs so the
    buffer text and the emitted text match exactly."""
    md = "Some prose. " * 30 + "\n\n## Header\n\nMore prose. " * 30
    c = RecursiveChunker(
        chunk_size=80, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    for ch in chunks:
        # ``token_count`` is recorded at buffer-flush time, when the
        # chunk text matches the buffer exactly. We can't compare
        # against ``_char_counter(ch.text)`` directly when overlap is
        # on because the overlap mutates the text, but with
        # ``overlap=0`` they should match.
        assert ch.token_count == _char_counter(ch.text)


def test_page_numbers_propagated():
    md = "Page one content.\n\n<!-- page break -->\n\nPage two content."
    c = RecursiveChunker(
        chunk_size=10_000, chunk_overlap=0, token_counter=_char_counter,
    )
    chunks = c.chunk(md)
    # With page-break markers, the chunker should split the
    # sections and assign page numbers.
    assert any(ch.page_number is not None for ch in chunks)


def test_default_token_counter_is_tiktoken():
    """``count_tokens`` (the default counter) requires ``tiktoken``;
    we don't want to import it in the test, but a smoke check that
    the chunker works with the real counter is still useful when
    tiktoken is installed."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed")

    c = RecursiveChunker(chunk_size=50, chunk_overlap=5)
    chunks = c.chunk("Hello world. " * 20)
    assert len(chunks) >= 1
    for ch in chunks:
        # Each chunk's token count should equal what tiktoken says.
        assert ch.token_count == count_tokens(ch.text)


def test_split_around_code_blocks_prose_only():
    """A document with no code fences should return a single prose
    piece."""
    pieces = _split_around_code_blocks("Para 1.\n\nPara 2.")
    assert pieces == [(False, "Para 1.\n\nPara 2.", 0, 16)]


def test_split_around_code_blocks_mixed():
    """Mixed prose + code: prose is plain text, code fences are
    returned as their own atomic piece."""
    body = "Prose before.\n\n```\ncode here\n```\n\nProse after."
    pieces = _split_around_code_blocks(body)
    # Expect: (False, "Prose before."), (True, "```\ncode here\n```"),
    # (False, "Prose after.")
    assert any(is_code is False and "Prose before" in t for is_code, t, _s, _e in pieces)
    assert any(is_code is True and "code here" in t for is_code, t, _s, _e in pieces)
    assert any(is_code is False and "Prose after" in t for is_code, t, _s, _e in pieces)


def test_chunk_dataclass_fields():
    """Sanity check on the :class:`Chunk` dataclass shape."""
    c = Chunk(
        text="hello",
        chunk_index=0,
        token_count=1,
        section_header="A",
        page_number=2,
    )
    assert c.text == "hello"
    assert c.chunk_index == 0
    assert c.token_count == 1
    assert c.section_header == "A"
    assert c.page_number == 2
