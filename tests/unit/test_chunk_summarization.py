"""Unit tests for :func:`summarize_chunks` — the per-chunk LLM summary.

Pins the behavior the ingest pipeline relies on:

- each chunk gets its ``summary`` filled from the chat answer;
- the prompt carries the section header when one exists;
- a failed/empty call leaves ``summary`` as ``None`` (the row still gets
  a ``summary_vector`` later, embedded from chunk text);
- calls fan out through a bounded thread pool.
"""
from __future__ import annotations

import threading

from vector_service.chunking.base import Chunk
from vector_service.chunking.llm_chunker import summarize_chunks


def _chunk(i: int, *, section: str = "") -> Chunk:
    return Chunk(
        text=f"chunk text {i}",
        chunk_index=i,
        token_count=10,
        section_header=section,
    )


class RecordingChat:
    def __init__(self, answer="a summary"):
        self.answer = answer
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()

    def __call__(self, messages):
        with self._lock:
            self.calls.append(messages)
        return self.answer


def test_fills_summary_and_returns_same_chunks():
    chunks = [_chunk(0), _chunk(1)]
    out = summarize_chunks(chunks, chat_fn=RecordingChat(), max_concurrency=2)
    assert out is chunks
    assert [c.summary for c in chunks] == ["a summary", "a summary"]


def test_prompt_carries_system_and_section():
    chat = RecordingChat()
    summarize_chunks([_chunk(0, section="1. Intro")], chat_fn=chat)
    system, user = chat.calls[0]
    assert system["role"] == "system" and system["content"]
    assert user["role"] == "user"
    assert "Section: 1. Intro" in user["content"]
    assert "chunk text 0" in user["content"]


def test_section_omitted_when_absent():
    chat = RecordingChat()
    summarize_chunks([_chunk(0)], chat_fn=chat)
    user = chat.calls[0][1]
    assert "Section:" not in user["content"]


def test_chat_exception_leaves_none_but_others_succeed():
    def flaky(messages):
        if flaky.n == 0:
            flaky.n += 1
            raise RuntimeError("upstream timeout")
        return "ok"
    flaky.n = 0

    chunks = [_chunk(0), _chunk(1)]
    summarize_chunks(chunks, chat_fn=flaky, max_concurrency=1)
    assert chunks[0].summary is None
    assert chunks[1].summary == "ok"


def test_empty_answer_leaves_none():
    chunks = [_chunk(0)]
    summarize_chunks(chunks, chat_fn=RecordingChat(answer="   "))
    assert chunks[0].summary is None


def test_answers_are_stripped():
    chunks = [_chunk(0)]
    summarize_chunks(chunks, chat_fn=RecordingChat(answer="\n summary text \n"))
    assert chunks[0].summary == "summary text"


def test_runs_through_pool_with_many_chunks():
    # max_concurrency=1 still processes every chunk exactly once.
    chat = RecordingChat(answer="s")
    chunks = [_chunk(i) for i in range(8)]
    summarize_chunks(chunks, chat_fn=chat, max_concurrency=1)
    assert len(chat.calls) == 8
    assert all(c.summary == "s" for c in chunks)


def test_think_block_is_stripped():
    answer = "<think>reasoning about the chunk\nmore thoughts</think>\n\n这是中文摘要"
    chunks = [_chunk(0)]
    summarize_chunks(chunks, chat_fn=RecordingChat(answer=answer))
    assert chunks[0].summary == "这是中文摘要"


def test_unclosed_think_block_is_stripped():
    # Generation cut off before </think>: no usable answer remains.
    chunks = [_chunk(0)]
    summarize_chunks(chunks, chat_fn=RecordingChat(answer="<think>reasoning to EOF"))
    assert chunks[0].summary is None


def test_think_only_answer_leaves_none():
    answer = "<think>entirely reasoning</think>"
    chunks = [_chunk(0)]
    summarize_chunks(chunks, chat_fn=RecordingChat(answer=answer))
    assert chunks[0].summary is None


def test_prompt_requires_same_language():
    chat = RecordingChat()
    summarize_chunks([_chunk(0)], chat_fn=chat)
    system, user = chat.calls[0]
    assert "same language" in system["content"].lower()
    assert "SAME" in user["content"] and "language" in user["content"]


def test_empty_input_makes_no_calls():
    chat = RecordingChat()
    out = summarize_chunks([], chat_fn=chat)
    assert out == []
    assert chat.calls == []
