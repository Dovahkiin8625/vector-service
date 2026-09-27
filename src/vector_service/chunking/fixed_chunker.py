"""Fixed-size token window chunker.

The structure-agnostic baseline: the input is encoded to token ids
and sliced into hard ``chunk_size`` windows, consecutive windows
sharing ``chunk_overlap`` tokens. It is the fastest strategy and
works on anything tokenizable — logs, code dumps, prose — at the
cost of ignoring natural boundaries.

By default code fences are still protected (``protect_code``): the
input is split around fenced blocks, each block is emitted as one
atomic chunk and only the prose runs are window-sliced.
"""
from __future__ import annotations

from collections.abc import Iterable

from vector_service.chunking.base import BaseChunker, Chunk, register_strategy
from vector_service.chunking.structure import split_around_code_blocks
from vector_service.chunking.tokens import decode_ids, encode_ids
from vector_service.core.logging import get_logger

log = get_logger(__name__)


@register_strategy("fixed")
class FixedChunker(BaseChunker):
    """Hard token-window chunker.

    Args:
        chunk_size: Exact target tokens per window.
        chunk_overlap: Shared tokens between adjacent windows;
            ``< chunk_size``.
        protect_code: When true (default), fenced code blocks are
            emitted atomically instead of being sliced mid-code.
    """

    def __init__(self, *args, protect_code: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.protect_code = protect_code

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,  # noqa: ARG002
    ) -> list[Chunk]:
        """Slice ``markdown`` into fixed token windows. Empty → ``[]``."""
        if not markdown or not markdown.strip():
            return []

        if not self.protect_code:
            return _window_text(
                markdown, self.chunk_size, self.chunk_overlap,
                min_window=self.min_chunk_size,
            )

        chunks: list[Chunk] = []
        for is_code, piece in split_around_code_blocks(markdown):
            if is_code:
                ids = encode_ids(piece)
                chunks.append(
                    Chunk(
                        text=piece,
                        chunk_index=len(chunks),
                        token_count=len(ids),
                        section_header="",
                        page_number=None,
                    )
                )
                if len(ids) > self.chunk_size:
                    log.warning(
                        "fixed_oversized_code_block",
                        token_count=len(ids),
                        chunk_size=self.chunk_size,
                    )
            else:
                chunks.extend(
                    _window_text(
                        piece, self.chunk_size, self.chunk_overlap,
                        offset=len(chunks), min_window=self.min_chunk_size,
                    )
                )
        return _reindex(chunks)


def _window_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    offset: int = 0,
    min_window: int = 1,
) -> list[Chunk]:
    """Slice one text run into token windows.

    A final window shorter than ``min_window`` tokens is absorbed
    into the previous window by decoding the combined span — the
    last chunk may then slightly exceed ``chunk_size``, but it beats
    emitting a handful of tokens as a fragment.
    """
    ids = encode_ids(text)
    step = chunk_size - chunk_overlap
    ranges: list[tuple[int, int]] = []
    for start in range(0, len(ids), step):
        ranges.append((start, min(start + chunk_size, len(ids))))
        if start + chunk_size >= len(ids):
            break

    if len(ranges) >= 2 and ranges[-1][1] - ranges[-1][0] < min_window:
        prev_start, _prev_end = ranges[-2]
        ranges[-2] = (prev_start, ranges[-1][1])
        ranges.pop()

    out: list[Chunk] = []
    for start, end in ranges:
        window = ids[start:end]
        out.append(
            Chunk(
                text=decode_ids(window),
                chunk_index=offset + len(out),
                token_count=len(window),
                section_header="",
                page_number=None,
            )
        )
    return out


def _reindex(chunks: list[Chunk]) -> list[Chunk]:
    """Assign sequential chunk_index values after assembly."""
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks
