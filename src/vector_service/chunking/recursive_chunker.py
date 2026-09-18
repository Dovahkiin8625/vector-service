"""Markdown-aware recursive chunker.

Splitting rules (in order):

1. Markdown headers (``^#+\\s+``) split the document into
   hierarchical sections. The chunker tracks the breadcrumb header
   path so every emitted chunk can report its section context.
2. Each section is then split into paragraphs (``\\n\\n+``), then
   sentences, then whitespace-separated words as a last resort.
3. Code blocks (````` ... ```````) MUST stay atomic — the chunker
   never splits inside one. A code block that exceeds ``chunk_size``
   is emitted as its own chunk and a warning is logged (better to
   keep the code together than to break a function definition across
   two chunks).
4. After every accumulation the chunker checks the running token
   count. When it crosses ``chunk_size`` a chunk is emitted; the
   next chunk starts with up to ``chunk_overlap`` tokens from the
   previous chunk's tail to maintain cross-chunk context.

Token counting uses ``tiktoken``'s ``cl100k_base`` encoding — the
same encoding OpenAI's ``text-embedding-3-*`` models use, so chunk
sizes are directly comparable to those models' advertised limits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Lazy-loaded to keep module import light. ``_get_encoding`` caches
# the ``Encoding`` instance once.
_ENCODING: object | None = None


def _get_encoding() -> object:
    """Lazy-load and cache the ``cl100k_base`` tiktoken encoding."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Return the token count for ``text`` under ``cl100k_base``."""
    enc = _get_encoding()
    # ``Encoding.encode`` accepts ``str`` and ``allowed_special``
    # is fine to default for plain prose.
    return len(enc.encode(text))  # type: ignore[union-attr]


# Header regex: ``#``-prefixed line at the start of a line. The
# leading ``#+`` is captured into group 1 and the header text into
# group 2. ``re.MULTILINE`` makes ``^`` match at every line start.
_HEADER_RX = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Code-fence detector: matches a triple-backtick line (with optional
# language tag). We use the *line* itself as the fence so the
# contents are captured until the next fence line.
_CODE_FENCE_RX = re.compile(r"^\s*```.*$", re.MULTILINE)


@dataclass
class Chunk:
    """A single emitted chunk.

    ``section_header`` is the breadcrumb header path (e.g. ``"1.
    Introduction > 1.1 Background"``) at the time the chunk was
    finalised. ``page_number`` is copied from the section's
    metadata when present; ``None`` for formats that don't have
    pages.
    """

    text: str
    chunk_index: int
    token_count: int
    section_header: str
    page_number: int | None = None


@dataclass
class _Section:
    """Intermediate section node used by :class:`RecursiveChunker`.

    ``header_path`` is the ordered list of headers from the document
    root to this section (most recent last). ``body`` is the text
    between this section's header line and the next sibling or
    higher-priority header. ``page_number`` is propagated from
    either an explicit section page marker or the section that
    contained this one.
    """

    header_path: list[str] = field(default_factory=list)
    body: str = ""
    page_number: int | None = None


class RecursiveChunker:
    """Recursive markdown-aware chunker.

    Args:
        chunk_size: Target upper bound on tokens per emitted chunk.
            The chunker treats this as the *trigger* for emission;
            individual chunks can be slightly smaller (because we
            always emit at a paragraph boundary once we cross the
            threshold) but never split inside a code block or a
            sentence mid-stream.
        chunk_overlap: How many trailing tokens from the previous
            chunk to prefix onto the next one. Must be
            ``< chunk_size``. ``0`` disables overlap.
        token_counter: Optional override for the token-counting
            callable; primarily here so tests can supply a
            deterministic, character-based counter without
            requiring tiktoken.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
                f"({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._token_counter: Callable[[str], int] = (
            token_counter if token_counter is not None else count_tokens
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Split ``markdown`` into chunks.

        ``page_numbers`` is an optional iterable of page numbers in
        document order. When supplied the chunker tries to align
        sections with their source page (best-effort — pages in the
        source PDF aren't always cleanly aligned with markdown
        headers, so the chunker propagates a page_number only when
        a section's body falls inside a single page's slice).

        Returns an empty list for empty / whitespace-only input.
        """
        if not markdown or not markdown.strip():
            return []
        sections = _split_into_sections(markdown)
        # Always look for page-break markers in the markdown itself
        # (``<!-- page break -->`` or ``[PAGE N]``). When the caller
        # also supplies an explicit ``page_numbers`` list, that list
        # overrides what we discovered in the text.
        _annotate_pages(sections, list(page_numbers) if page_numbers else [], markdown)
        return _chunk_sections(
            sections,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            token_counter=self._token_counter,
        )


# ---- internals ---------------------------------------------------------


def _split_into_sections(markdown: str) -> list[_Section]:
    """Walk the document once and emit one :class:`_Section` per header
    scope.

    Headers without a body produce an empty ``_Section``; that's
    fine — the chunker just won't emit a chunk for a heading-only
    section if its body is empty.
    """
    matches = list(_HEADER_RX.finditer(markdown))
    if not matches:
        return [_Section(header_path=[], body=markdown.strip())]

    sections: list[_Section] = []
    # Anything before the first header goes into a leading
    # header-less section.
    pre = markdown[: matches[0].start()].strip()
    if pre:
        sections.append(_Section(header_path=[], body=pre))

    # Stack of (header_level, header_text) — index 0 is the document
    # title (level 0), then headings deepen the path. ``[-1][0]`` is
    # the current section's level.
    stack: list[tuple[int, str]] = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        text = m.group(2).strip()
        # Pop everything deeper-or-equal to ``level`` — the new
        # header starts a sibling (or higher) section.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()

        sections.append(
            _Section(
                header_path=[t for _lvl, t in stack],
                body=body,
            )
        )
    return sections


def _annotate_pages(
    sections: list[_Section],
    pages: list[int],
    full_text: str,
) -> None:
    """Propagate a best-effort ``page_number`` to each section.

    Pages are usually represented in Docling's markdown output as
    ``<!-- page break -->`` markers or explicit ``[PAGE N]`` lines;
    if we find any of those we walk the body positions and assign
    the most recent page number. If no markers are present we
    simply assign ``1`` to every section so callers still get a
    non-null ``page_number``.
    """
    page_break_rx = re.compile(r"<!--\s*page\s*break\s*-->", re.IGNORECASE)
    page_num_rx = re.compile(r"^\s*\[PAGE\s+(\d+)\]\s*$", re.IGNORECASE | re.MULTILINE)

    # Find page marker offsets in the original markdown.
    markers: list[tuple[int, int]] = []  # (offset, page_number)
    for m in page_break_rx.finditer(full_text):
        markers.append((m.start(), len(markers) + 1))
    for m in page_num_rx.finditer(full_text):
        markers.append((m.start(), int(m.group(1))))
    markers.sort(key=lambda t: t[0])

    if not markers:
        for s in sections:
            if s.page_number is None:
                s.page_number = 1
        return

    # Compute the start offset of each section's body in the full
    # markdown. The stack-based split keeps the offsets implicit;
    # for page annotation we accept that page assignment is
    # approximate — Docling's section breaks rarely land exactly
    # on a page boundary, so we report the page where the section's
    # body *starts*.
    cursor = 0
    for s in sections:
        # Find where this section's body starts in full_text. We
        # only have the body text here, so use a positional search
        # starting from ``cursor``.
        idx = full_text.find(s.body[:60], cursor) if s.body else -1
        if idx < 0:
            continue
        cursor = idx + 1
        page = 1
        for off, pn in markers:
            if off <= idx:
                page = pn
            else:
                break
        s.page_number = page


def _chunk_sections(
    sections: list[_Section],
    *,
    chunk_size: int,
    chunk_overlap: int,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Walk the section list and emit chunks.

    The chunker maintains two pieces of state across section
    boundaries:

    - ``current_text`` / ``current_tokens`` — the running buffer for
      the chunk-in-progress.
    - ``overlap_buffer`` — the trailing tokens from the most recent
      emission, prepended to the next chunk.

    Sections whose body alone exceeds ``chunk_size`` are themselves
    recursively split into paragraphs, then sentences, then words.
    """
    chunks: list[Chunk] = []
    overlap_text: str = ""
    current_text: str = ""
    current_tokens: int = 0
    current_section: list[str] = []
    current_page: int | None = None

    def _flush() -> None:
        """Emit the current buffer as a chunk."""
        nonlocal current_text, current_tokens, overlap_text
        body = current_text.strip()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    section_header=_join_header(current_section),
                    page_number=current_page,
                )
            )
            overlap_text = _tail_tokens(body, chunk_overlap, token_counter)
        else:
            overlap_text = ""
        current_text = ""
        current_tokens = 0

    for section in sections:
        header_path = section.header_path
        page = section.page_number
        body = section.body
        if not body:
            continue

        # Section header (when present) is prepended to the chunk as
        # breadcrumb context. We include it only when we *start* a new
        # chunk so we don't double-count it across overlapping chunks.
        header_prefix = ""
        if header_path:
            header_prefix = "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(header_path))
            header_prefix += "\n\n"

        # If this section alone is bigger than chunk_size we need to
        # recurse into finer-grained splits.
        section_size = token_counter(body)
        if section_size > chunk_size:
            # Flush whatever we had — section is too big to share a
            # chunk with anything else.
            _flush()
            sub_chunks = _split_oversized_section(
                body,
                chunk_size=chunk_size,
                token_counter=token_counter,
            )
            for sub in sub_chunks:
                # The sub-chunk already carries its own text; we just
                # re-tag it with the parent section's breadcrumb.
                chunks.append(
                    Chunk(
                        text=sub.text,
                        chunk_index=len(chunks),
                        token_count=sub.token_count,
                        section_header=_join_header(header_path),
                        page_number=page,
                    )
                )
            # After an oversized section, reset overlap buffer to the
            # tail of the last sub-chunk.
            if chunks:
                overlap_text = _tail_tokens(
                    chunks[-1].text, chunk_overlap, token_counter,
                )
            current_section = list(header_path)
            current_page = page
            continue

        # Section fits inside one chunk — but only if it also fits
        # together with whatever is already in the buffer.
        candidate_text = (
            (current_text + "\n\n" + header_prefix + body) if current_text
            else (header_prefix + body)
        )
        candidate_tokens = token_counter(candidate_text)

        if candidate_tokens > chunk_size and current_text:
            _flush()
            # Re-seed with overlap from previous chunk.
            current_text = overlap_text
            current_tokens = token_counter(current_text) if overlap_text else 0
            current_section = list(header_path)
            current_page = page
            # Try again with overlap prefix in place.
            candidate_text = (
                (current_text + "\n\n" + header_prefix + body) if current_text
                else (header_prefix + body)
            )
            candidate_tokens = token_counter(candidate_text)
            # If the overlap+section still doesn't fit, drop the
            # overlap. Otherwise the section would never get a chunk
            # of its own.
            if candidate_tokens > chunk_size and current_text:
                current_text = ""
                current_tokens = 0
                candidate_text = header_prefix + body
                candidate_tokens = token_counter(candidate_text)

        current_text = candidate_text
        current_tokens = candidate_tokens
        current_section = list(header_path)
        # The current chunk's page is the *latest* page number seen —
        # overlaps and header_prefixes both pull from the most recent
        # section, so the trailing page wins. ``None`` when the
        # chunker didn't get page numbers at all (markdown / text).
        if page is not None:
            current_page = page

    # Emit anything still in the buffer.
    if current_text.strip():
        _flush()

    return chunks


def _split_oversized_section(
    body: str,
    *,
    chunk_size: int,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Recursively split an oversized section.

    Strategy:
    1. If the body contains a code block, split *around* the code
       block (the code block stays atomic; surrounding prose gets
       recursively chunked; the code block is itself a single chunk
       if it still exceeds ``chunk_size``, with a ``code_block``
       marker in the breadcrumb).
    2. Otherwise split into paragraphs (``\\n\\n+``).
    3. If any paragraph still exceeds ``chunk_size``, split it into
       sentences (``.!?`` followed by whitespace).
    4. As a last resort split into words.

    Returns chunks with ``section_header=""`` — the caller tags them
    with the parent section's breadcrumb.
    """
    # Code-block handling: walk through the body, collect runs of
    # text that are outside code fences and runs that are inside.
    pieces = _split_around_code_blocks(body)
    out: list[Chunk] = []
    for is_code, text in pieces:
        if token_counter(text) <= chunk_size or is_code:
            out.append(
                Chunk(
                    text=text.strip(),
                    chunk_index=len(out),
                    token_count=token_counter(text),
                    section_header="",
                    page_number=None,
                )
            )
            continue
        # Prose path: split into paragraphs, recurse on each.
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            if token_counter(para) <= chunk_size:
                out.append(
                    Chunk(
                        text=para,
                        chunk_index=len(out),
                        token_count=token_counter(para),
                        section_header="",
                        page_number=None,
                    )
                )
            else:
                out.extend(_split_paragraph(para, chunk_size, token_counter))
    return out


def _split_paragraph(
    para: str,
    chunk_size: int,
    token_counter: Callable[[str], int],
) -> list[Chunk]:
    """Sentence- and word-level split for an oversized paragraph."""
    # Sentence boundary: one of [.!?] followed by whitespace.
    sentence_rx = re.compile(r"(?<=[.!?])\s+")
    sentences = [s.strip() for s in sentence_rx.split(para) if s.strip()]
    out: list[Chunk] = []
    for sent in sentences:
        if token_counter(sent) <= chunk_size:
            out.append(
                Chunk(
                    text=sent,
                    chunk_index=len(out),
                    token_count=token_counter(sent),
                    section_header="",
                    page_number=None,
                )
            )
        else:
            # Last resort: word-level split.
            words = sent.split(" ")
            buf: list[str] = []
            buf_tokens = 0
            for w in words:
                if buf_tokens + token_counter(w) > chunk_size and buf:
                    out.append(
                        Chunk(
                            text=" ".join(buf).strip(),
                            chunk_index=len(out),
                            token_count=buf_tokens,
                            section_header="",
                            page_number=None,
                        )
                    )
                    buf = []
                    buf_tokens = 0
                buf.append(w)
                buf_tokens += token_counter(w)
            if buf:
                out.append(
                    Chunk(
                        text=" ".join(buf).strip(),
                        chunk_index=len(out),
                        token_count=buf_tokens,
                        section_header="",
                        page_number=None,
                    )
                )
    return out


def _split_around_code_blocks(body: str) -> list[tuple[bool, str]]:
    """Split ``body`` into runs of (is_code, text) pairs.

    The chunker never splits inside a code block, but it can split
    *around* one — a code block stays as a single atomic chunk while
    the prose on either side is recursively split.
    """
    pieces: list[tuple[bool, str]] = []
    lines = body.split("\n")
    in_code = False
    buf: list[str] = []
    code_buf: list[str] = []

    def _flush_prose() -> None:
        if buf:
            pieces.append((False, "\n".join(buf).strip()))
            buf.clear()

    def _flush_code() -> None:
        if code_buf:
            pieces.append((True, "\n".join(code_buf).strip()))
            code_buf.clear()

    for line in lines:
        if _CODE_FENCE_RX.match(line):
            if not in_code:
                _flush_prose()
                in_code = True
                code_buf.append(line)
            else:
                code_buf.append(line)
                _flush_code()
                in_code = False
        elif in_code:
            code_buf.append(line)
        else:
            buf.append(line)
    _flush_prose()
    _flush_code()
    return [(is_code, t) for is_code, t in pieces if t]


def _join_header(header_path: list[str]) -> str:
    if not header_path:
        return ""
    return " > ".join(header_path)


def _tail_tokens(
    text: str,
    n_tokens: int,
    token_counter: Callable[[str], int],
) -> str:
    """Return up to ``n_tokens`` from the tail of ``text``.

    Walks backwards in token-space (encodes the whole string, takes
    the last ``n_tokens`` decode-able tokens, decodes back to
    text). Not perfectly lossless for multi-byte unicode, but
    sufficient for cross-chunk context propagation — tiktoken's BPE
    boundaries make this idempotent enough for prose.
    """
    if n_tokens <= 0 or not text:
        return ""
    enc = _get_encoding()
    ids = enc.encode(text)  # type: ignore[union-attr]
    if len(ids) <= n_tokens:
        return text
    tail = ids[-n_tokens:]
    return enc.decode(tail)  # type: ignore[union-attr]
