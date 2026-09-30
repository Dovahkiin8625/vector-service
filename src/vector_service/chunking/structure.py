"""Shared document-structure utilities.

Extracted verbatim (behaviour-wise) from the original
``recursive_chunker`` so every strategy shares one implementation
of:

- markdown header tracking (breadcrumb sections),
- page-marker annotation,
- code-fence protection,
- paragraph / sentence splitting,
- greedy unit packing into token-sized chunks.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from vector_service.chunking.base import LEVEL_CHUNK, Chunk
from vector_service.chunking.tokens import tail_tokens

# Header regex: ``#``-prefixed line at the start of a line. The
# leading ``#+`` is group 1, header text group 2. ``re.MULTILINE``
# makes ``^`` match at every line start.
HEADER_RX = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Code-fence detector: matches a triple-backtick line (with optional
# language tag).
CODE_FENCE_RX = re.compile(r"^\s*```.*$", re.MULTILINE)

# Sentence boundary: terminal punctuation (Latin + CJK) followed by
# whitespace, or a CJK full stop/exclamation directly followed by
# the next character (CJK prose has no inter-sentence space).
SENTENCE_BOUNDARY_RX = re.compile(
    r"(?<=[.!?。！？])\s+|(?<=[。！？])(?=\S)"
)


@dataclass
class Section:
    """One header scope in a document.

    ``header_path`` is the ordered list of headers from the document
    root to this section (most recent last). ``body`` is the text
    between this section's header line and the next sibling or
    higher-priority header. ``page_number`` is propagated from page
    markers or the section that contained this one.

    ``ord`` is the dense, document-wide section ordinal (including a
    leading header-less block), used in the ``section:{ord}`` link
    key. ``char_start`` / ``char_end`` are the raw source offsets of
    the section (from its own header to the next header / EOF).
    """

    header_path: list[str] = field(default_factory=list)
    body: str = ""
    page_number: int | None = None
    ord: int | None = None
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class Unit:
    """A pre-split piece of text awaiting greedy packing.

    ``char_start`` / ``char_end`` locate the unit in its source
    string (callers translate to document offsets).
    """

    text: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None


# ---- sections / pages --------------------------------------------------


def split_into_sections(markdown: str) -> list[Section]:
    """Walk the document once and emit one :class:`Section` per header
    scope.

    Headers without a body produce an empty ``Section``; that's fine
    — chunkers simply skip a heading-only section when its body is
    empty. Every emitted section (including a leading header-less
    block) gets a dense ``ord`` and raw source offsets.
    """
    matches = list(HEADER_RX.finditer(markdown))
    if not matches:
        return [
            Section(
                header_path=[], body=markdown.strip(), ord=0,
                char_start=0, char_end=len(markdown),
            )
        ]

    sections: list[Section] = []
    # Anything before the first header goes into a leading
    # header-less section.
    pre = markdown[: matches[0].start()].strip()
    if pre:
        sections.append(
            Section(
                header_path=[], body=pre, ord=0,
                char_start=0, char_end=matches[0].start(),
            )
        )

    # Stack of (header_level, header_text); ``[-1]`` is the current
    # section's level.
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
            Section(
                header_path=[t for _lvl, t in stack],
                body=body,
                ord=len(sections),
                char_start=m.start(),
                char_end=body_end,
            )
        )
    return sections


def annotate_pages(
    sections: list[Section],
    pages: list[int],
    full_text: str,
) -> None:
    """Propagate a best-effort ``page_number`` to each section.

    Pages are represented in Docling's markdown output as
    ``<!-- page break -->`` markers or explicit ``[PAGE N]`` lines;
    if we find any we walk body positions and assign the most recent
    page number. If no markers are present we assign ``1`` to every
    section so callers still get a non-null value.
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
    # markdown. Page assignment is approximate — Docling's section
    # breaks rarely land exactly on a page boundary, so report the
    # page where the section's body *starts*.
    cursor = 0
    for s in sections:
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


def prepare_sections(
    markdown: str,
    page_numbers: Iterable[int] | None = None,
) -> list[Section]:
    """Split into sections and annotate pages in one call.

    Explicit ``page_numbers`` override markers discovered in text
    (currently the explicit list is passed straight through; the
    marker-based path runs when no list is supplied).
    """
    sections = split_into_sections(markdown)
    annotate_pages(sections, list(page_numbers) if page_numbers else [], markdown)
    return sections


# ---- paragraphs / sentences -------------------------------------------


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empties. Does not strip internals."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences; drop empties."""
    return [s.strip() for s in SENTENCE_BOUNDARY_RX.split(text) if s.strip()]


def split_words(text: str) -> list[str]:
    """Whitespace word split (last-resort unit source)."""
    return [w for w in text.split(" ") if w]


# ---- code blocks -------------------------------------------------------


def split_around_code_blocks(
    body: str
) -> list[tuple[bool, str, int, int]]:
    """Split ``body`` into runs of (is_code, text, char_start, char_end).

    Code blocks stay atomic; prose on either side stays available
    for recursive splitting. Offsets are relative to ``body`` and
    point at the stripped piece's exact source position.
    """
    pieces: list[tuple[bool, str, int, int]] = []
    line_spans: list[tuple[int, int]] = []
    cursor = 0
    for line in body.split("\n"):
        line_spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # account for the split-off "\n"
    in_code = False
    buf: list[int] = []
    code_buf: list[int] = []

    def _span_for(indices: list[int]) -> tuple[int, int]:
        start = line_spans[indices[0]][0]
        end = line_spans[indices[-1]][1]
        return start, end

    def _flush_prose() -> None:
        if buf:
            start, end = _span_for(buf)
            text = body[start:end].strip()
            shift = body[start:end].find(text)
            pieces.append((False, text, start + shift, start + shift + len(text)))
            buf.clear()

    def _flush_code() -> None:
        if code_buf:
            start, end = _span_for(code_buf)
            text = body[start:end].strip()
            shift = body[start:end].find(text)
            pieces.append((True, text, start + shift, start + shift + len(text)))
            code_buf.clear()

    for i, line in enumerate(body.split("\n")):
        if CODE_FENCE_RX.match(line):
            if not in_code:
                _flush_prose()
                in_code = True
                code_buf.append(i)
            else:
                code_buf.append(i)
                _flush_code()
                in_code = False
        elif in_code:
            code_buf.append(i)
        else:
            buf.append(i)
    _flush_prose()
    _flush_code()
    return pieces


# ---- headers -----------------------------------------------------------


def join_header(header_path: list[str]) -> str:
    """Render a header path as a ``" > "`` breadcrumb string."""
    if not header_path:
        return ""
    return " > ".join(header_path)


def header_prefix(header_path: list[str]) -> str:
    """Render the literal markdown headers to prepend to a chunk.

    Levels are reconstructed from position in the path, so a chunk
    seeded with this prefix still shows its hierarchy when read in
    isolation.
    """
    if not header_path:
        return ""
    return (
        "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(header_path))
        + "\n\n"
    )


# ---- greedy packing ----------------------------------------------------


def pack_units(
    units: Iterable[Unit | str],
    *,
    chunk_size: int,
    token_counter,
    section_header: str = "",
    page_number: int | None = None,
    separator: str = "\n\n",
    overlap: int = 0,
    section_ord: int | None = None,
) -> list[Chunk]:
    """Greedily pack text units into token-sized chunks.

    A chunk is emitted (at a unit boundary) once adding the next
    unit would cross ``chunk_size``. The next chunk starts with up
    to ``overlap`` tail tokens from the emitted chunk. A single unit
    larger than ``chunk_size`` is emitted on its own rather than
    split — callers pre-split oversized units into sentences/words
    when hard size bounds matter.

    Every emitted chunk carries ``section_ord`` and a best-effort
    char span: the union of its constituent units' source offsets
    (the overlap prefix is not counted).
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_spans: list[tuple[int, int]] = []
    prefix = ""  # overlap tail carried into the in-progress chunk

    def _render() -> str:
        parts = ([prefix] if prefix else []) + buf
        return separator.join(parts).strip()

    def _flush() -> None:
        nonlocal buf, buf_spans, prefix
        body = _render()
        if body:
            char_start = min(s for s, _e in buf_spans) if buf_spans else None
            char_end = max(e for _s, e in buf_spans) if buf_spans else None
            chunks.append(
                Chunk(
                    text=body,
                    chunk_index=len(chunks),
                    token_count=token_counter(body),
                    section_header=section_header,
                    page_number=page_number,
                    level=LEVEL_CHUNK,
                    section_ord=section_ord,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            prefix = tail_tokens(body, overlap) if overlap else ""
        else:
            prefix = ""
        buf = []
        buf_spans = []

    for raw in units:
        if isinstance(raw, Unit):
            text, span = raw.text, (raw.char_start, raw.char_end)
        else:
            text, span = raw, None
        text = text.strip()
        if not text:
            continue

        if buf and token_counter(_render_with(buf, text, prefix, separator)) > chunk_size:
            _flush()

        buf.append(text)
        if span is not None and span[0] is not None and span[1] is not None:
            buf_spans.append(span)

        # Single unit (plus any overlap prefix) over the cap: drop
        # the overlap first; if it's still too big the unit itself is
        # oversized — emit it alone rather than spinning.
        if token_counter(_render()) > chunk_size:
            if prefix:
                prefix = ""
            if token_counter(_render()) > chunk_size:
                _flush()

    _flush()
    return chunks


def _render_with(buf: list[str], extra: str, prefix: str, separator: str) -> str:
    parts = ([prefix] if prefix else []) + buf + [extra]
    return separator.join(parts).strip()


# ---- unit building / degradation --------------------------------------


def _segments_with_spans(
    pattern: re.Pattern, text: str
) -> list[tuple[str, int, int]]:
    """Split ``text`` at pattern matches, yielding stripped (segment,
    start, end) tuples with exact source offsets."""
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for m in pattern.finditer(text):
        seg = text[cursor:m.start()]
        stripped = seg.strip()
        if stripped:
            shift = seg.find(stripped)
            segments.append(
                (stripped, cursor + shift, cursor + shift + len(stripped))
            )
        cursor = m.end()
    seg = text[cursor:]
    stripped = seg.strip()
    if stripped:
        shift = seg.find(stripped)
        segments.append(
            (stripped, cursor + shift, cursor + shift + len(stripped))
        )
    return segments


# Blank-line separator between paragraphs (mirrors split_paragraphs).
_PARAGRAPH_SEP_RX = re.compile(r"\n\s*\n")

# Word token for last-resort packing.
_WORD_RX = re.compile(r"\S+")


def units_for_section(
    body: str,
    *,
    chunk_size: int,
    token_counter,
) -> list[Unit]:
    """Build packable units for one section body.

    - fenced code blocks stay atomic (even when oversized);
    - paragraphs that fit ``chunk_size`` become one unit each;
    - oversized paragraphs degrade to sentences, then word packs.

    Every unit carries offsets relative to ``body``.
    """
    units: list[Unit] = []
    for is_code, piece, p_start, p_end in split_around_code_blocks(body):
        if is_code:
            units.append(Unit(text=piece, char_start=p_start, char_end=p_end))
            continue
        for para, pr_start, pr_end in _segments_with_spans(
            _PARAGRAPH_SEP_RX, piece
        ):
            start = p_start + pr_start
            end = p_start + pr_end
            if token_counter(para) <= chunk_size:
                units.append(Unit(text=para, char_start=start, char_end=end))
            else:
                units.extend(
                    degrade_paragraph(
                        para,
                        chunk_size=chunk_size,
                        token_counter=token_counter,
                        base=start,
                    )
                )
    return units


def degrade_paragraph(
    para: str,
    *,
    chunk_size: int,
    token_counter,
    base: int = 0,
) -> list[Unit]:
    """Split an oversized paragraph into sentence units.

    A sentence that is itself too long is word-packed as a last
    resort. Whitespace-only input yields no units. ``base`` is the
    source offset of ``para`` in the caller's string.
    """
    units: list[Unit] = []
    for sent, s_start, s_end in _segments_with_spans(SENTENCE_BOUNDARY_RX, para):
        start = base + s_start
        end = base + s_end
        if token_counter(sent) <= chunk_size:
            units.append(Unit(text=sent, char_start=start, char_end=end))
            continue
        words: list[str] = []
        word_spans: list[tuple[int, int]] = []
        buf_tokens = 0

        def _emit_words() -> None:
            nonlocal words, word_spans, buf_tokens
            if words:
                units.append(
                    Unit(
                        text=" ".join(words),
                        char_start=start + word_spans[0][0],
                        char_end=start + word_spans[-1][1],
                    )
                )
            words = []
            word_spans = []
            buf_tokens = 0

        for wm in _WORD_RX.finditer(sent):
            word = wm.group(0)
            if words and buf_tokens + token_counter(word) > chunk_size:
                _emit_words()
            words.append(word)
            word_spans.append((wm.start(), wm.end()))
            buf_tokens += token_counter(word)
        _emit_words()
    return units


# ---- small-chunk merging -----------------------------------------------


def merge_small_chunks(
    chunks: list[Chunk],
    *,
    chunk_size: int,
    token_counter,
    min_size: int,
    separator: str = "\n\n",
) -> list[Chunk]:
    """Absorb undersized chunks into their neighbours.

    Chunks shorter than ``min_size`` tokens carry too little
    information to be useful retrieval hits. Such a chunk is merged
    into the previous chunk while the merge stays within
    ``chunk_size``; if that would overflow, it tries the next chunk
    instead. A tiny *trailing* chunk is absorbed regardless of the
    cap — one slightly-oversized chunk beats an uninformative
    fragment. Metadata (breadcrumb / page) of the dominant side is
    kept; char spans are unioned.
    """
    if not chunks or min_size <= 1:
        return chunks

    def _merge_into(prev: Chunk, ch: Chunk) -> None:
        prev.text = prev.text + separator + ch.text
        prev.token_count = token_counter(prev.text)
        starts = [s for s in (prev.char_start, ch.char_start) if s is not None]
        ends = [e for e in (prev.char_end, ch.char_end) if e is not None]
        if starts:
            prev.char_start = min(starts)
        if ends:
            prev.char_end = max(ends)

    out: list[Chunk] = []
    for ch in chunks:
        # Never merge across section boundaries: every leaf must keep
        # belonging to exactly one section.
        if (
            out
            and out[-1].section_ord == ch.section_ord
            and out[-1].token_count < min_size
        ):
            candidate = out[-1].text + separator + ch.text
            if token_counter(candidate) <= chunk_size:
                _merge_into(out[-1], ch)
                continue
        out.append(ch)

    # A tiny chunk that couldn't merge forward (previous near cap)
    # merges with the following chunk as it arrives; what remains is
    # a tiny *tail*, absorbed into the previous chunk unconditionally
    # (but still never across the section boundary).
    if (
        len(out) >= 2
        and out[-1].section_ord == out[-2].section_ord
        and out[-1].token_count < min_size
    ):
        _merge_into(out[-2], out[-1])
        out.pop()

    for i, c in enumerate(out):
        c.chunk_index = i
    return out
