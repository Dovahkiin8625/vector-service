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

from vector_service.chunking.base import Chunk
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
    """

    header_path: list[str] = field(default_factory=list)
    body: str = ""
    page_number: int | None = None


@dataclass
class Unit:
    """A pre-split piece of text awaiting greedy packing."""

    text: str
    page_number: int | None = None


# ---- sections / pages --------------------------------------------------


def split_into_sections(markdown: str) -> list[Section]:
    """Walk the document once and emit one :class:`Section` per header
    scope.

    Headers without a body produce an empty ``Section``; that's fine
    — chunkers simply skip a heading-only section when its body is
    empty.
    """
    matches = list(HEADER_RX.finditer(markdown))
    if not matches:
        return [Section(header_path=[], body=markdown.strip())]

    sections: list[Section] = []
    # Anything before the first header goes into a leading
    # header-less section.
    pre = markdown[: matches[0].start()].strip()
    if pre:
        sections.append(Section(header_path=[], body=pre))

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
            Section(header_path=[t for _lvl, t in stack], body=body)
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


def split_around_code_blocks(body: str) -> list[tuple[bool, str]]:
    """Split ``body`` into runs of (is_code, text) pairs.

    Code blocks stay atomic; prose on either side stays available
    for recursive splitting.
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
        if CODE_FENCE_RX.match(line):
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
) -> list[Chunk]:
    """Greedily pack text units into token-sized chunks.

    A chunk is emitted (at a unit boundary) once adding the next
    unit would cross ``chunk_size``. The next chunk starts with up
    to ``overlap`` tail tokens from the emitted chunk. A single unit
    larger than ``chunk_size`` is emitted on its own rather than
    split — callers pre-split oversized units into sentences/words
    when hard size bounds matter.
    """
    chunks: list[Chunk] = []
    buf: list[str] = []
    prefix = ""  # overlap tail carried into the in-progress chunk

    def _render() -> str:
        parts = ([prefix] if prefix else []) + buf
        return separator.join(parts).strip()

    def _flush() -> None:
        nonlocal buf, prefix
        body = _render()
        if body:
            chunks.append(
                Chunk(
                    text=body,
                    chunk_index=len(chunks),
                    token_count=token_counter(body),
                    section_header=section_header,
                    page_number=page_number,
                )
            )
            prefix = tail_tokens(body, overlap) if overlap else ""
        else:
            prefix = ""
        buf = []

    for raw in units:
        text = raw.text if isinstance(raw, Unit) else raw
        text = text.strip()
        if not text:
            continue

        if buf and token_counter(_render_with(buf, text, prefix, separator)) > chunk_size:
            _flush()

        buf.append(text)

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
    """
    units: list[Unit] = []
    for is_code, piece in split_around_code_blocks(body):
        if is_code:
            units.append(Unit(text=piece))
            continue
        for para in split_paragraphs(piece):
            if token_counter(para) <= chunk_size:
                units.append(Unit(text=para))
            else:
                units.extend(
                    degrade_paragraph(
                        para, chunk_size=chunk_size, token_counter=token_counter
                    )
                )
    return units


def degrade_paragraph(
    para: str,
    *,
    chunk_size: int,
    token_counter,
) -> list[Unit]:
    """Split an oversized paragraph into sentence units.

    A sentence that is itself too long is word-packed as a last
    resort. Whitespace-only input yields no units.
    """
    units: list[Unit] = []
    for sent in split_sentences(para):
        if token_counter(sent) <= chunk_size:
            units.append(Unit(text=sent))
            continue
        buf: list[str] = []
        buf_tokens = 0
        for w in split_words(sent):
            if buf and buf_tokens + token_counter(w) > chunk_size:
                units.append(Unit(text=" ".join(buf)))
                buf = []
                buf_tokens = 0
            buf.append(w)
            buf_tokens += token_counter(w)
        if buf:
            units.append(Unit(text=" ".join(buf)))
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
    kept.
    """
    if not chunks or min_size <= 1:
        return chunks

    out: list[Chunk] = []
    for ch in chunks:
        if out and out[-1].token_count < min_size:
            candidate = out[-1].text + separator + ch.text
            if token_counter(candidate) <= chunk_size:
                out[-1].text = candidate
                out[-1].token_count = token_counter(candidate)
                continue
        out.append(ch)

    # A tiny chunk that couldn't merge forward (previous near cap)
    # merges with the following chunk as it arrives; what remains is
    # a tiny *tail*, absorbed into the previous chunk unconditionally.
    if len(out) >= 2 and out[-1].token_count < min_size:
        prev = out[-2]
        prev.text = prev.text + separator + out[-1].text
        prev.token_count = token_counter(prev.text)
        out.pop()

    for i, c in enumerate(out):
        c.chunk_index = i
    return out
