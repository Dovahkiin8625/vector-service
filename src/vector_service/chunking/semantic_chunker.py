"""Semantic chunker.

Embeds every sentence with an injected embedder and cuts at the
largest adjacent-sentence semantic distances — the percentile
breakpoint method popularised by LangChain's ``SemanticChunker``:

1. split each header section into sentences;
2. embed the sentences in one batch (``embed_fn`` — in this service
   the already-loaded bge-m3, so no extra model / dependency);
3. compute cosine distance between consecutive sentences and take
   the ``breakpoint_percentile``-th percentile as the cut
   threshold — only the most pronounced topic shifts become
   boundaries;
4. each resulting sentence group is packed to ``chunk_size`` (hard
   sentence-bound cap as fallback).

Per the 2025 evaluations semantic segmentation helps most on
documents with clear topic shifts; on uniform prose it converges
to ordinary size packing, so the downside is bounded to the extra
embedding cost.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from vector_service.chunking.base import BaseChunker, Chunk, register_strategy
from vector_service.chunking.structure import (
    header_prefix,
    join_header,
    merge_small_chunks,
    prepare_sections,
    split_sentences,
    pack_units,
)


@register_strategy("semantic")
class SemanticChunker(BaseChunker):
    """Embedding-similarity chunker.

    Args:
        embed_fn: ``list[str] -> list[vector]`` batch embedder.
        breakpoint_percentile: Percentile (``(0, 100]``) of the
            adjacent-distance distribution at/above which a cut is
            made. ``95`` = split only at the 5% biggest distances.
    """

    def __init__(
        self,
        *args,
        embed_fn=None,
        breakpoint_percentile: float = 95.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if embed_fn is None:
            raise ValueError("SemanticChunker requires an embed_fn")
        if not 0 < breakpoint_percentile <= 100:
            raise ValueError(
                "breakpoint_percentile must be in (0, 100], got "
                f"{breakpoint_percentile}"
            )
        self._embed_fn = embed_fn
        self._percentile = float(breakpoint_percentile)

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Chunk ``markdown`` at semantic breakpoints. Empty → ``[]``."""
        if not markdown or not markdown.strip():
            return []

        sections = prepare_sections(markdown, page_numbers)
        out: list[Chunk] = []
        for section in sections:
            if not section.body:
                continue
            out.extend(
                self._chunk_section(
                    section.body,
                    section_header=join_header(section.header_path),
                    page_number=section.page_number,
                    prepend_header=(
                        header_prefix(section.header_path)
                        if section.header_path else ""
                    ),
                )
            )
        return merge_small_chunks(
            out,
            chunk_size=self.chunk_size,
            token_counter=self.token_counter,
            min_size=self.min_chunk_size,
            separator=" ",
        )

    def _chunk_section(
        self,
        body: str,
        *,
        section_header: str,
        page_number: int | None,
        prepend_header: str,
    ) -> list[Chunk]:
        sentences = split_sentences(body)
        if not sentences:
            return []

        groups = self._semantic_groups(sentences)
        out: list[Chunk] = []
        for group in groups:
            if prepend_header and not out:
                group[0] = prepend_header + group[0]
            out.extend(
                pack_units(
                    group,
                    chunk_size=self.chunk_size,
                    token_counter=self.token_counter,
                    section_header=section_header,
                    page_number=page_number,
                    separator=" ",
                    overlap=self.chunk_overlap,
                )
            )
        return out

    def _semantic_groups(self, sentences: list[str]) -> list[list[str]]:
        """Group sentences by the percentile-distance rule."""
        if len(sentences) == 1:
            return [sentences]

        vectors = self._embed_fn(sentences)
        distances = _adjacent_distances(vectors)
        if not distances:
            return [sentences]

        threshold = _percentile(distances, self._percentile)
        groups = [[sentences[0]]]
        for i, sentence in enumerate(sentences[1:], start=1):
            # Strictly greater: uniform distances (all 0) stay one
            # group even though threshold is also 0; percentile=100
            # (threshold = max) similarly produces no cuts.
            if distances[i - 1] > threshold:
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        return groups


def _adjacent_distances(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Cosine distance between each consecutive vector pair."""
    distances: list[float] = []
    prev = _normalize(vectors[0])
    for vec in vectors[1:]:
        cur = _normalize(vec)
        similarity = sum(a * b for a, b in zip(prev, cur))
        # Numerical clamp; identical vectors → distance 0.
        distances.append(max(0.0, 1.0 - similarity))
        prev = cur
    return distances


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy ``linear`` convention)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    rank = (p / 100) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + frac * (ordered[high] - ordered[low])
