"""LLM-guided chunking + contextual chunk enrichment.

Two capabilities, one chat backend:

1. :class:`LLMChunker` — LumberChunker-style segmentation.
   Sentences are presented numbered; the LLM returns the positions
   of topic shifts as JSON. Best single-document retrieval method
   in the 2025 taxonomy studies, at the price of one chat call per
   (bounded) sentence window.
2. :func:`contextualize_chunks` — Anthropic contextual retrieval:
   an LLM writes a 1–2 sentence situating context per chunk.
   Anthropic reports a 35% reduction in top-20 retrieval failures;
   failures here never abort the pipeline (the chunk keeps
   ``context=None`` and is embedded as-is).

Chat traffic goes through an OpenAI-compatible ``/chat/completions``
endpoint (:class:`OpenAIChatClient`, httpx). The service itself
does not serve chat; the endpoint is configured via
``VS_LLM__*`` settings.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

from vector_service.chunking.base import BaseChunker, Chunk, register_strategy
from vector_service.chunking.structure import (
    header_prefix,
    join_header,
    merge_small_chunks,
    prepare_sections,
    split_sentences,
    pack_units,
)
from vector_service.chunking.tokens import decode_ids, encode_ids
from vector_service.core.logging import get_logger

log = get_logger(__name__)

ChatFn = "Callable[[Sequence[dict]], str]"

# Default per-window input budget for the segmentation prompt —
# leaves headroom for the answer inside typical 16k+ endpoints.
DEFAULT_INPUT_TOKEN_BUDGET = 12_000

# Whole-document budget inside each contextualization prompt.
DEFAULT_DOC_TOKEN_BUDGET = 12_000


# ---- LLM chunker -------------------------------------------------------


_BOUNDARY_SYSTEM = (
    "You are an expert at deciding where to split text into "
    "topically coherent chunks for a search index. "
    "Respond with JSON only."
)

_BOUNDARY_INSTRUCTION = (
    "Below are numbered sentences from one document section. "
    "Identify the sentence numbers AFTER which a new, topically "
    "different chunk should begin. Prefer fewer, high-confidence "
    "boundaries. Respond exclusively as JSON of the form "
    '{"break_after": [number, ...]}; use [] when the whole '
    "section is one topic."
)


@register_strategy("llm")
class LLMChunker(BaseChunker):
    """LLM-guided topic-boundary chunker.

    Args:
        chat_fn: ``messages -> assistant text`` (sync).
        input_token_budget: Max tokens of numbered sentences per
            chat call; larger sections are windowed.
    """

    def __init__(
        self,
        *args,
        chat_fn=None,
        input_token_budget: int = DEFAULT_INPUT_TOKEN_BUDGET,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if chat_fn is None:
            raise ValueError("LLMChunker requires a chat_fn")
        self._chat_fn = chat_fn
        self._input_budget = int(input_token_budget)

    def chunk(
        self,
        markdown: str,
        *,
        page_numbers: Iterable[int] | None = None,
    ) -> list[Chunk]:
        """Chunk ``markdown`` via LLM-selected boundaries. Empty → ``[]``."""
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

        out: list[Chunk] = []
        for window in _window_sentences(sentences, self._input_budget, self.token_counter):
            break_after = self._ask_boundaries(window)
            groups = _groups_at(window, break_after)
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

    def _ask_boundaries(self, sentences: list[str]) -> set[int]:
        """Call the LLM and return validated break positions."""
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
        messages = [
            {"role": "system", "content": _BOUNDARY_SYSTEM},
            {"role": "user", "content": f"{_BOUNDARY_INSTRUCTION}\n\n{numbered}"},
        ]
        try:
            answer = self._chat_fn(messages)
        except Exception as e:  # noqa: BLE001 — degrade, don't crash
            log.warning("llm_boundary_call_failed", error=str(e))
            return set()
        return _parse_break_indices(answer, len(sentences))


def _parse_break_indices(answer: str, n: int) -> set[int]:
    """Parse ``{"break_after": [..]}`` with regex fallback.

    Returned positions are 1-based sentence numbers, clamped to
    ``[1, n-1]`` (you cannot break after the last sentence).
    """
    numbers: Iterable[int] = []
    try:
        data = json.loads(answer)
        if isinstance(data, dict):
            raw = data.get("break_after", [])
            if isinstance(raw, list):
                numbers = raw
    except (json.JSONDecodeError, TypeError):
        numbers = re.findall(r"\d+", answer)

    valid: set[int] = set()
    for x in numbers:
        try:
            idx = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n - 1:
            valid.add(idx)
    return valid


def _groups_at(sentences: list[str], break_after: Iterable[int]) -> list[list[str]]:
    """Split sentences into groups at the given 1-based positions."""
    cuts = set(break_after)
    if not cuts:
        return [list(sentences)]
    groups: list[list[str]] = [[]]
    for i, sentence in enumerate(sentences, start=1):
        groups[-1].append(sentence)
        if i in cuts and i < len(sentences):
            groups.append([])
    return [g for g in groups if g]


def _window_sentences(
    sentences: list[str],
    budget: int,
    token_counter,
) -> list[list[str]]:
    """Split sentences into prompt windows under ``budget`` tokens."""
    windows: list[list[str]] = []
    current: list[str] = []
    used = 0
    for sentence in sentences:
        size = token_counter(sentence)
        if current and used + size > budget:
            windows.append(current)
            current = []
            used = 0
        current.append(sentence)
        used += size
        # Single oversized sentence still goes through (the LLM
        # call may be long but ordering is preserved).
    if current:
        windows.append(current)
    return windows


# ---- contextual retrieval ---------------------------------------------


_CONTEXTUAL_INSTRUCTION = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the overall "
    "document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short, succinct context to situate this chunk "
    "within the overall document, for the purpose of improving "
    "search retrieval of the chunk. Answer only with the short "
    "context and nothing else; do not repeat the chunk."
)


def contextualize_chunks(
    chunks: list[Chunk],
    *,
    document: str,
    chat_fn,
    max_concurrency: int = 4,
    doc_token_budget: int = DEFAULT_DOC_TOKEN_BUDGET,
) -> list[Chunk]:
    """Fill ``context`` on each chunk in place.

    Runs the per-chunk calls in a bounded thread pool. Any error
    (timeout, bad response, ...) on one chunk leaves its context
    ``None`` — the caller embeds it unchanged instead of failing
    the whole document.
    """
    if not chunks:
        return chunks

    doc = _truncate(document, doc_token_budget)

    def _one(chunk: Chunk) -> None:
        prompt = _CONTEXTUAL_INSTRUCTION.format(document=doc, chunk=chunk.text)
        try:
            answer = chat_fn([{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001 — enrichment is optional
            log.warning("contextualize_call_failed", error=str(e))
            return
        answer = (answer or "").strip()
        if answer:
            chunk.context = answer

    workers = max(1, min(int(max_concurrency), len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_one, chunks))
    return chunks


def _truncate(text: str, token_budget: int) -> str:
    """Truncate ``text`` in token space, best-effort."""
    try:
        ids = encode_ids(text)
    except Exception:  # noqa: BLE001 — fallback to a hard char cut
        return text[: token_budget * 4]
    if len(ids) <= token_budget:
        return text
    return decode_ids(ids[:token_budget])


def embed_text(chunk: Chunk) -> str:
    """Return the text to embed for ``chunk`` (context prefix if set)."""
    if chunk.context:
        return f"{chunk.context}\n\n{chunk.text}"
    return chunk.text


# ---- OpenAI-compatible chat client -------------------------------------


class OpenAIChatClient:
    """Minimal sync OpenAI-compatible chat client over httpx."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self._timeout = float(timeout)

    def complete(self, messages: Sequence[dict], *, model: str | None = None) -> str:
        """Call ``/chat/completions`` and return the message content."""
        import httpx

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": model or self.model,
            "messages": list(messages),
            "temperature": 0,
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def as_chat_fn(self):
        """Return a bound ``messages -> str`` callable."""
        return self.complete


_CACHED_CLIENT: OpenAIChatClient | None = None


def is_llm_configured(settings) -> bool:
    """Whether ``settings.llm`` carries enough to build a client."""
    llm = getattr(settings, "llm", None)
    return bool(llm is not None and llm.base_url and llm.model)


def get_chat_client(settings) -> OpenAIChatClient:
    """Build (and cache) the chat client from ``settings.llm``.

    Raises ``ValueError`` when LLM settings are absent.
    """
    global _CACHED_CLIENT
    if not is_llm_configured(settings):
        raise ValueError("LLM is not configured (VS_LLM__BASE_URL / VS_LLM__MODEL)")
    if _CACHED_CLIENT is None:
        llm = settings.llm
        secret = llm.api_key
        api_key = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
        _CACHED_CLIENT = OpenAIChatClient(
            base_url=llm.base_url,
            api_key=api_key or "",
            model=llm.model,
            timeout=getattr(llm, "timeout_seconds", 60.0),
        )
    return _CACHED_CLIENT
