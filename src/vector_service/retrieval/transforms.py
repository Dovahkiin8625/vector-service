"""LLM query transforms.

All transforms share one contract: an LLM hiccup (non-JSON text, an
empty list, a missing key) degrades to the original query — or to no
rewrite — and is logged, never raised. Retrieval must not depend on the
LLM being well-behaved.
"""
from __future__ import annotations

import json
import logging
import math
import re

from vector_service.retrieval.base import RecallSpec

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are an expert answering questions. Write a short (2-4 sentences), "
    "factual paragraph that directly answers the user's question, as it "
    "would appear in an authoritative document. No preamble, no caveats."
)
_MULTI_SYSTEM = (
    "You rewrite search queries. Return ONLY a JSON object: "
    '{{"queries": ["..."]}} with {n} alternative queries that approach the '
    "user's question from different angles or use different wording."
)
_STEP_BACK_SYSTEM = (
    "You abstract questions. Return ONLY JSON: "
    '{"query": "..."} containing one broader, more fundamental question '
    "that the specific question is an instance of."
)
_DECOMPOSE_SYSTEM = (
    "You break complex multi-hop questions into atomic sub-questions. "
    'Return ONLY JSON: {"sub_queries": ["...", "..."]}. Each sub-question '
    "must be answerable with a single fact."
)


def parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from an LLM response.

    Tolerates ```json fences and prose before/after the object. Raises
    ``ValueError`` when no parseable object exists.
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    start = text.find("{")
    if start >= 0:
        tail = text[start:]
        candidates.append(tail)
        end = tail.rfind("}")
        if end > 0:
            candidates.append(tail[: end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no JSON object in LLM response: {raw[:120]!r}")


def _chat_json(chat_fn, system: str, query: str) -> dict | None:
    try:
        raw = chat_fn([
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ])
        return parse_json_object(raw)
    except Exception as e:  # noqa: BLE001 — any LLM failure degrades
        logger.warning("query transform failed, falling back: %s", e)
        return None


def multi_query(chat_fn, query: str, n: int = 3) -> list[str]:
    """Up to ``n`` reformulations; falls back to ``[query]``."""
    data = _chat_json(chat_fn, _MULTI_SYSTEM.format(n=n), query)
    if not data:
        return [query]
    variants = [
        str(q).strip() for q in data.get("queries", []) if str(q).strip()
    ][:n]
    return variants or [query]


def step_back(chat_fn, query: str) -> str:
    """One broader question; falls back to the original query."""
    data = _chat_json(chat_fn, _STEP_BACK_SYSTEM, query)
    broader = str((data or {}).get("query", "")).strip()
    return broader or query


def decompose(chat_fn, query: str) -> list[str]:
    """Atomic sub-questions; empty list when the LLM cannot decompose."""
    data = _chat_json(chat_fn, _DECOMPOSE_SYSTEM, query)
    if not data:
        return []
    return [
        str(q).strip() for q in data.get("sub_queries", []) if str(q).strip()
    ]


def hyde(chat_fn, query: str, embedder, alpha: float = 0.7) -> RecallSpec:
    """HyDE leg: L2-normalized convex mix of q-vector and answer-vector.

    ``q' = norm((1−α)·q + α·h)`` — the TREC 2025 RAG winning recipe;
    the original wording stays on the spec for trace display.

    Any LLM-side failure (timeout, bad output, missing client) degrades
    to a plain ``RecallSpec(query=query)`` — ``DenseChannel`` then
    embeds the original query itself — and retrieval continues. An
    embedder failure inside the fallback spec still surfaces at the
    channel layer, where dense recall without an embedder is genuinely
    impossible (the API answers with a 503); that failure belongs there.
    """
    try:
        raw_doc = chat_fn([
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": query},
        ])
        doc = (raw_doc or "").strip() or query
        q_vec = embedder.embed_query(query)
        h_vec = embedder.embed_query(doc)
        mix = [(1.0 - alpha) * a + alpha * b for a, b in zip(q_vec, h_vec)]
        norm = math.sqrt(sum(x * x for x in mix)) or 1.0
        return RecallSpec(
            query=query,
            vector=[x / norm for x in mix],
            hypothetical=doc,
        )
    except Exception:  # noqa: BLE001 — LLM-side hiccups never abort retrieval
        logger.warning("HyDE transform failed, using plain query", exc_info=True)
        return RecallSpec(query=query)
