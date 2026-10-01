"""LLM community summaries for global (community-based) retrieval.

One bounded thread pool calls the chat backend per community; a failed
call leaves the summary empty and the caller embeds the concatenated
member names instead — community retrieval must never fail because one
summary call did.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vector_service.core.logging import get_logger

log = get_logger(__name__)

# Input bounds per community report.
MAX_MEMBER_NAMES = 40
MAX_INPUT_CHARS = 6000

_COMMUNITY_SYSTEM = (
    "You are an expert analyst summarizing a community from a knowledge "
    "graph. You write in the same language as the entity names and "
    "relationships you are given."
)

_COMMUNITY_INSTRUCTION = (
    "Below are the entities and key relationships of one graph community. "
    "Write a concise factual paragraph (3-6 sentences) capturing the "
    "community's main themes, entities and how they relate. Preserve key "
    "names, numbers and facts. Answer with the paragraph only, no "
    "preamble.\n\n"
    "Entities:\n{entities}\n\n"
    "Relationships:\n{relationships}"
)


@dataclass
class CommunitySummaryInput:
    """One community to summarize."""

    community_id: str
    members: list[str]
    edges: list  # BuiltEdge rows inside the community


def _build_prompt(
    item: CommunitySummaryInput, name_for: dict[str, str]
) -> str:
    names = [
        name_for.get(entity_id, entity_id)
        for entity_id in item.members[:MAX_MEMBER_NAMES]
    ]
    lines = []
    used = 0
    for edge in item.edges:
        src = name_for.get(edge.source_id, edge.source_id)
        tgt = name_for.get(edge.target_id, edge.target_id)
        line = f"- {src} → {tgt}"
        if edge.description:
            line += f": {edge.description}"
        if used + len(line) > MAX_INPUT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    entities = ", ".join(names)
    return _COMMUNITY_INSTRUCTION.format(
        entities=entities, relationships="\n".join(lines)
    )


def summarize_communities(
    items: list[CommunitySummaryInput],
    *,
    name_for: dict[str, str],
    chat_fn,
    max_concurrency: int = 4,
) -> dict[str, str]:
    """Return ``community_id -> summary``; failed calls map to ``""``."""
    if not items:
        return {}

    def _one(item: CommunitySummaryInput) -> tuple[str, str]:
        try:
            answer = chat_fn([
                {"role": "system", "content": _COMMUNITY_SYSTEM},
                {"role": "user", "content": _build_prompt(item, name_for)},
            ])
        except Exception as e:  # noqa: BLE001 — optional enrichment
            log.warning(
                "community_summary_failed",
                community_id=item.community_id, error=str(e),
            )
            return item.community_id, ""
        return item.community_id, " ".join(str(answer).split())

    workers = max(1, min(int(max_concurrency), len(items)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_one, items))
