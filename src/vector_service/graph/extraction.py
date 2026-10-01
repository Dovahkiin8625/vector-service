"""LLM extraction of entities, relationships and claims from one chunk.

The extraction prompt returns one JSON object; parsing is deliberately
tolerant (fences, prose, missing keys, malformed entries) and a failed
call degrades to an empty :class:`ChunkExtraction` — one unparseable
chunk must not abort the graph build.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vector_service.core.logging import get_logger
from vector_service.retrieval.transforms import parse_json_object

log = get_logger(__name__)

# Per-chunk caps: a single chunk never dominates the merged graph, and
# the values stay inside typical chat JSON responses.
MAX_ENTITIES = 60
MAX_EDGES = 60
MAX_CLAIMS = 60
MAX_NAME_CHARS = 128
MAX_TEXT_CHARS = 512

_EXTRACTION_SYSTEM = (
    "You are an expert knowledge-graph extraction engine. You identify "
    "entities, the relationships between them, and factual claims. Respond "
    "with JSON only, in the same language as the input text."
)

_EXTRACTION_INSTRUCTION = (
    'Analyze the text and return ONE JSON object of the exact shape:\n'
    '{\n'
    '  "entities": [{"name": "...", "type": "...", "description": "..."}],\n'
    '  "relationships": [{"source": "...", "target": "...", '
    '"description": "..."}],\n'
    '  "claims": [{"subject": "...", "object": "..." or null, "type": "...", '
    '"status": "true|false|suspect", "statement": "..."}]\n'
    '}\n'
    'Rules: entity names are canonical, short noun phrases; the subject of '
    'every relationship and claim must be an entity you listed; descriptions '
    'and statements are concise factual clauses; emit nothing that is not '
    'supported by the text. Use [] for sections with no entries.'
)


@dataclass
class RawEntity:
    """One entity observed inside a chunk."""

    name: str
    entity_type: str
    description: str


@dataclass
class RawEdge:
    """One directed relationship observed inside a chunk."""

    source: str
    target: str
    description: str


@dataclass
class RawClaim:
    """One factual claim about a subject entity."""

    subject: str
    object: str | None
    claim_type: str
    status: str
    statement: str


@dataclass
class ChunkExtraction:
    """All graph elements extracted from one chunk."""

    entities: list[RawEntity] = field(default_factory=list)
    edges: list[RawEdge] = field(default_factory=list)
    claims: list[RawClaim] = field(default_factory=list)


def _clean(value: object, *, limit: int = MAX_TEXT_CHARS) -> str:
    """Stringify, collapse whitespace and cap length."""
    text = " ".join(str(value).split())
    return text[:limit]


def _clean_name(value: object) -> str:
    return _clean(value, limit=MAX_NAME_CHARS)


def extract_for_chunk(
    chat_fn, text: str, *, entity_types: tuple[str, ...] = ()
) -> ChunkExtraction:
    """Run the extraction prompt for one chunk; never raises.

    ``entity_types`` optionally appends an allowed-type hint to the prompt.
    Any LLM/parse failure returns an empty extraction.
    """
    instruction = _EXTRACTION_INSTRUCTION
    if entity_types:
        hint = ", ".join(entity_types[:20])
        instruction += f"\nPrefer these entity types when applicable: {hint}."
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": f"{instruction}\n\nText:\n{text[:20000]}"},
    ]
    try:
        raw = chat_fn(messages)
        data = parse_json_object(raw)
    except Exception as e:  # noqa: BLE001 — degrade, don't crash
        log.warning("graph_extraction_failed", error=str(e))
        return ChunkExtraction()
    return _build(data)


def _build(data: dict) -> ChunkExtraction:
    """Validate the parsed JSON object into a ChunkExtraction."""
    entities: list[RawEntity] = []
    seen_names: set[str] = set()
    for item in data.get("entities", [])[:MAX_ENTITIES]:
        if not isinstance(item, dict):
            continue
        name = _clean_name(item.get("name", ""))
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        entities.append(
            RawEntity(
                name=name,
                entity_type=_clean(item.get("type", ""), limit=64),
                description=_clean(item.get("description", "")),
            )
        )

    edges: list[RawEdge] = []
    for item in data.get("relationships", [])[:MAX_EDGES]:
        if not isinstance(item, dict):
            continue
        source = _clean_name(item.get("source", ""))
        target = _clean_name(item.get("target", ""))
        if not source or not target:
            continue
        edges.append(
            RawEdge(
                source=source,
                target=target,
                description=_clean(item.get("description", "")),
            )
        )

    claims: list[RawClaim] = []
    for item in data.get("claims", [])[:MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        subject = _clean_name(item.get("subject", ""))
        statement = _clean(item.get("statement", ""))
        if not subject or not statement:
            continue
        obj = _clean_name(item.get("object", ""))
        claims.append(
            RawClaim(
                subject=subject,
                object=obj or None,
                claim_type=_clean(item.get("type", ""), limit=64),
                status=_clean(item.get("status", ""), limit=32),
                statement=statement,
            )
        )

    return ChunkExtraction(entities=entities, edges=edges, claims=claims)
