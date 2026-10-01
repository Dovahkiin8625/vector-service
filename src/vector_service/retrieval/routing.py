"""Query-intent routing: pick routes by what the query looks like.

Four routes share the index surface:

- **keyword** — quoted phrases, error codes, hashes, paths, version
  numbers, identifiers: lexical BM25 is the strong leg;
- **semantic** — natural-language questions: dense ANN (and the summary
  leg when the collection has one);
- **graph** — relationship / "who is connected" questions: the
  GraphRAG entity/community stage;
- **metadata** — inline predicates (``author:…``, ``page>=10`` …)
  compiled onto the thin scalars.

Two routers share one output shape. :func:`heuristic_route` is
deterministic and free. :func:`llm_route` makes one JSON classification
call and returns ``None`` on any failure — the caller degrades to the
heuristic decision, so intent routing never blocks retrieval.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from vector_service.retrieval.base import MetaPredicate

INTENTS = ("keyword", "semantic", "graph", "metadata")

# documents-table text columns resolved in the corpus → doc_id lists.
TEXT_DOC_FIELDS = ("filename", "title", "author", "status", "mime")
# documents-table numeric columns.
NUM_DOC_FIELDS = ("page_count",)
# Thin-index scalars addressed directly by a Milvus expression.
THIN_DOC_ID = "doc_id"
THIN_CHUNK_INDEX = "chunk_index"

_FIELD_ALIASES = {
    "file": "filename",
    "filename": "filename",
    "title": "title",
    "author": "author",
    "status": "status",
    "mime": "mime",
    "page": "page_count",
    "pages": "page_count",
    "page_count": "page_count",
    "doc_id": "doc_id",
    "chunk": "chunk_index",
    "chunk_index": "chunk_index",
}

_OPS = ("like", "==", "!=", ">=", "<=", ">", "<")

_PREDICATE_RE = re.compile(
    r"(?P<key>[A-Za-z_]+)\s*(?P<op>>=|<=|!=|[:=><])\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|\S+)"
)

_QUOTED_RE = re.compile(r"[\"']([^\"']+)[\"']")
_ERROR_CODE_RE = re.compile(r"\b[A-Z]{2,}-\d+\b")
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
_PATH_RE = re.compile(r"[\w.-]*[\w][\\/][\w\\/.-]+")
_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,}\b")
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+){1,}\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")

_CJK_RE = re.compile(r"[一-鿿]")
_GRAPH_RE = re.compile(
    r"关系|之间|联系|关联|往来|合作|认识|谁和谁|谁跟谁|谁与谁|"
    r"relationship|relation|connected|related|associate",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"[?？]|怎么|如何|为什么|为何|啥|什么|哪些|哪个|哪里|哪儿|吗|呢|谁|"
    r"\bwhat\b|\bwhy\b|\bhow\b|\bwhen\b|\bwhere\b|\bwhich\b|\bwho\b|"
    r"\bshould\b|\bcan\b|\bis\b|\bare\b|\bdoes\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    """The router output, before availability gating by the pipeline."""

    router: str
    intents: list[str]
    signals: list[str]
    dense: bool
    bm25: bool
    summary: bool
    graph: bool
    predicates: list[MetaPredicate]
    query: str


def _add(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _extract_predicates(query: str) -> tuple[str, list[MetaPredicate]]:
    """Pull ``key op value`` predicates out of the query surface.

    Returns the query with matched spans removed. Predicates on unknown
    fields, with non-numeric values on numeric fields, or with empty
    values are left in the query text (and ignored).
    """
    predicates: list[MetaPredicate] = []
    spans: list[tuple[int, int]] = []

    for match in _PREDICATE_RE.finditer(query):
        field = _FIELD_ALIASES.get(match.group("key").lower())
        if field is None:
            continue
        raw_op = match.group("op")
        value = _unquote(match.group("value")).strip()
        if not value:
            continue
        if field in NUM_DOC_FIELDS or field == THIN_CHUNK_INDEX:
            if raw_op in (":", "="):
                op = "=="
            else:
                op = raw_op
            if op not in _OPS or not re.fullmatch(r"-?\d+", value):
                continue
        elif raw_op in (":", "="):
            op = "like"
        else:
            op = raw_op
            if op not in _OPS:
                continue
        predicates.append(MetaPredicate(field=field, op=op, value=value))
        spans.append(match.span())

    if not spans:
        return query.strip(), predicates

    cleaned_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        cleaned_parts.append(query[cursor:start])
        cursor = end
    cleaned_parts.append(query[cursor:])
    return " ".join("".join(cleaned_parts).split()), predicates


def _keyword_signals(query: str) -> tuple[list[str], bool]:
    """Return keyword signals and whether any is route-exclusive.

    Code-like surfaces (error codes, hashes, paths, versions, compound
    identifiers) are strong enough to mute dense ANN; short keyword
    phrasing only adds the BM25 leg.
    """
    signals: list[str] = []
    exclusive = False

    checks = (
        ("quoted_phrase", _QUOTED_RE, False),
        ("error_code", _ERROR_CODE_RE, True),
        ("hex_hash", _HEX_RE, True),
        ("path", _PATH_RE, True),
        ("version", _VERSION_RE, True),
        ("camel_case", _CAMEL_RE, True),
        ("snake_case", _SNAKE_RE, True),
    )
    for name, pattern, strong in checks:
        match = pattern.search(query)
        if match:
            _add(signals, name)
            exclusive = exclusive or strong

    stripped = _QUOTED_RE.sub(r"\1", query).strip()
    has_cjk = bool(_CJK_RE.search(stripped))
    words = stripped.split()
    if not has_cjk:
        if words and len(words) <= 3 and stripped.isascii():
            _add(signals, "short_ascii_terms")
    elif len(stripped) <= 6:
        _add(signals, "short_cjk_terms")
    return signals, exclusive


def _decision(
    *,
    router: str,
    intents: list[str],
    signals: list[str],
    predicates: list[MetaPredicate],
    query: str,
    exclusive_keyword: bool,
) -> RouteDecision:
    """Map classified intents onto channel booleans.

    Strong code-like keyword signals run BM25 alone; weak keyword
    phrasing keeps the dense leg as a safety net. Metadata alone does
    not pick a content channel — the hybrid default applies. The
    decision always ends with at least one recall channel enabled.
    """
    dense = False
    bm25 = False
    summary = False
    graph = "graph" in intents

    if "keyword" in intents:
        bm25 = True
        if not exclusive_keyword:
            dense = True
    if "semantic" in intents:
        dense = True
        summary = True

    if predicates:
        _add(intents, "metadata")
    if not (dense or bm25):
        dense = True
        bm25 = True
        _add(signals, "hybrid_fallback")
    if summary and not dense:
        dense = True

    return RouteDecision(
        router=router,
        intents=intents,
        signals=signals,
        dense=dense,
        bm25=bm25,
        summary=summary,
        graph=graph,
        predicates=predicates,
        query=query,
    )


def heuristic_route(query: str) -> RouteDecision:
    """Classify a query from deterministic surface signals."""
    cleaned, predicates = _extract_predicates(query)
    intents: list[str] = []
    signals: list[str] = []

    keyword_signals, exclusive = _keyword_signals(cleaned)
    for signal in keyword_signals:
        _add(signals, signal)
    if keyword_signals:
        _add(intents, "keyword")

    if _GRAPH_RE.search(cleaned):
        _add(intents, "graph")
        _add(signals, "relation_phrase")

    is_question = bool(_QUESTION_RE.search(cleaned))
    long_cjk = len(_CJK_RE.findall(cleaned)) >= 10
    if is_question or long_cjk:
        _add(intents, "semantic")
        if is_question:
            _add(signals, "question_form")
        if long_cjk:
            _add(signals, "long_cjk_text")

    return _decision(
        router="heuristic",
        intents=intents,
        signals=signals,
        predicates=predicates,
        query=cleaned,
        exclusive_keyword=exclusive,
    )


_LLM_PROMPT = """You classify one retrieval query for a multi-route index.

Routes:
- keyword: exact terms, error codes, hashes, paths, identifiers, quoted phrases;
- semantic: natural-language questions answered by meaning;
- graph: questions about relationships / connections between things;
- metadata: structured conditions on documents or chunks.

Answer with one JSON object and nothing else:
{
  "intents": ["keyword" | "semantic" | "graph" | "metadata"],
  "filters": [
    {"field": "filename|title|author|status|mime|page_count|doc_id|chunk_index",
     "op": "like|==|!=|>=|<=|>|<", "value": "string"}
  ],
  "query": "the query with metadata filters removed"
}

Use "like" for text equality phrasing. Empty filters when none apply.
"""


def _parse_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _llm_predicates(raw: object) -> list[MetaPredicate] | None:
    if not isinstance(raw, list):
        return None
    predicates: list[MetaPredicate] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        field = str(item.get("field", "")).strip().lower()
        op = str(item.get("op", "")).strip()
        value = str(item.get("value", "")).strip()
        if field not in (*TEXT_DOC_FIELDS, *NUM_DOC_FIELDS,
                         THIN_DOC_ID, THIN_CHUNK_INDEX):
            return None
        if op not in _OPS or not value:
            return None
        if (field in NUM_DOC_FIELDS or field == THIN_CHUNK_INDEX) and (
            not re.fullmatch(r"-?\d+", value)
        ):
            return None
        predicates.append(MetaPredicate(field=field, op=op, value=value))
    return predicates


def llm_route(chat_fn, query: str) -> RouteDecision | None:
    """Classify via one LLM JSON call; ``None`` on any failure.

    The caller is expected to fall back to :func:`heuristic_route`.
    """
    try:
        raw_text = chat_fn([
            {"role": "system", "content": _LLM_PROMPT},
            {"role": "user", "content": query},
        ])
    except Exception:  # noqa: BLE001 — routing must not break retrieval
        return None
    parsed = _parse_json_object(raw_text)
    if parsed is None:
        return None

    raw_intents = parsed.get("intents")
    if not isinstance(raw_intents, list) or not raw_intents:
        return None
    intents = [
        str(value).strip().lower()
        for value in raw_intents
        if str(value).strip().lower() in INTENTS
    ]
    intents = list(dict.fromkeys(intents))
    if not intents:
        return None

    predicates = _llm_predicates(parsed.get("filters", []))
    if predicates is None:
        return None

    cleaned = str(parsed.get("query") or query).strip() or query.strip()
    decision = _decision(
        router="llm",
        intents=intents,
        signals=["llm_classified"],
        predicates=predicates,
        query=cleaned,
        exclusive_keyword="keyword" in intents and "semantic" not in intents,
    )
    return decision
