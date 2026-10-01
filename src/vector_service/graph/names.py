"""Physical collection naming for derived graph layers.

Lives below both the retrieval pipeline and the graph API so either can
use it without an import cycle: ``api.graph`` → ``graph`` →
``retrieval.transforms`` → ``retrieval`` → ``pipeline`` must never loop
back into the API layer.
"""
from __future__ import annotations

import re


def safe_name(collection: str) -> str:
    """Sanitize a logical name into a Milvus-legal token."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", collection)
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def graph_collection_names(logical_collection: str) -> tuple[str, str]:
    """Physical (entity, community) collection names for a logical name."""
    safe = safe_name(logical_collection)
    return f"{safe}__graph_entities", f"{safe}__graph_communities"
