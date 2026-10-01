"""GraphRAG: extracted knowledge graph with community retrieval.

The graph is a *derived* layer, like the vector index: a graph build job
extracts entities / edges / claims from the leaf chunks via an LLM,
merges them by canonical name, detects communities (label propagation),
summarizes them with the LLM, and indexes entity and community vectors
into independent thin Milvus collections. SQLite owns the graph rows;
Milvus holds only the derived vectors.
"""

from vector_service.graph.communities import (
    community_id_for,
    detect_communities,
)
from vector_service.graph.extraction import (
    ChunkExtraction,
    RawClaim,
    RawEdge,
    RawEntity,
    extract_for_chunk,
)
from vector_service.graph.merge import (
    BuiltClaim,
    BuiltEdge,
    BuiltEntity,
    BuiltGraph,
    GraphBuilder,
    canonical_name,
)
from vector_service.graph.names import (
    graph_collection_names,
    safe_name,
)
from vector_service.graph.summaries import (
    CommunitySummaryInput,
    summarize_communities,
)

__all__ = [
    "BuiltClaim",
    "BuiltEdge",
    "BuiltEntity",
    "BuiltGraph",
    "ChunkExtraction",
    "CommunitySummaryInput",
    "GraphBuilder",
    "RawClaim",
    "RawEdge",
    "RawEntity",
    "canonical_name",
    "community_id_for",
    "detect_communities",
    "extract_for_chunk",
    "graph_collection_names",
    "safe_name",
    "summarize_communities",
]
