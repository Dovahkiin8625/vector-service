"""Community detection over the merged graph.

Weighted asynchronous label propagation (Raghavan et al.) with
deterministic tie-breaking (highest aggregated edge weight, then the
smallest label) so the same graph always yields the same partition and
the iteration order cannot introduce randomness. Edges are treated as
undirected for clustering.

NetworkX/igraph are deliberately avoided: the algorithm is a few dozen
lines and keeps the dependency surface unchanged.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

# Default and maximum propagation rounds. Label propagation usually
# converges in <10 rounds; the cap bounds worst-case oscillation — the
# final labels still form a valid partition.
DEFAULT_ITERATIONS = 20


def detect_communities(
    entities, edges, *, iterations: int = DEFAULT_ITERATIONS
) -> list[list[str]]:
    """Partition entity ids into communities.

    Communities are returned with members sorted, ordered by size
    descending then smallest member id. Isolated entities form
    singleton communities.
    """
    ids = [entity.entity_id for entity in entities]
    if not ids:
        return []

    weight: dict[frozenset[str], float] = defaultdict(float)
    for edge in edges:
        if edge.source_id == edge.target_id:
            continue
        weight[frozenset((edge.source_id, edge.target_id))] += float(edge.weight)

    neighbors: dict[str, dict[str, float]] = {i: defaultdict(float) for i in ids}
    for pair, w in weight.items():
        a, b = tuple(pair)
        neighbors[a][b] += w
        neighbors[b][a] += w

    labels = {i: i for i in ids}
    order = sorted(ids)
    for _ in range(max(1, iterations)):
        changed = False
        for node in order:
            tally: dict[str, float] = defaultdict(float)
            for nb, w in neighbors[node].items():
                tally[labels[nb]] += w
            if not tally:
                continue
            # Highest vote; deterministic tie: smallest label id.
            best = min(tally, key=lambda l: (-tally[l], l))
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break

    groups: dict[str, list[str]] = defaultdict(list)
    for node in ids:
        groups[labels[node]].append(node)
    out = [sorted(members) for members in groups.values()]
    out.sort(key=lambda members: (-len(members), members[0]))
    return out


def community_id_for(
    database: str, collection: str, members: list[str]
) -> str:
    """Deterministic community id from its member entity ids."""
    digest = hashlib.sha1(
        f"{database}/{collection}/{'|'.join(members)}".encode()
    ).hexdigest()
    return f"cm_{digest[:16]}"
