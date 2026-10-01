"""GraphRAG build, status and delete administration.

Like the vector index, the knowledge graph is a **derived** layer:
``POST .../graph/build`` submits a job that extracts entities, edges and
claims from the SQLite leaf chunks via an LLM, merges them by canonical
name, detects communities (weighted label propagation), summarizes the
kept communities with the LLM, and indexes entity/community vectors
into independent thin Milvus collections. The worker dispatches the
``graph_build`` job type here (:func:`run_graph_build_pipeline`).

Graph rows in SQLite and the two vector collections are replaced
wholesale on success; no migration or compatibility branches.
"""
from __future__ import annotations

import asyncio
import functools
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.api.ingest import (
    JobCancelled,
    _store_http_error,
    ensure_database,
)
from vector_service.chunking.llm_chunker import get_chat_client
from vector_service.core.errors import (
    BackendError,
    CollectionNotFound,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
)
from vector_service.core.logging import get_logger
from vector_service.core.threadpools import (
    run_in_model,
    run_in_sqlite,
    run_in_store,
)
from vector_service.corpus import (
    JOB_EXTRACTING,
    JOB_GRAPHING,
)
from vector_service.graph import (
    CommunitySummaryInput,
    GraphBuilder,
    community_id_for,
    detect_communities,
    extract_for_chunk,
    graph_collection_names,
    summarize_communities,
)
from vector_service.schemas.graph import (
    CommunityInfo,
    GraphBuildRequest,
    GraphBuildSubmitResponse,
    GraphDeleteResponse,
    GraphStatusResponse,
)
from vector_service.stores.base import FieldSpec, IndexSpec

router = APIRouter(prefix="/v1", tags=["graph"])
log = get_logger(__name__)

_PRIMARY_FIELD = "id"
_VECTOR_FIELD = "vector"

# Length caps when assembling embedding inputs from graph rows.
_ENTITY_EMBED_CHARS = 1024
_EMBED_BATCH = 64


# ---- collection creation ----------------------------------------------


def _create_graph_collection(
    store: Any, database: str, name: str, dim: int
) -> None:
    """Create a thin graph collection: primary id + one dense vector."""
    store.create_collection(
        database=database,
        name=name,
        primary_field=_PRIMARY_FIELD,
        vector_field=FieldSpec(
            name=_VECTOR_FIELD, dtype="float_vector", dim=dim
        ),
        scalar_fields=[
            FieldSpec(
                name=_PRIMARY_FIELD, dtype="varchar",
                is_primary=True, max_length=64,
            )
        ],
        indexes=[
            IndexSpec(
                field_name=_VECTOR_FIELD,
                metric_type="cosine",
                index_type="HNSW",
                params={"M": 16, "efConstruction": 200},
            )
        ],
    )


# ---- blocking worker-thread helpers -----------------------------------


def _extract_batch(
    builder: GraphBuilder,
    chat_fn: Any,
    leaves: list[dict[str, Any]],
    entity_types: tuple[str, ...],
) -> None:
    """Extract + merge one leaf batch (runs in one executor call)."""
    for leaf in leaves:
        extraction = extract_for_chunk(
            chat_fn, leaf["text"], entity_types=entity_types
        )
        builder.add_chunk(leaf["chunk_id"], extraction)


def _entity_embed_input(entity: Any) -> str:
    """Embed ``name`` plus the merged description when present."""
    if entity.description:
        return f"{entity.name}: {entity.description[:_ENTITY_EMBED_CHARS]}"
    return entity.name


@dataclass
class _CommunityRow:
    community_id: str
    community_index: int
    summary: str
    created_ts: float


# ---- build pipeline (worker side) -------------------------------------


async def run_graph_build_pipeline(
    *,
    settings: Any,
    store: Any,
    repo: Any,
    embedder: Any,
    database: str,
    logical_collection: str,
    entity_coll: str,
    community_coll: str,
    entity_types: tuple[str, ...],
    community_iterations: int,
    min_community_size: int,
    include_claims: bool,
    batch_size: int,
    inference_timeout_seconds: float,
    emit: Any,
    job_id: str,
    should_cancel: Any,
) -> None:
    """Build the whole derived graph from the corpus leaves."""
    cleaned = False
    now = time.time()

    async def _cleanup() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        for name in (entity_coll, community_coll):
            try:
                await run_in_store(
                    store.drop_collection, database, name
                )
            except CollectionNotFound:
                pass
            except Exception as e:  # noqa: BLE001 — cleanup best-effort
                log.warning(
                    "graph_cleanup_failed",
                    job_id=job_id, collection=name, error=str(e),
                )

    async def _check_cancel() -> None:
        if await should_cancel():
            await _cleanup()
            await run_in_sqlite(repo.mark_cancelled, job_id)
            raise JobCancelled(job_id)

    try:
        # ---- enumerate leaves + build chat fn --------------------
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_EXTRACTING),
        )
        await emit({"type": "stage", "stage": "extract"})
        leaves = await run_in_sqlite(
            repo.list_leaf_chunks, database, logical_collection
        )
        if not leaves:
            raise HTTPException(
                400,
                detail={
                    "error": {
                        "code": "graph_empty",
                        "message": (
                            f"no leaf chunks in {database!r}/"
                            f"{logical_collection!r}; cannot build graph"
                        ),
                    }
                },
            )
        try:
            chat_client = await asyncio.to_thread(get_chat_client, settings)
        except ValueError as e:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "llm_unavailable",
                        "message": str(e),
                    }
                },
            ) from e
        chat_fn = chat_client.as_chat_fn()

        # ---- LLM extraction per leaf batch -----------------------
        builder = GraphBuilder(database, logical_collection, now=now)
        total = len(leaves)
        for start in range(0, total, batch_size):
            await _check_cancel()
            batch = leaves[start : start + batch_size]
            await asyncio.to_thread(
                _extract_batch, builder, chat_fn, batch, entity_types,
            )
            done = min(start + batch_size, total)
            await run_in_sqlite(
                repo.set_job_progress, job_id, done, total
            )
        built = builder.build()

        # ---- communities: detect + summarize ---------------------
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_GRAPHING),
        )
        await emit({"type": "stage", "stage": "graph"})
        member_sets = await run_in_model(
            functools.partial(
                detect_communities,
                built.entities, built.edges,
                iterations=community_iterations,
            ),
        )
        kept = [
            members for members in member_sets
            if len(members) >= min_community_size
        ]

        name_for = {e.entity_id: e.name for e in built.entities}
        summary_inputs: list[CommunitySummaryInput] = []
        for index, members in enumerate(kept):
            member_set = set(members)
            intra = [
                edge for edge in built.edges
                if edge.source_id in member_set and edge.target_id in member_set
            ]
            cid = community_id_for(database, logical_collection, members)
            summary_inputs.append(
                CommunitySummaryInput(cid, members, intra)
            )
        summaries = await asyncio.to_thread(
            functools.partial(
                summarize_communities,
                summary_inputs,
                name_for=name_for,
                chat_fn=chat_fn,
                max_concurrency=settings.llm.max_concurrency,
            ),
        )
        community_rows = [
            _CommunityRow(
                community_id=item.community_id,
                community_index=index,
                summary=summaries.get(item.community_id, ""),
                created_ts=now,
            )
            for index, item in enumerate(summary_inputs)
        ]

        # ---- embeddings: entities then communities ---------------
        entity_vectors = await _embed_in_batches(
            embedder=embedder,
            items=built.entities,
            text_fn=_entity_embed_input,
            inference_timeout_seconds=inference_timeout_seconds,
            embed_model=embedder.model_name,
        )
        community_texts: list[str] = []
        for row, item in zip(community_rows, summary_inputs):
            if row.summary:
                community_texts.append(row.summary)
            else:
                names = [
                    name_for.get(entity_id, entity_id)
                    for entity_id in item.members
                ]
                community_texts.append(", ".join(names))
        community_vectors = await _embed_in_batches(
            embedder=embedder,
            items=community_texts,
            text_fn=str,
            inference_timeout_seconds=inference_timeout_seconds,
            embed_model=embedder.model_name,
        )

        # ---- replace derived collections, then upsert ------------
        await _check_cancel()
        try:
            await run_in_store(ensure_database, store, database)
            existing = await run_in_store(store.list_collections, database)
            for name in (entity_coll, community_coll):
                if name in existing:
                    await run_in_store(store.drop_collection, database, name)
            await run_in_store(
                _create_graph_collection,
                store, database, entity_coll, embedder.dim,
            )
            await run_in_store(
                _create_graph_collection,
                store, database, community_coll, embedder.dim,
            )
            await run_in_store(
                functools.partial(
                    store.upsert,
                    database, entity_coll,
                    _PRIMARY_FIELD, _VECTOR_FIELD,
                    [e.entity_id for e in built.entities],
                    entity_vectors,
                    [{} for _ in built.entities],
                ),
            )
            if summary_inputs:
                await run_in_store(
                    functools.partial(
                        store.upsert,
                        database, community_coll,
                        _PRIMARY_FIELD, _VECTOR_FIELD,
                        [row.community_id for row in community_rows],
                        community_vectors,
                        [{} for _ in community_rows],
                    ),
                )
        except (BackendError, StoreError) as e:
            raise _store_http_error(e, op="graph_index") from e

        # ---- verify row counts -----------------------------------
        try:
            entity_actual = await run_in_store(
                store.count_rows, database, entity_coll
            )
            community_actual = await run_in_store(
                store.count_rows, database, community_coll
            )
        except (BackendError, StoreError) as e:
            raise _store_http_error(e, op="count_rows") from e
        expected_entities = len(built.entities)
        expected_communities = len(community_rows)
        if int(entity_actual) != expected_entities or int(
            community_actual
        ) != expected_communities:
            raise HTTPException(
                500,
                detail={
                    "error": {
                        "code": "graph_count_mismatch",
                        "message": (
                            f"graph build wrote {entity_actual} entities / "
                            f"{community_actual} communities, expected "
                            f"{expected_entities} / {expected_communities}"
                        ),
                    }
                },
            )

        # ---- persist graph rows, then finish ---------------------
        community_members: list[tuple[str, str]] = []
        for item in summary_inputs:
            for entity_id in item.members:
                community_members.append((item.community_id, entity_id))
        await run_in_sqlite(
            functools.partial(
                repo.replace_graph,
                database,
                logical_collection,
                entities=built.entities,
                entity_mentions=built.entity_mentions,
                edges=built.edges,
                edge_mentions=built.edge_mentions,
                claims=built.claims if include_claims else [],
                communities=community_rows,
                community_members=community_members,
            ),
        )
        await run_in_sqlite(
            functools.partial(
                repo.finish_job,
                job_id,
                chunk_count=expected_entities,
                page_count=None,
                tokens_used=0,
            ),
        )
        log.info(
            "graph_build_done",
            job_id=job_id,
            collection=f"{database}/{logical_collection}",
            entities=expected_entities,
            edges=len(built.edges),
            communities=expected_communities,
        )
    except HTTPException:
        await _cleanup()
        raise
    except JobCancelled:
        raise
    except Exception as e:  # pragma: no cover — safety net
        log.exception("graph_build_pipeline_failed")
        await _cleanup()
        raise HTTPException(
            500,
            detail={
                "error": {
                    "code": "internal_error",
                    "message": str(e) or "internal error",
                }
            },
        ) from e


async def _embed_in_batches(
    *,
    embedder: Any,
    items: list[Any],
    text_fn: Any,
    inference_timeout_seconds: float,
    embed_model: str,
) -> list[list[float]]:
    """Embed items in batches with the inference timeout envelope."""
    vectors: list[list[float]] = []
    for start in range(0, len(items), _EMBED_BATCH):
        batch = items[start : start + _EMBED_BATCH]
        texts = [text_fn(item) for item in batch]
        try:
            batch_vectors = await asyncio.wait_for(
                run_in_model(embedder.embed_documents, texts),
                timeout=inference_timeout_seconds,
            )
        except TimeoutError as e:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "embedder_unavailable",
                        "message": (
                            f"embedder {embed_model!r} did not finish within "
                            f"{inference_timeout_seconds}s"
                        ),
                        "model": embed_model,
                    }
                },
            ) from e
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "embedder_unavailable",
                        "message": str(e) or "embedder unavailable",
                        "model": embed_model,
                    }
                },
            ) from e
        vectors.extend(batch_vectors)
    return vectors


# ---- routes ------------------------------------------------------------


@router.post(
    "/databases/{db}/collections/{coll}/graph/build",
    response_model=GraphBuildSubmitResponse,
    status_code=202,
    summary="Submit a knowledge-graph build job",
    description=(
        "Extract entities / edges / claims from the corpus leaf chunks, "
        "detect and summarize communities, and index entity and "
        "community vectors. All existing graph rows and graph collections "
        "are replaced on success."
    ),
)
async def submit_graph_build(
    db: str, coll: str, body: GraphBuildRequest, request: Request
) -> GraphBuildSubmitResponse:
    repo = request.app.state.corpus
    is_corpus = await run_in_sqlite(
        repo.is_corpus_collection, db, coll
    )
    if not is_corpus:
        raise HTTPException(
            422,
            detail={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        f"{db!r}/{coll!r} is not a corpus-backed logical "
                        "collection; no chunks to extract from"
                    ),
                }
            },
        )

    entity_coll, community_coll = graph_collection_names(coll)
    params = {
        "entity_coll": entity_coll,
        "community_coll": community_coll,
        "entity_types": body.entity_types,
        "community_iterations": body.community_iterations,
        "min_community_size": body.min_community_size,
        "include_claims": body.include_claims,
        "batch_size": body.batch_size,
    }
    job_id = uuid.uuid4().hex
    await run_in_sqlite(
        repo.create_job,
        job_id,
        job_type="graph_build",
        database=db,
        collection=coll,
        filename=None,
        mime=None,
        params_json=json.dumps(params),
        max_attempts=request.app.state.settings.jobs.max_attempts,
    )

    worker = getattr(request.app.state, "job_worker", None)
    if worker is not None and request.app.state.settings.jobs.wake_on_submit:
        worker.wake()

    return GraphBuildSubmitResponse(
        job_id=job_id,
        entity_collection=entity_coll,
        community_collection=community_coll,
    )


@router.get(
    "/databases/{db}/collections/{coll}/graph",
    response_model=GraphStatusResponse,
    summary="Get graph status",
    description="Counts and the detected communities of one collection.",
)
async def get_graph(
    db: str, coll: str, request: Request
) -> GraphStatusResponse:
    entity_coll, community_coll = graph_collection_names(coll)
    stats = await run_in_sqlite(
        request.app.state.corpus.graph_stats, db, coll
    )
    return GraphStatusResponse(
        database=db,
        collection=coll,
        entity_collection=entity_coll,
        community_collection=community_coll,
        entities_count=stats["entities_count"],
        edges_count=stats["edges_count"],
        claims_count=stats["claims_count"],
        communities_count=stats["communities_count"],
        communities=[
            CommunityInfo(
                community_id=row["community_id"],
                community_index=row["community_index"],
                size=row["size"],
                summary=row["summary"],
            )
            for row in stats["communities"]
        ],
    )


@router.delete(
    "/databases/{db}/collections/{coll}/graph",
    response_model=GraphDeleteResponse,
    summary="Delete the derived graph",
    description=(
        "Remove all graph rows from the corpus and drop the entity / "
        "community vector collections. Missing collections are ignored."
    ),
)
async def delete_graph(
    db: str, coll: str, request: Request
) -> GraphDeleteResponse:
    store = request.app.state.store
    entity_coll, community_coll = graph_collection_names(coll)
    await run_in_sqlite(
        request.app.state.corpus.delete_graph, db, coll
    )
    for name in (entity_coll, community_coll):
        try:
            await run_in_store(store.drop_collection, db, name)
        except CollectionNotFound:
            pass
    return GraphDeleteResponse(
        deleted=True,
        entity_collection=entity_coll,
        community_collection=community_coll,
    )
