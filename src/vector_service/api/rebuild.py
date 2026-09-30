"""Index rebuild, promotion and consistency administration.

The SQLite corpus is the system of record; every physical vector
collection is a rebuildable derivation. This module owns the flows
that replace one:

- ``POST .../reindex`` — submit a ``rebuild`` job: a fresh physical
  collection is populated from the corpus leaves (dense + sparse +
  summary vectors all recomputed with the current embedder and a
  freshly fitted BM25 statistics set) and parked as the binding's
  canary while the old index keeps serving;
- ``POST .../reindex/promote`` — atomically promote the canary
  (binding + chunk_indexes registry rewritten in one transaction),
  then drop and discard the old physical index;
- ``POST .../consistency`` — compare the corpus leaf-id set with each
  physical index (active and canary), reporting missing/orphan rows;
  with ``repair=true`` orphans are deleted and missing rows
  rederived.

Rebuild jobs are executed by the same worker as ingests — the worker
dispatches on the row's ``job_type`` and calls
:func:`run_rebuild_pipeline`.
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from vector_service.api.ingest import (
    _CHUNK_INDEX_FIELD,
    _DOC_ID_FIELD,
    _PRIMARY_FIELD,
    _SPARSE_FIELD,
    _SUMMARY_VECTOR_FIELD,
    _VECTOR_FIELD,
    JobCancelled,
    _store_http_error,
    create_thin_collection,
    ensure_database,
)
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
    JOB_EMBEDDING,
    JOB_UPSERTING,
)
from vector_service.schemas.index_admin import (
    ConsistencyRequest,
    ConsistencyResponse,
    IndexConsistencyItem,
    PromoteResponse,
    ReindexRequest,
    ReindexSubmitResponse,
)

router = APIRouter(prefix="/v1", tags=["index-admin"])
log = get_logger(__name__)

#: Page size when browsing physical rows for a consistency scan, and
#: batch size for repair re-embedding.
_SCAN_PAGE = 200
_REPAIR_BATCH = 64


# ---- rebuild pipeline (worker side) -----------------------------------


def _embed_input(leaf: dict[str, Any]) -> str:
    """Dense embedding input for a stored leaf (context prefix if set)."""
    context = leaf["context"]
    if context:
        return f"{context}\n\n{leaf['text']}"
    return leaf["text"]


async def run_rebuild_pipeline(
    *,
    settings: Any,
    store: Any,
    repo: Any,
    bm25: Any,
    embedder: Any,
    database: str,
    logical_collection: str,
    target_ref: str,
    batch_size: int,
    canary_percent: int,
    embed_model: str,
    inference_timeout_seconds: float,
    emit: Any,
    job_id: str,
    should_cancel: Any,
) -> None:
    """Rebuild one physical index from the corpus leaves.

    Stage marks reuse ``embedding`` / ``upserting``. On cancellation
    the half-written target collection is cleaned and the job ends
    cancelled; every other failure cleans the target and raises
    :class:`HTTPException` for the worker to classify.
    """
    cleaned = False

    async def _cleanup() -> None:
        # Best effort, idempotent — a failed cleanup must not mask the
        # original failure or leave the job stuck.
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        try:
            await run_in_store(
                store.drop_collection, database, target_ref
            )
        except CollectionNotFound:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning(
                "rebuild_cleanup_collection_failed",
                job_id=job_id,
                target_ref=target_ref,
                error=str(e),
            )
        try:
            # Filesystem cleanup (stats file) — default executor.
            await asyncio.to_thread(
                bm25.discard, database, target_ref
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "rebuild_cleanup_bm25_failed",
                job_id=job_id,
                error=str(e),
            )

    async def _check_cancel() -> None:
        if await should_cancel():
            await _cleanup()
            await run_in_sqlite(repo.mark_cancelled, job_id)
            raise JobCancelled(job_id)

    try:
        # ---- enumerate leaves ------------------------------------
        await _check_cancel()
        await run_in_sqlite(
            functools.partial(repo.mark_job, job_id, JOB_EMBEDDING),
        )
        await emit({"type": "stage", "stage": "embed"})
        leaves = await run_in_sqlite(
            repo.list_leaf_chunks, database, logical_collection
        )
        if not leaves:
            # Parameter/state error — nothing to rebuild; 4xx terminal.
            raise HTTPException(
                400,
                detail={
                    "error": {
                        "code": "rebuild_empty",
                        "message": (
                            f"no leaf chunks in {database!r}/"
                            f"{logical_collection!r}; nothing to rebuild"
                        ),
                    }
                },
            )

        # ---- fresh BM25 stats for the new physical index --------
        try:
            await run_in_model(
                bm25.fit_for_rebuild,
                repo,
                database,
                logical_collection,
                target_ref,
            )
        except Exception as e:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "sparse_encode_failed",
                        "message": str(e) or "BM25 fit failed",
                    }
                },
            ) from e

        # ---- fresh physical collection with the thin schema -----
        try:
            await run_in_store(ensure_database, store, database)
            existing = await run_in_store(
                store.list_collections, database
            )
            if target_ref in existing:
                # Leftover from an earlier attempt — drop before
                # recreating so the row count verification is exact.
                await run_in_store(
                    store.drop_collection, database, target_ref
                )
            await run_in_store(
                create_thin_collection,
                store,
                database,
                target_ref,
                embedder.dim,
            )
        except (
            BackendError,
            EmbedderError,
            StoreError,
        ) as e:
            raise _store_http_error(e, op="create_collection") from e

        # ---- recompute every vector in batches ------------------
        total = len(leaves)
        for start in range(0, total, batch_size):
            await _check_cancel()
            batch = leaves[start : start + batch_size]
            dense_texts = [_embed_input(leaf) for leaf in batch]
            raw_texts = [leaf["text"] for leaf in batch]
            summary_texts = [
                leaf["summary"] if leaf["summary"] else leaf["text"]
                for leaf in batch
            ]
            try:
                vectors = await asyncio.wait_for(
                    run_in_model(embedder.embed_documents, dense_texts),
                    timeout=inference_timeout_seconds,
                )
                summary_vectors = await asyncio.wait_for(
                    run_in_model(embedder.embed_documents, summary_texts),
                    timeout=inference_timeout_seconds,
                )
                sparse_vectors = await run_in_model(
                    bm25.encode_documents,
                    database,
                    target_ref,
                    raw_texts,
                )
            except TimeoutError as e:
                raise HTTPException(
                    503,
                    detail={
                        "error": {
                            "code": "embedder_unavailable",
                            "message": (
                                f"embedder {embed_model!r} did not finish "
                                f"within {inference_timeout_seconds}s"
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

            if start == 0:
                await run_in_sqlite(
                    functools.partial(
                        repo.mark_job, job_id, JOB_UPSERTING
                    ),
                )
                await emit({"type": "stage", "stage": "upsert"})

            fields = [
                {
                    _DOC_ID_FIELD: leaf["doc_id"],
                    _CHUNK_INDEX_FIELD: int(leaf["chunk_index"]),
                }
                for leaf in batch
            ]
            try:
                await run_in_store(
                    functools.partial(
                        store.upsert,
                        database,
                        target_ref,
                        _PRIMARY_FIELD,
                        _VECTOR_FIELD,
                        [leaf["chunk_id"] for leaf in batch],
                        vectors,
                        fields,
                        extra_vectors={
                            _SUMMARY_VECTOR_FIELD: summary_vectors
                        },
                        sparse_vectors={
                            _SPARSE_FIELD: sparse_vectors
                        },
                    ),
                )
            except (BackendError, StoreError) as e:
                raise _store_http_error(e, op="upsert") from e

            done = min(start + batch_size, total)
            await run_in_sqlite(
                repo.set_job_progress, job_id, done, total
            )

        # ---- verify the rebuilt row count ------------------------
        try:
            actual = await run_in_store(
                store.count_rows, database, target_ref
            )
        except (BackendError, StoreError) as e:
            raise _store_http_error(e, op="count_rows") from e
        if int(actual) != total:
            # The embedder/store silently dropped rows — retryable.
            raise HTTPException(
                500,
                detail={
                    "error": {
                        "code": "rebuild_count_mismatch",
                        "message": (
                            f"rebuilt index {target_ref!r} has {actual} rows, "
                            f"corpus has {total} leaves"
                        ),
                        "expected": total,
                        "got": int(actual),
                    }
                },
            )

        # ---- park as canary; old index keeps serving -------------
        await run_in_sqlite(
            functools.partial(
                repo.set_canary,
                database,
                logical_collection,
                canary_ref=target_ref,
                canary_percent=canary_percent,
                model=embed_model,
            ),
        )

        tokens_used = sum(int(leaf["token_count"]) for leaf in leaves)
        await run_in_sqlite(
            functools.partial(
                repo.finish_job,
                job_id,
                chunk_count=total,
                page_count=None,
                tokens_used=tokens_used,
            ),
        )
        log.info(
            "rebuild_done",
            job_id=job_id,
            collection=f"{database}/{logical_collection}",
            target_ref=target_ref,
            leaves=total,
        )
    except HTTPException:
        await _cleanup()
        raise
    except JobCancelled:
        # Cleanup + cancelled mark were handled at the cancel gate.
        raise
    except Exception as e:  # pragma: no cover — safety net
        log.exception("rebuild_pipeline_failed")
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


# ---- routes: submit / promote -----------------------------------------


def _physical_ref_name(collection: str) -> str:
    """Mint a physical collection name for a rebuild target.

    Milvus collection names must start with a letter/underscore and
    contain only word characters.
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", collection)
    if not safe or safe[0].isdigit():
        safe = f"_{safe}"
    return f"{safe}__rebuild_{uuid.uuid4().hex[:12]}"


@router.post(
    "/databases/{db}/collections/{coll}/reindex",
    response_model=ReindexSubmitResponse,
    status_code=202,
    summary="Submit an index rebuild job",
    description=(
        "Rebuild the physical index of one corpus-backed logical "
        "collection: all leaf vectors (dense / sparse / summary) are "
        "recomputed from the SQLite corpus into a fresh collection, "
        "which becomes the binding's canary. The old index keeps "
        "serving until promotion. Pin requests with "
        '``{"index_ref": "..."}`` on /v1/retrieval.'
    ),
)
async def submit_reindex_job(
    db: str, coll: str, body: ReindexRequest, request: Request
) -> ReindexSubmitResponse:
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
                        "collection; nothing in the corpus to rebuild"
                    ),
                }
            },
        )

    target_ref = _physical_ref_name(coll)
    params = {
        "target_ref": target_ref,
        "target_embed_model": body.embed_model,
        "canary_percent": body.canary_percent,
        "batch_size": body.batch_size,
    }
    job_id = uuid.uuid4().hex
    await run_in_sqlite(
        repo.create_job,
        job_id,
        job_type="rebuild",
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

    return ReindexSubmitResponse(job_id=job_id, index_ref=target_ref)


@router.post(
    "/databases/{db}/collections/{coll}/reindex/promote",
    response_model=PromoteResponse,
    summary="Promote the canary index and retire the old one",
    description=(
        "Atomically promote the canary (binding + chunk_indexes "
        "registry rewritten in one transaction), then drop the old "
        "physical collection and its BM25 state. 409 when no canary "
        "is parked, or when the collection has a regression gate whose "
        "latest check is not a pass for this canary."
    ),
)
async def promote_reindex_job(
    db: str, coll: str, request: Request
) -> PromoteResponse:
    repo = request.app.state.corpus
    store = request.app.state.store
    bm25 = request.app.state.bm25

    gate = await run_in_sqlite(
        repo.get_gate_for_collection, db, coll
    )
    if gate is not None:
        # A gate exists for this scope: the latest check must pass for
        # the canary currently parked. A pass for an older canary does
        # not authorize a different physical index.
        binding = await run_in_sqlite(repo.get_binding, db, coll)
        canary_ref = binding["canary_ref"] if binding else None
        check = await run_in_sqlite(
            repo.get_latest_gate_check, gate["gate_id"]
        )
        ok = (
            check is not None
            and check["status"] == "passed"
            and check["candidate_ref"] == canary_ref
        )
        if not ok:
            raise HTTPException(
                409,
                detail={
                    "error": {
                        "code": "gate_blocked",
                        "message": (
                            "promotion blocked by regression gate "
                            f"{gate['gate_id']!r}: submit a gate check and "
                            "wait for it to pass before promoting"
                        ),
                        "gate_id": gate["gate_id"],
                        "latest_status": (
                            check["status"] if check is not None else None
                        ),
                    }
                },
            )
    try:
        promoted = await run_in_sqlite(
            repo.promote_index, db, coll
        )
    except LookupError as e:
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "reindex_no_canary",
                    "message": str(e),
                }
            },
        ) from e

    old_ref = promoted["old_ref"]
    new_ref = promoted["new_ref"]
    if old_ref != new_ref:
        try:
            await run_in_store(store.drop_collection, db, old_ref)
        except CollectionNotFound:
            pass
        except (BackendError, StoreError) as e:
            raise _store_http_error(e, op="drop_collection") from e
        # Filesystem cleanup (stats file) — default executor.
        await asyncio.to_thread(bm25.discard, db, old_ref)
    return PromoteResponse(
        active_ref=new_ref,
        retired_ref=old_ref,
        model=promoted["model"],
    )


# ---- routes: consistency check + repair --------------------------------


async def _scan_physical(
    store: Any, database: str, ref: str
) -> dict[str, Any]:
    """List row ids + authoritative count of one physical collection.

    A missing collection scans as zero rows (repair may recreate it).
    """
    ids: set[str] = set()
    offset = 0
    while True:
        try:
            rows = await run_in_store(
                functools.partial(
                    store.browse,
                    database,
                    ref,
                    _PRIMARY_FIELD,
                    limit=_SCAN_PAGE,
                    offset=offset,
                    output_fields=[_PRIMARY_FIELD],
                ),
            )
        except CollectionNotFound:
            return {"exists": False, "ids": set(), "count": 0}
        if not rows:
            break
        ids.update(str(row["id"]) for row in rows)
        if len(rows) < _SCAN_PAGE:
            break
        offset += _SCAN_PAGE
    try:
        count = await run_in_store(store.count_rows, database, ref)
    except CollectionNotFound:
        return {"exists": False, "ids": set(), "count": 0}
    return {"exists": True, "ids": ids, "count": int(count)}


async def _repair_physical(
    *,
    store: Any,
    repo: Any,
    bm25: Any,
    embedder: Any,
    database: str,
    logical_collection: str,
    ref: str,
    scan: dict[str, Any],
    leaves_by_id: dict[str, dict[str, Any]],
    inference_timeout_seconds: float,
) -> None:
    """Delete orphans, recreate/rederive missing rows for one index."""
    orphan_ids = list(scan["ids"] - set(leaves_by_id))
    missing_ids = [
        chunk_id
        for chunk_id in leaves_by_id
        if chunk_id not in scan["ids"]
    ]

    if not scan["exists"] and missing_ids:
        await run_in_store(ensure_database, store, database)
        await run_in_store(
            create_thin_collection,
            store,
            database,
            ref,
            embedder.dim,
        )

    if orphan_ids:
        await run_in_store(
            store.delete,
            database,
            ref,
            _PRIMARY_FIELD,
            orphan_ids,
        )

    if not missing_ids:
        return

    await run_in_model(
        bm25.ensure_ready, repo, database, logical_collection, ref
    )
    for start in range(0, len(missing_ids), _REPAIR_BATCH):
        batch_ids = missing_ids[start : start + _REPAIR_BATCH]
        leaves = [leaves_by_id[chunk_id] for chunk_id in batch_ids]
        dense_texts = [_embed_input(leaf) for leaf in leaves]
        raw_texts = [leaf["text"] for leaf in leaves]
        summary_texts = [
            leaf["summary"] if leaf["summary"] else leaf["text"]
            for leaf in leaves
        ]
        vectors = await asyncio.wait_for(
            run_in_model(embedder.embed_documents, dense_texts),
            timeout=inference_timeout_seconds,
        )
        summary_vectors = await asyncio.wait_for(
            run_in_model(embedder.embed_documents, summary_texts),
            timeout=inference_timeout_seconds,
        )
        sparse_vectors = await run_in_model(
            bm25.encode_documents,
            database,
            ref,
            raw_texts,
        )
        fields = [
            {
                _DOC_ID_FIELD: leaf["doc_id"],
                _CHUNK_INDEX_FIELD: int(leaf["chunk_index"]),
            }
            for leaf in leaves
        ]
        await run_in_store(
            functools.partial(
                store.upsert,
                database,
                ref,
                _PRIMARY_FIELD,
                _VECTOR_FIELD,
                batch_ids,
                vectors,
                fields,
                extra_vectors={_SUMMARY_VECTOR_FIELD: summary_vectors},
                sparse_vectors={_SPARSE_FIELD: sparse_vectors},
            ),
        )


@router.post(
    "/databases/{db}/collections/{coll}/consistency",
    response_model=ConsistencyResponse,
    summary="Check corpus/index consistency",
    description=(
        "Compare the SQLite leaf chunks with the physical index rows "
        "(active and canary): leaves missing from an index and rows "
        "with no corpus chunk. With ``repair=true`` orphans are "
        "deleted and missing rows rederived (a fully missing physical "
        "collection is recreated)."
    ),
)
async def consistency_check(
    db: str, coll: str, body: ConsistencyRequest, request: Request
) -> ConsistencyResponse:
    store = request.app.state.store
    repo = request.app.state.corpus
    bm25 = request.app.state.bm25
    embedder = request.app.state.embedder
    settings = request.app.state.settings

    leaves = await run_in_sqlite(repo.list_leaf_chunks, db, coll)
    leaves_by_id = {leaf["chunk_id"]: leaf for leaf in leaves}
    leaf_ids = set(leaves_by_id)

    binding = await run_in_sqlite(repo.get_binding, db, coll)
    if binding is None:
        targets = [{"ref": coll, "canary": False}]
    else:
        targets = [{"ref": binding["active_ref"], "canary": False}]
        if binding["canary_ref"]:
            targets.append(
                {"ref": binding["canary_ref"], "canary": True}
            )

    scans: list[dict[str, Any]] = []
    for target in targets:
        scan = await _scan_physical(store, db, target["ref"])
        scans.append({**target, "scan": scan})

    if body.repair:
        if embedder is None:
            raise HTTPException(
                503,
                detail={
                    "error": {
                        "code": "embedder_unavailable",
                        "message": (
                            "text embedder is not loaded; cannot rederive "
                            "missing rows"
                        ),
                    }
                },
            )
        for entry in scans:
            await _repair_physical(
                store=store,
                repo=repo,
                bm25=bm25,
                embedder=embedder,
                database=db,
                logical_collection=coll,
                ref=entry["ref"],
                scan=entry["scan"],
                leaves_by_id=leaves_by_id,
                inference_timeout_seconds=settings.inference_timeout_seconds,
            )
        # Re-scan so the post-repair report reflects real state.
        for entry in scans:
            entry["scan"] = await _scan_physical(store, db, entry["ref"])

    items: list[IndexConsistencyItem] = []
    for entry in scans:
        scan = entry["scan"]
        ids = scan["ids"]
        missing = sorted(leaf_ids - ids)
        orphans = sorted(ids - leaf_ids)
        items.append(
            IndexConsistencyItem(
                ref=entry["ref"],
                canary=entry["canary"],
                milvus_rows=scan["count"],
                missing_in_milvus=missing,
                orphans_in_milvus=orphans,
                ok=not missing and not orphans,
            )
        )

    return ConsistencyResponse(
        database=db,
        collection=coll,
        corpus_leaves=len(leaf_ids),
        indexes=items,
        ok=all(item.ok for item in items),
        repaired=body.repair,
    )
