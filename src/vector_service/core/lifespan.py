"""Application lifespan: load model + open vector store."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
from vector_service.core.instance_lock import InstanceLock
from vector_service.core.errors import (
    ImageEmbedderError,
    ModelNotLoaded,
    ModelNotLoadedForImages,
)
from vector_service.core.logging import get_logger, setup_logging
from vector_service.core.metrics import MODEL_LOADED, VS_INFO
from vector_service.core.model_lifecycle import attach_default_slots
from vector_service.core.threadpools import (
    ThreadPools,
    bind_pools,
    run_in_model,
    run_in_sqlite,
    run_in_store,
    unbind_pools,
)
from vector_service.core.tracing import init_tracing, shutdown_tracing
from vector_service.corpus import CorpusRepository
from vector_service.corpus.blobs import BlobStore
from vector_service.corpus.bm25 import SparseBM25
from vector_service.embeddings.image_base import ImageEmbedder
from vector_service.embeddings.image_registry import get_image_embedder_class
from vector_service.embeddings.multimodal_base import MultimodalEmbedder
from vector_service.embeddings.multimodal_registry import (
    get_multimodal_embedder_class,
)
from vector_service.jobs import (
    IngestWorker,
    JobEventBus,
    recover_interrupted,
)
from vector_service.jobs.maintenance import MaintenanceWorker
from vector_service.embeddings.registry import get_embedder_class
from vector_service.parsers.docling_parser import (
    ParserUnavailable,
    get_docling_parser,
)
from vector_service.rerankers.base import Reranker
from vector_service.rerankers.registry import get_reranker_class
from vector_service.stores.registry import build_store

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


def build_embedder(settings: Settings):
    cls = get_embedder_class(settings.embedding_backend)
    return cls(settings=settings)


def build_reranker(settings: Settings) -> Reranker:
    """Construct a reranker for ``settings.reranker``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_reranker_class(settings.reranker.backend)
    return cls(settings=settings.reranker)


def build_image_embedder(settings: Settings) -> ImageEmbedder:
    """Construct an image embedder instance for ``settings.image_embedding.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_image_embedder_class(settings.image_embedding.backend)
    return cls(settings=settings.image_embedding)


def build_multimodal_embedder(settings: Settings) -> MultimodalEmbedder:
    """Construct a multimodal embedder for ``settings.multimodal_embedding.backend``.

    Raises ``KeyError`` if the backend name is not registered.
    """
    cls = get_multimodal_embedder_class(settings.multimodal_embedding.backend)
    return cls(settings=settings.multimodal_embedding)


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    settings = get_settings()
    setup_logging(settings.log_format, settings.log_level)
    VS_INFO.labels(
        version="0.2.0",
        embedding_backend=settings.embedding_backend,
        vector_store_backend=settings.vector_store_backend,
    ).set(1)

    # Opt-in OpenTelemetry provider. Disabled (default) returns None
    # and the pipeline's spans hit the API's cheap no-op proxy.
    tracer_provider = init_tracing(settings)

    app.state.settings = settings
    app.state.startup_ts = time.time()

    # Isolated bounded pools (store RPC / model inference / SQLite).
    # Bound before any blocking call exists so every call site — routes,
    # worker, recovery — lands on them; shut down last.
    thread_pools = ThreadPools(settings.runtime)
    app.state.thread_pools = thread_pools
    bind_pools(thread_pools)

    # Single-instance constraint: two processes against one corpus
    # directory would duplicate job workers and split derived state.
    # Take the OS byte-range lock before touching anything shared; a
    # second instance fails fast here. The OS releases it on crash.
    instance_lock = InstanceLock(settings.instance_lock_path)
    instance_lock.acquire()

    app.state.store = build_store(settings)
    store = app.state.store

    # SQLite corpus — system of record for documents/chunks — plus the
    # client-side BM25 encoder for the thin derived index.
    corpus = CorpusRepository(settings.corpus_db_path)
    await run_in_sqlite(corpus.initialize)
    app.state.corpus = corpus
    app.state.bm25 = SparseBM25(settings.bm25_state_dir)

    # Content-addressed originals: upload binaries persist here (never
    # in SQLite), referenced via documents.content_hash.
    blob_store = BlobStore(settings.originals_dir)
    app.state.blob_store = blob_store

    # Eagerly open the vector-store connection so /readyz can report a
    # truthful state. If Milvus is unreachable at startup we still want
    # the process to come up so embeddings keep working; surface the
    # error in the log and let /readyz return 503.
    try:
        await run_in_store(store._ensure_connected)
        log.info("store_connected", backend=store.backend_name, uri=store.uri)
    except Exception as e:
        log.warning("store_connect_failed", backend=store.backend_name, error=str(e))

    # Attach the per-family ``ModelSlot`` holders and empty family
    # mirrors BEFORE any optional loads so the hot-reload and inference
    # routes always see every attribute. With the default
    # ``auto_load=false`` for every family the slots and mirrors stay
    # ``None`` until an operator calls ``POST /v1/models/{id}/load``.
    attach_default_slots(app, settings=settings)
    MODEL_LOADED.labels(kind="embedder").set(0)
    MODEL_LOADED.labels(kind="reranker").set(0)
    MODEL_LOADED.labels(kind="image_embedder").set(0)
    MODEL_LOADED.labels(kind="multimodal_embedder").set(0)

    # ---- text embedder (opt-in eager load) -----------------------
    # ``embedding_auto_load`` defaults to ``False`` so a fresh process
    # starts in a zero-state: no model constructed, no slot populated,
    # /v1/embeddings returns 503. Operators trigger the load via the
    # dashboard or ``POST /v1/models/{id}/load``. Fail-open: a failed
    # eager load is logged + surfaced via /readyz but does NOT kill
    # the process.
    if settings.embedding_auto_load:
        t0 = time.perf_counter()
        try:
            embedder = build_embedder(settings)
            await run_in_model(embedder.load)
        except ModelNotLoaded as e:
            log.error("model_load_failed", error=str(e))
        except Exception as e:
            log.error(
                "model_load_unexpected", error=str(e), exception_type=type(e).__name__
            )
        else:
            load_duration = time.perf_counter() - t0
            app.state.embedder = embedder
            app.state._slot_embedder.set_instance(embedder, load_duration=load_duration)
            MODEL_LOADED.labels(kind="embedder").set(1)
            device = getattr(embedder, "_device", "unknown")
            log.info(
                "model_loaded",
                model=embedder.model_name,
                device=device,
                dim=embedder.dim,
            )

    # ---- reranker (opt-in eager load) ---------------------------
    # All four model families share the same fail-open contract: a
    # failed eager load is logged + surfaced via /readyz but does NOT
    # kill the process. K8s can keep the pod in service and route
    # around it via /readyz, which is strictly better than exiting
    # with a non-zero code for one of four backends.
    if settings.reranker.auto_load:
        t0 = time.perf_counter()
        try:
            reranker = build_reranker(settings)
            await run_in_model(reranker.load)
        except Exception as exc:  # noqa: BLE001 — fail-open, all families share policy
            log.error(
                "reranker_load_failed",
                backend=settings.reranker.backend,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            load_duration = time.perf_counter() - t0
            app.state.reranker = reranker
            app.state._slot_reranker.set_instance(reranker, load_duration=load_duration)
            MODEL_LOADED.labels(kind="reranker").set(1)
            log.info(
                "reranker_loaded",
                model=reranker.model_name,
                backend=settings.reranker.backend,
            )

    # ---- image embedder (opt-in eager load, fail-open) ----------
    if settings.image_embedding.auto_load:
        t0 = time.perf_counter()
        try:
            image_embedder = build_image_embedder(settings)
            await run_in_model(image_embedder.load)
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.error(
                "image_embedder_load_failed",
                backend=settings.image_embedding.backend,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            load_duration = time.perf_counter() - t0
            app.state.image_embedder = image_embedder
            app.state._slot_image.set_instance(
                image_embedder, load_duration=load_duration
            )
            MODEL_LOADED.labels(kind="image_embedder").set(1)
            log.info(
                "image_embedder_loaded",
                model=image_embedder.model_name,
                device=getattr(image_embedder, "_device", "unknown"),
                dim=image_embedder.dim,
            )

    # ---- multimodal embedder (opt-in eager load, fail-open) -----
    if settings.multimodal_embedding.auto_load:
        t0 = time.perf_counter()
        try:
            multimodal_embedder = build_multimodal_embedder(settings)
            await run_in_model(multimodal_embedder.load)
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.error(
                "multimodal_embedder_load_failed",
                backend=settings.multimodal_embedding.backend,
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            load_duration = time.perf_counter() - t0
            app.state.multimodal_embedder = multimodal_embedder
            app.state._slot_multimodal.set_instance(
                multimodal_embedder, load_duration=load_duration
            )
            MODEL_LOADED.labels(kind="multimodal_embedder").set(1)
            log.info(
                "multimodal_embedder_loaded",
                model=multimodal_embedder.model_name,
                device=getattr(multimodal_embedder, "_device", "unknown"),
                dim=multimodal_embedder.dim,
            )

    # ---- parser (opt-in eager load, fail-open) -----------------
    # Docling's DocumentConverter is heavy on first use (model
    # download + layout-pipeline warmup). ``VS_PARSER__AUTO_LOAD=true``
    # triggers an eager ``load()`` here so the first /v1/parse or
    # /v1/ingest request doesn't pay the cold-start cost. Failures
    # are logged + surfaced via /readyz but do NOT kill the process
    # — the same fail-open contract used by every other family.
    if settings.parser.auto_load:
        try:
            # Same process-wide singleton the routes use — warming a
            # separately constructed parser would not take the cold
            # model-build cost off the first request.
            parser = get_docling_parser()
            await run_in_model(parser.load)
        except ParserUnavailable as exc:
            # Docling is an optional dep; a host that hasn't
            # installed it can still serve the rest of the API.
            log.warning("parser_unavailable", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.error(
                "parser_load_failed",
                backend="docling",
                error=str(exc),
                exception_type=type(exc).__name__,
            )
        else:
            app.state.parser = parser
            log.info("parser_loaded", backend="docling")

    # ---- background job worker ----------------------------------
    # Event bus first — SSE nudges for worker state changes fan out
    # through it. Then reset rows the previous process left mid-stage
    # BEFORE the worker can claim anything (a failure here must not
    # block startup; rows simply stay for a manual/next restart), and
    # finally start the in-process worker.
    job_bus = JobEventBus()
    app.state.job_bus = job_bus
    try:
        recovered = await recover_interrupted(app)
        if recovered:
            log.info("startup_jobs_recovered", count=recovered)
    except Exception as exc:  # noqa: BLE001 — recovery is best-effort, worker still starts
        log.warning(
            "job_recovery_failed",
            error=str(exc),
            exception_type=type(exc).__name__,
        )
    job_worker = IngestWorker(app)
    app.state.job_worker = job_worker
    job_worker.start()

    # Periodic VACUUM / backup / blob-sweep loop. Stop it before the
    # layers it cycles over (corpus, blob store, vector store).
    maintenance_worker = MaintenanceWorker(app)
    app.state.maintenance_worker = maintenance_worker
    maintenance_worker.start()

    try:
        yield
    finally:
        # Stop claiming and cancel an in-flight job before tearing
        # down the layers it may be using. A job interrupted here is
        # handled by recovery on the next process.
        try:
            await maintenance_worker.stop()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            log.warning("maintenance_worker_stop_failed", error=str(exc))
        try:
            await job_worker.stop()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            log.warning("job_worker_stop_failed", error=str(exc))
        # Always release any model instances we managed to construct
        # before shutdown — covers both a clean shutdown (yield returns)
        # and a startup failure where one of the four families raised
        # before the others had a chance to load. Without this, an
        # early reranker raise after the text embedder was already on
        # GPU would leak GPU memory until process death.
        for slot_attr in (
            "_slot_embedder",
            "_slot_reranker",
            "_slot_image",
            "_slot_multimodal",
        ):
            slot = getattr(app.state, slot_attr, None)
            if slot is None:
                continue
            try:
                if slot.get() is not None:
                    await run_in_model(slot.unload)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                log.warning(
                    "shutdown_unload_failed",
                    kind=slot.kind,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
        try:
            await run_in_store(store.close)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            log.warning("store_close_failed", error=str(exc))
        unbind_pools()
        thread_pools.shutdown()
        instance_lock.release()
        log.info("shutdown")
        # Last: flush/stop the tracer so no span is created afterwards.
        shutdown_tracing(tracer_provider)
