"""Application settings via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    """Reranker subsystem configuration.

    Env prefix: ``VS_RERANKER__`` (double underscore — pydantic-settings
    nested-field separator).
    """

    model_config = SettingsConfigDict(env_prefix="VS_RERANKER__", extra="ignore")

    backend: str = Field(
        ..., description="Reranker backend name; must exist in RERANKER_REGISTRY."
    )

    # Eager-load on startup. When False (default) the lifespan skips
    # constructing AND loading the reranker — ``app.state.reranker``
    # stays ``None`` until ``POST /v1/models/{id}/load`` is called.
    # Set ``VS_RERANKER__AUTO_LOAD=true`` to eager-load.
    auto_load: bool = False

    # Model identity
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_dir: str = "./models/bge-reranker-v2-m3"
    auto_download: bool = True
    download_source: Literal["huggingface", "modelscope"] = "modelscope"
    hf_repo: str = "BAAI/bge-reranker-v2-m3"
    ms_repo: str = "BAAI/bge-reranker-v2-m3"

    # Inference
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(32, ge=1, le=512)
    max_length: int = Field(512, ge=1, le=8192)

    # Input limits
    max_documents_per_request: int = Field(256, ge=1, le=4096)
    max_chars_per_doc: int = Field(8192, ge=1, le=32768)
    max_query_chars: int = Field(2048, ge=1, le=8192)
    max_top_n: int = Field(64, ge=1, le=1024)
    top_n_default: int = Field(10, ge=1, le=1024)

    @field_validator("top_n_default")
    @classmethod
    def _top_n_default_le_max(cls, v: int, info) -> int:
        max_top_n = info.data.get("max_top_n", 64)
        if v > max_top_n:
            raise ValueError(f"top_n_default ({v}) must be <= max_top_n ({max_top_n})")
        return v


class ImageEmbeddingSettings(BaseSettings):
    """Image embedding subsystem configuration.

    Env prefix: ``VS_IMAGE_EMBEDDING__`` (double underscore — pydantic-settings
    nested-field separator).
    """

    model_config = SettingsConfigDict(env_prefix="VS_IMAGE_EMBEDDING__", extra="ignore")

    backend: str = "openclip-vit-l-14"
    model_dir: str = "./models/openclip-vit-l-14"
    auto_download: bool = True
    # Eager-load on startup. Default False: the lifespan does not
    # construct or load the image embedder — operators trigger loading
    # explicitly via ``POST /v1/models/{id}/load``.
    auto_load: bool = False
    # OpenCLIP weights are fetched by open_clip itself (not via
    # huggingface_hub). The download_source / hf_repo fields are kept for
    # future embedders that DO use HF directly; default values are
    # inert placeholders.
    download_source: Literal["huggingface"] = "huggingface"
    hf_repo: str = ""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(16, ge=1, le=512)
    max_images_per_request: int = Field(64, ge=1, le=1024)
    max_image_bytes: int = Field(10 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    allowed_mime: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {v!r}")
        return v


class ParserSettings(BaseSettings):
    """Document-parser subsystem configuration.

    Mirrors the pattern used by the embedder / reranker / image /
    multimodal settings blocks. Env prefix: ``VS_PARSER__`` (double
    underscore — pydantic-settings nested-field separator).
    """

    model_config = SettingsConfigDict(env_prefix="VS_PARSER__", extra="ignore")

    #: Eager-load Docling on startup. Default ``False`` so a fresh
    #: process stays in a zero-state (no Docling weights loaded,
    #: /v1/parse returns 503 until either ``VS_PARSER__AUTO_LOAD=true``
    #: is set or an operator triggers a parse that lazily initialises
    #: the converter). Idempotent and 409-tolerant in the lifespan
    #: handler so concurrent workers can't trip over each other.
    auto_load: bool = False

    #: Soft cap on per-request upload size. Uploads above this
    #: yield 413 from ``POST /v1/parse`` and ``POST /v1/ingest``.
    max_file_size_mb: int = Field(100, ge=1, le=2048)

    #: HuggingFace Hub endpoint Docling will pull its layout/OCR models
    #: from. Applied as ``HF_ENDPOINT`` before Docling/HF is imported,
    #: so hosts where huggingface.co is unreachable can point this at
    #: ``https://hf-mirror.com``. Empty (default) = the upstream
    #: default; an explicit env var already set in the shell wins.
    hf_endpoint: str = ""

    #: Inference device for Docling's layout / TableFormer / OCR / VLM
    #: models. ``auto`` lets Docling pick (CUDA when available, else
    #: CPU); ``cuda``/``cpu`` force it.
    device: Literal["auto", "cpu", "cuda"] = "auto"

    #: Persist pictures extracted from parsed documents to disk and
    #: reference them in the markdown via ``artifacts_url_prefix``
    #: instead of emitting an ``<!-- image -->`` placeholder. The files
    #: land under ``artifacts_dir/<doc-stem>/images/`` and are served by
    #: the ``/artifacts`` static mount.
    save_images: bool = True

    #: Root directory for per-document artifact folders (extracted
    #: images). Relative paths resolve against the process working
    #: directory, like ``data_dir``.
    artifacts_dir: Path = Path("./data/artifacts")

    #: URL prefix used in the markdown references and mounted as a
    #: static directory in ``main.py``. Must not carry a trailing slash.
    artifacts_url_prefix: str = "/artifacts"

    #: PP-OCR / RapidOCR language packs for the OCR engine. ``ch`` is
    #: the Chinese+English model; other valid values include ``en``,
    #: ``japan``, ``korean`` (RapidOCR PP-OCRv4 language codes). JSON
    #: list via env, e.g. ``VS_PARSER__OCR_LANGS='["ch","en"]'``.
    ocr_langs: list[str] = Field(default_factory=lambda: ["ch"])

    #: Resolution scale at which page pictures are cropped and saved
    #: (Docling ``images_scale``). Higher = sharper extracted images and
    #: OCR input, at proportionally higher VRAM and disk use.
    images_scale: float = Field(2.0, gt=0, le=8)

    #: Preset id for the optional ``vlm`` profile (used only when a
    #: request explicitly selects it — weights are never downloaded for
    #: the default ``standard`` profile). Built-ins include
    #: ``granite_docling`` (default, ibm-granite/granite-docling-258M),
    #: ``smoldocling``, ``qwen``, ``glm_ocr`` … (see
    #: ``VlmConvertOptions.list_preset_ids()``).
    vlm_preset: str = "granite_docling"


class ChunkingSettings(BaseSettings):
    """Chunking subsystem configuration.

    The default chunk_size / chunk_overlap match the values
    documented in the ingest API spec. Env prefix: ``VS_CHUNKING__``.
    """

    model_config = SettingsConfigDict(env_prefix="VS_CHUNKING__", extra="ignore")

    # Default strategy when a request omits one. Per-request values
    # come from /v1/chunk and /v1/ingest; see
    # docs/ingest-pipeline.md for the strategy guide.
    strategy: Literal["fixed", "paragraph", "recursive", "semantic", "llm"] = (
        "recursive"
    )

    chunk_size: int = Field(500, ge=1, le=8192)
    chunk_overlap: int = Field(75, ge=0, le=4096)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v


class LLMSettings(BaseSettings):
    """External OpenAI-compatible chat backend configuration.

    Used by the ``llm`` chunking strategy and by contextual chunk
    enrichment (Anthropic contextual retrieval) — the service
    itself serves embeddings/rerank, never chat, so the model lives
    elsewhere. Any endpoint speaking ``POST /chat/completions``
    works (OpenAI, vLLM, LM Studio, DashScope compat mode …).

    Env prefix: ``VS_LLM__``. Left empty by default; LLM features
    return ``503 llm_unavailable`` until configured.
    """

    model_config = SettingsConfigDict(env_prefix="VS_LLM__", extra="ignore")

    base_url: str = Field(
        default="",
        description="Root URL of the OpenAI-compatible API, e.g. https://api.openai.com/v1",
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Bearer API key. Sent in the Authorization header.",
    )
    model: str = Field(
        default="",
        description="Chat model name to send in the request payload.",
    )
    timeout_seconds: float = Field(60.0, ge=1.0, le=600.0)
    max_concurrency: int = Field(
        4,
        ge=1,
        le=32,
        description="Bound on concurrent chat calls during contextualization.",
    )


class MultimodalEmbeddingSettings(BaseSettings):
    """Multimodal (text + image) embedding subsystem configuration.

    Powers cross-modal retrieval: text-search-image and image-search-text.
    Vectors from ``embed_text`` and ``embed_images`` must live in the
    same space (typically a projection head output, e.g. 512d for
    Chinese-CLIP). Env prefix: ``VS_MULTIMODAL_EMBEDDING__``.
    """

    model_config = SettingsConfigDict(
        env_prefix="VS_MULTIMODAL_EMBEDDING__", extra="ignore"
    )

    backend: str = "chinese-clip-vit-base-patch16"
    model_dir: str = "./models/chinese-clip-vit-base-patch16"
    auto_download: bool = True
    # Eager-load on startup. Default False: the lifespan does not
    # construct or load the multimodal embedder until operators call
    # ``POST /v1/models/{id}/load``.
    auto_load: bool = False
    hf_repo: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(16, ge=1, le=512)

    # Mixed text + image input limits — single cap, regardless of modality.
    max_items_per_request: int = Field(64, ge=1, le=1024)
    max_text_chars: int = Field(512, ge=1, le=32768)
    max_image_bytes: int = Field(10 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    allowed_mime: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device must be auto|cpu|cuda, got {v!r}")
        return v


class JobSettings(BaseSettings):
    """Background job subsystem configuration.

    Covers the in-process worker loop, upload spooling, retries and the
    SSE event channel. Env prefix: ``VS_JOBS__`` (double underscore —
    pydantic-settings nested-field separator).
    """

    model_config = SettingsConfigDict(env_prefix="VS_JOBS__", extra="ignore")

    #: Per-job upload spool root. Each submission lands at
    #: ``<spool_dir>/<job_id>/upload`` before its row is inserted.
    spool_dir: Path = Path("./data/corpus/spool")

    #: Maximum pipeline attempts per job, counting the first run. A
    #: 5xx failure requeues until this is spent; 4xx fails immediately.
    max_attempts: int = Field(3, ge=1, le=32)

    #: Wake the worker immediately on submission instead of waiting for
    #: its next poll.
    wake_on_submit: bool = True

    #: Idle worker poll interval when no wake arrives.
    poll_interval_seconds: float = Field(1.0, ge=0.01, le=60.0)

    #: SSE heartbeat interval (comment frame keeps proxies alive).
    heartbeat_seconds: float = Field(15.0, ge=1.0, le=300.0)

    #: SSE fallback full-resync interval, covering frames dropped to a
    #: slow consumer.
    sse_resync_seconds: float = Field(2.0, ge=1.0, le=60.0)

    #: Base of the exponential 5xx retry backoff
    #: (``base * 2 ** (attempts - 1)``).
    retry_backoff_base_seconds: float = Field(1.0, ge=0.0, le=600.0)

    #: Cap on the computed retry backoff.
    retry_backoff_max_seconds: float = Field(300.0, ge=0.0, le=3600.0)

    @model_validator(mode="after")
    def _backoff_max_covers_base(self):
        if self.retry_backoff_max_seconds < self.retry_backoff_base_seconds:
            raise ValueError(
                "retry_backoff_max_seconds must be >= retry_backoff_base_seconds"
            )
        return self


class BackupSettings(BaseSettings):
    """Corpus snapshot backup configuration.

    Env prefix: ``VS_BACKUP__``. Only SQLite is backed up — the vector
    index and BM25 stats are derived and rebuild from the corpus.
    """

    model_config = SettingsConfigDict(env_prefix="VS_BACKUP__", extra="ignore")

    enabled: bool = True

    #: Directory for ``corpus-<UTC stamp>.db`` snapshots.
    dir: Path = Path("./data/backups")

    #: Snapshots to keep after each cycle (age-count retention).
    retain: int = Field(7, ge=1, le=400)


class MaintenanceSettings(BaseSettings):
    """Periodic VACUUM / backup / blob-sweep configuration.

    Env prefix: ``VS_MAINTENANCE__``.
    """

    model_config = SettingsConfigDict(
        env_prefix="VS_MAINTENANCE__", extra="ignore"
    )

    enabled: bool = True

    #: Time between maintenance cycles.
    interval_seconds: float = Field(3600.0, ge=10.0, le=604800.0)

    #: Run VACUUM once free pages reach this share of total pages.
    vacuum_min_free_ratio: float = Field(0.2, ge=0.0, le=1.0)

    #: Remove blob-store originals no document references.
    blob_sweep_enabled: bool = True


class PoolSettings(BaseModel):
    """One isolated thread pool: worker count + admission cap.

    ``max_pending`` bounds running + queued submissions of this pool
    (async callers queue at this point — backpressure instead of an
    unbounded work pile) and must cover ``workers``.
    """

    workers: int = Field(..., ge=1, le=64)
    max_pending: int = Field(..., ge=1, le=4096)

    @model_validator(mode="after")
    def _pending_covers_workers(self):
        if self.max_pending < self.workers:
            raise ValueError("max_pending must be >= workers")
        return self


class RuntimeSettings(BaseSettings):
    """Blocking-call isolation: independent bounded pools.

    Env prefix ``VS_RUNTIME__``; pools are ``store`` (vector-store RPC),
    ``model`` (local model/CPU inference incl. BM25 + parsing) and
    ``sqlite`` (corpus repository). Anything uncategorized keeps using
    the asyncio default executor.

    Fields are flat (``store_workers`` / ``store_max_pending`` / ...) so
    a single env var (e.g. ``VS_RUNTIME__STORE_WORKERS``) can override
    one knob without replacing — and invalidating — the whole pool
    object. Typed per-pool views are exposed via the ``store`` /
    ``model`` / ``sqlite`` properties.
    """

    model_config = SettingsConfigDict(
        env_prefix="VS_RUNTIME__", extra="ignore"
    )

    store_workers: int = Field(default=8, ge=1, le=64)
    store_max_pending: int = Field(default=64, ge=1, le=4096)
    model_workers: int = Field(default=2, ge=1, le=64)
    model_max_pending: int = Field(default=16, ge=1, le=4096)
    sqlite_workers: int = Field(default=4, ge=1, le=64)
    sqlite_max_pending: int = Field(default=32, ge=1, le=4096)

    @model_validator(mode="after")
    def _pending_covers_workers(self):
        for name in ("store", "model", "sqlite"):
            workers = getattr(self, f"{name}_workers")
            max_pending = getattr(self, f"{name}_max_pending")
            if max_pending < workers:
                raise ValueError(
                    f"{name}_max_pending must be >= {name}_workers"
                )
        return self

    def _pool(self, name: str) -> PoolSettings:
        return PoolSettings(
            workers=getattr(self, f"{name}_workers"),
            max_pending=getattr(self, f"{name}_max_pending"),
        )

    @property
    def store(self) -> PoolSettings:
        return self._pool("store")

    @property
    def model(self) -> PoolSettings:
        return self._pool("model")

    @property
    def sqlite(self) -> PoolSettings:
        return self._pool("sqlite")


class Settings(BaseSettings):
    # 服务
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    debug: bool = False

    #: Per-request timeout for model inference (embeddings / rerank /
    #: similarity) and the embed legs of the ingest pipeline. Env:
    #: ``VS_INFERENCE_TIMEOUT_SECONDS``.
    inference_timeout_seconds: float = Field(60.0, ge=1.0, le=600.0)

    # 嵌入
    embedding_backend: str = "bge-m3"
    embedding_model_dir: Path = Path("./models/bge-m3")
    embedding_auto_download: bool = True
    # Eager-load on startup. Default False: the lifespan does not
    # construct or load the text embedder until operators call
    # ``POST /v1/models/{id}/load`` from the dashboard or an external
    # orchestrator. Set ``VS_EMBEDDING_AUTO_LOAD=true`` to opt back into
    # the eager-load behaviour (required for ``/readyz`` to report
    # ``embedder: loaded`` at boot).
    embedding_auto_load: bool = False
    embedding_device: str = "auto"  # auto | cpu | cuda
    embedding_batch_size: int = Field(32, ge=1, le=512)
    embedding_max_length: int = Field(512, ge=1, le=8192)
    embedding_max_texts_per_request: int = Field(256, ge=1)
    embedding_max_chars_per_text: int = Field(8192, ge=1)
    embedding_hf_repo: str = "BAAI/bge-m3"
    embedding_ms_repo: str = "BAAI/bge-m3"
    # 权重下载来源：huggingface | modelscope
    embedding_download_source: str = "modelscope"

    # 向量库：直连 Milvus server（standalone / cluster）。
    # 用 `VS_VECTOR_STORE_BACKEND=milvus` 启用；本服务只支持这一种后端。
    vector_store_backend: str = "milvus"
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_token: str = ""
    milvus_timeout: float = Field(30.0, ge=0.1, le=600.0)

    # 运行时
    data_dir: Path = Path("./data")

    #: SQLite corpus database — the system of record for documents and
    #: chunks. The vector index is derived from it and fully rebuildable.
    #: Env: ``VS_CORPUS_DB_PATH``.
    corpus_db_path: Path = Path("./data/corpus/corpus.db")

    #: Directory for per-collection client-side BM25 statistics; also
    #: derived and rebuilt from the corpus when missing. Env:
    #: ``VS_BM25_STATE_DIR``.
    bm25_state_dir: Path = Path("./data/corpus/bm25")

    #: Content-addressed store for original upload binaries. Raw bytes
    #: never enter SQLite; files live here keyed by SHA-256 and are
    #: referenced via documents.content_hash. Env: ``VS_ORIGINALS_DIR``.
    originals_dir: Path = Path("./data/corpus/originals")

    #: Exclusive OS file lock held for the process lifetime. A second
    #: service against the same corpus directory fails fast at startup.
    #: Env: ``VS_INSTANCE_LOCK_PATH``.
    instance_lock_path: Path = Path("./data/corpus/instance.lock")

    # Reranker (nested; env prefix VS_RERANKER__)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)

    # Image embedding (nested; env prefix VS_IMAGE_EMBEDDING__)
    image_embedding: ImageEmbeddingSettings = Field(
        default_factory=ImageEmbeddingSettings
    )

    # Multimodal embedding (nested; env prefix VS_MULTIMODAL_EMBEDDING__)
    multimodal_embedding: MultimodalEmbeddingSettings = Field(
        default_factory=MultimodalEmbeddingSettings
    )

    # Parser (nested; env prefix VS_PARSER__)
    parser: ParserSettings = Field(default_factory=ParserSettings)

    # Chunking (nested; env prefix VS_CHUNKING__)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # External LLM chat backend (nested; env prefix VS_LLM__)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Background jobs (nested; env prefix VS_JOBS__)
    jobs: JobSettings = Field(default_factory=JobSettings)

    # Blocking-call thread pools (nested; env prefix VS_RUNTIME__)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    # Corpus snapshot backups (nested; env prefix VS_BACKUP__)
    backup: BackupSettings = Field(default_factory=BackupSettings)

    # Periodic storage maintenance (nested; env prefix VS_MAINTENANCE__)
    maintenance: MaintenanceSettings = Field(
        default_factory=MaintenanceSettings
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VS_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("embedding_device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("auto", "cpu", "cuda"):
            raise ValueError(f"embedding_device must be auto|cpu|cuda, got {v!r}")
        return v

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        if v not in ("json", "console"):
            raise ValueError(f"log_format must be json|console, got {v!r}")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
