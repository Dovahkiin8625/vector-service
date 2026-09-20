"""Application settings via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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
    # Operators who want the legacy behaviour can set
    # ``VS_RERANKER__AUTO_LOAD=true``.
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
            raise ValueError(
                f"top_n_default ({v}) must be <= max_top_n ({max_top_n})"
            )
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

    #: Backend name — currently only ``"docling"`` is registered.
    #: Kept as a field for forward compatibility (e.g. adding a
    #: Marker / Unstructured backend later).
    backend: str = "docling"

    #: Soft cap on per-request upload size. Uploads above this
    #: yield 413 from ``POST /v1/parse`` and ``POST /v1/ingest``.
    max_file_size_mb: int = Field(100, ge=1, le=2048)

    #: HuggingFace Hub endpoint Docling will pull its layout/OCR models
    #: from. Applied as ``HF_ENDPOINT`` before Docling/HF is imported,
    #: so hosts where huggingface.co is unreachable can point this at
    #: ``https://hf-mirror.com``. Empty (default) = the upstream
    #: default; an explicit env var already set in the shell wins.
    hf_endpoint: str = ""

    #: When to run OCR on PDFs: ``auto`` (default) only runs the full
    #: OCR pipeline on PDFs whose pages lack an extractable text layer
    #: (scans); text-native PDFs take a model-light fast path
    #: (``do_ocr=False`` — seconds vs tens of seconds on the same
    #: hardware). ``on`` reproduces Docling's default behaviour (OCR
    #: always available — required for mixed/unknown corpora); ``off``
    #: never OCRs (fastest, but scanned pages yield no text).
    ocr: Literal["auto", "on", "off"] = "auto"

    #: Reconstruct tables (TableFormer) on the text-layer fast path.
    #: Kept enabled because the stage costs almost nothing once warm on
    #: a GPU and preserves markdown table output; set false to skip the
    #: model entirely (saves some cold-start time and VRAM).
    table_structure: bool = True


class ChunkingSettings(BaseSettings):
    """Chunking subsystem configuration.

    The default chunk_size / chunk_overlap match the values
    documented in the ingest API spec. Env prefix: ``VS_CHUNKING__``.
    """

    model_config = SettingsConfigDict(env_prefix="VS_CHUNKING__", extra="ignore")

    chunk_size: int = Field(500, ge=1, le=8192)
    chunk_overlap: int = Field(75, ge=0, le=4096)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})"
            )
        return v


class MultimodalEmbeddingSettings(BaseSettings):
    """Multimodal (text + image) embedding subsystem configuration.

    Powers cross-modal retrieval: text-search-image and image-search-text.
    Vectors from ``embed_text`` and ``embed_images`` must live in the
    same space (typically a projection head output, e.g. 512d for
    Chinese-CLIP). Env prefix: ``VS_MULTIMODAL_EMBEDDING__``.
    """

    model_config = SettingsConfigDict(env_prefix="VS_MULTIMODAL_EMBEDDING__", extra="ignore")

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


class Settings(BaseSettings):
    # 服务
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    debug: bool = False

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
    # Kept for backwards compatibility with external configuration files
    # and the BGE-M3 design plan. Currently unused — BGE-M3 has no
    # ONNX code path; see ``bge_m3.py`` for why the CPU backend stays
    # on PyTorch int8. Documented but not read.
    embedding_onnx_repo: str = "BAAI/bge-m3-onnx"
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

    # Reranker (nested; env prefix VS_RERANKER__)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)

    # Image embedding (nested; env prefix VS_IMAGE_EMBEDDING__)
    image_embedding: ImageEmbeddingSettings = Field(default_factory=ImageEmbeddingSettings)

    # Multimodal embedding (nested; env prefix VS_MULTIMODAL_EMBEDDING__)
    multimodal_embedding: MultimodalEmbeddingSettings = Field(
        default_factory=MultimodalEmbeddingSettings
    )

    # Parser (nested; env prefix VS_PARSER__)
    parser: ParserSettings = Field(default_factory=ParserSettings)

    # Chunking (nested; env prefix VS_CHUNKING__)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

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