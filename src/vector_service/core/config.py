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
    embedding_device: str = "auto"  # auto | cpu | cuda
    embedding_batch_size: int = Field(32, ge=1, le=512)
    embedding_max_length: int = Field(512, ge=1, le=8192)
    embedding_max_texts_per_request: int = Field(256, ge=1)
    embedding_max_chars_per_text: int = Field(8192, ge=1)
    embedding_hf_repo: str = "BAAI/bge-m3"
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VS_",
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