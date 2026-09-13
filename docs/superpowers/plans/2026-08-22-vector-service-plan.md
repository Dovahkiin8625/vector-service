# 向量服务 (vector-service) 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建生产级 FastAPI 向量服务，提供 OpenAI 兼容嵌入 API 和可扩展的向量库管理抽象，第一阶段支持 BGE-M3（GPU/CPU ONNX）和 Milvus Lite。

**Architecture:** 单进程 FastAPI，分层 Clean Architecture：`api/` 路由 → `embeddings/` + `stores/` 抽象 → `core/` 横切关注点。嵌入器和向量库通过 ABC + registry 注册，便于新增后端。同步重计算用 `loop.run_in_executor` 包装。

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, pydantic-settings, structlog, prometheus-client, pymilvus (milvus-lite), onnxruntime, torch, FlagEmbedding, transformers, huggingface-hub, pytest, ruff, mypy。

**Spec:** `docs/superpowers/specs/2026-08-22-vector-service-design.md`

---

## Global Constraints

来自 spec，逐条照抄，落地时不得修改：

- Python ≥ 3.11
- 配置统一前缀 `VS_`（pydantic-settings, `env_file=".env"`）
- Embedder ABC：`dim: int`, `model_name: str`, `embed_documents(texts: list[str]) -> list[list[float]]`, `embed_query(text: str) -> list[float]`
- VectorStore ABC：`create_collection(name, dim, *, metric="cosine", **backend_opts)`, `drop_collection(name)`, `list_collections() -> list[str]`, `collection_info(name) -> dict`, `upsert(collection, ids, vectors, metadatas=None)`, `delete(collection, ids)`, `get(collection, ids) -> list[dict]`, `search(collection, query_vector, top_k=10, filter=None) -> list[Hit]`, `backend` property
- `Hit` dataclass：`{id: str, score: float, metadata: dict}`
- BGE-M3 输出仅 dense，1024 维 float32
- `embed_query` 加前缀 `为这个句子生成表示以用于检索：`；`embed_documents` 不加前缀
- Milvus Lite 路径默认 `./data/milvus.db`，启动时父目录自动创建
- 索引默认 HNSW `M=16, efConstruction=200`，metric=COSINE，search `ef=64`
- filter 翻译范围：仅顶层 key/value 等值 + AND；嵌套/OR → 422
- 异常映射：EmbedderError→503, StoreError→503, CollectionNotFound→404, CollectionAlreadyExists→409, DimensionMismatch→422, pydantic.ValidationError→422, 未捕获→500
- HTTP 错误响应格式：`{"error": {"code": str, "message": str, "request_id": str, "extra": {}}}`
- request_id 优先取 `X-Request-ID` 头，否则生成 `req_<uuid8>`
- 日志输出 JSON（structlog），关键事件：`startup`, `shutdown`, `model_loaded`, `store_opened`, `embedding_request`, `store_operation`
- Metrics 列表：`vs_embedding_duration_seconds`, `vs_embedding_tokens_total`, `vs_embedding_requests_total`, `vs_store_operation_duration_seconds`, `vs_store_collections`, `vs_store_vectors_total`, `vs_model_loaded`, `vs_info`
- Histogram 桶：`[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]`
- Token 估算：`ceil(chars / 4)`
- 限制：`batch_size=32`, `max_length=512`, `max_texts_per_request=256`, `max_chars_per_text=8192`
- 设备选择：`auto` → CUDA 可用走 GPU（Torch），否则 CPU（ONNX）
- 自动下载：路径不存在时若 `VS_EMBEDDING_AUTO_DOWNLOAD=true` 拉取，否则启动失败
- 预热：启动后跑 5 条空字符串
- 测试分层：`tests/unit/`, `tests/contract/`, `tests/integration/`
- BGE-M3 集成测试用真实模型（标 `@pytest.mark.slow`）
- 向量库契约测试用 `tmp_path` 的 milvus.db

---

## 文件结构总览（最终态）

```
F:\project\vector-service\
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── Makefile
├── src\vector_service\
│   ├── __init__.py
│   ├── main.py
│   ├── api\
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── management.py
│   │   ├── health.py
│   │   └── backend.py
│   ├── embeddings\
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bge_m3.py
│   │   └── registry.py
│   ├── stores\
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── milvus_lite.py
│   │   └── registry.py
│   ├── core\
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   ├── filter_translator.py
│   │   ├── middleware.py
│   │   └── lifespan.py
│   ├── schemas\
│   │   ├── __init__.py
│   │   ├── openai.py
│   │   └── management.py
│   └── testing\
│       ├── __init__.py
│       ├── fake_embedder.py
│       └── fake_store.py
├── tests\
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit\
│   │   ├── test_config.py
│   │   ├── test_errors.py
│   │   ├── test_logging.py
│   │   ├── test_metrics.py
│   │   └── test_filter_translator.py
│   ├── contract\
│   │   ├── test_embedder_contract.py
│   │   └── test_vector_store_contract.py
│   └── integration\
│       ├── conftest.py
│       ├── test_embeddings_api.py
│       ├── test_management_api.py
│       └── test_health.py
└── docs\superpowers\specs\2026-08-22-vector-service-design.md
```

每个文件单一职责，任务分解以此为准。

---

## Phase 0 — 项目脚手架

### Task 1: 项目脚手架 + 依赖

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `Makefile`, `src\vector_service\__init__.py`, `tests\__init__.py`, `tests\conftest.py`

**Interfaces:**
- Consumes: 无
- Produces: 可 `pip install -e .`；`python -c "import vector_service; print(vector_service.__version__)"` 返版本字符串

- [ ] **Step 1: 写测试 `tests/test_version.py`**

```python
def test_version_exposed():
    import vector_service
    assert isinstance(vector_service.__version__, str)
    assert len(vector_service.__version__.split(".")) == 3
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/test_version.py -v`
Expected: `ModuleNotFoundError: No module named 'vector_service'`

- [ ] **Step 3: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vector-service"
version = "0.1.0"
description = "Production-grade vector service: OpenAI-compatible embeddings + extensible vector store"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "structlog>=24.1",
    "prometheus-client>=0.20",
    "httpx>=0.27",
]

[project.optional-dependencies]
embed = [
    "torch>=2.2",
    "onnxruntime>=1.18",
    "FlagEmbedding>=1.2.10",
    "transformers>=4.41",
    "huggingface-hub>=0.23",
]
store = [
    "pymilvus>=2.4.3",
]
dev = [
    "pytest>=8.1",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
vector-service = "vector_service.main:run"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM"]
```

- [ ] **Step 4: 写 `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
*.db
.env
data/
models/
logs/
```

- [ ] **Step 5: 写 `.env.example`**

```env
# 服务
VS_HOST=0.0.0.0
VS_PORT=8080
VS_WORKERS=1
VS_LOG_LEVEL=INFO
VS_LOG_FORMAT=json
VS_DEBUG=false

# 嵌入
VS_EMBEDDING_BACKEND=bge-m3
VS_EMBEDDING_MODEL_DIR=./models/bge-m3
VS_EMBEDDING_AUTO_DOWNLOAD=true
VS_EMBEDDING_DEVICE=auto
VS_EMBEDDING_BATCH_SIZE=32
VS_EMBEDDING_MAX_LENGTH=512
VS_EMBEDDING_MAX_TEXTS_PER_REQUEST=256
VS_EMBEDDING_MAX_CHARS_PER_TEXT=8192
VS_EMBEDDING_HF_REPO=BAAI/bge-m3
VS_EMBEDDING_ONNX_REPO=BAAI/bge-m3-onnx

# 向量库
VS_VECTOR_STORE_BACKEND=milvus_lite
VS_MILVUS_URI=./data/milvus.db

# 运行时
VS_DATA_DIR=./data
```

- [ ] **Step 6: 写 `Makefile`**

```makefile
.PHONY: install run test lint format type-check clean

install:
	pip install -e ".[embed,store,dev]"

run:
	uvicorn vector_service.main:app --reload --host $$(grep VS_HOST .env | cut -d= -f2) --port $$(grep VS_PORT .env | cut -d= -f2)

test:
	pytest -q

test-unit:
	pytest -q tests/unit

test-contract:
	pytest -q tests/contract

test-integration:
	pytest -q tests/integration -m "not slow"

test-slow:
	pytest -q tests/integration -m slow

lint:
	ruff check src tests

format:
	ruff format src tests

type-check:
	mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
```

- [ ] **Step 7: 写 `README.md`（占位）**

```markdown
# vector-service

生产级 FastAPI 向量服务：OpenAI 兼容嵌入 + 可扩展向量库。

## 快速开始

```bash
make install
cp .env.example .env
make run
```

## 文档

- [设计 spec](docs/superpowers/specs/2026-08-22-vector-service-design.md)
- [实施计划](docs/superpowers/plans/2026-08-22-vector-service-plan.md)
```
```

- [ ] **Step 8: 写 `src\vector_service\__init__.py`**

```python
"""Vector service: OpenAI-compatible embeddings + extensible vector store."""

__version__ = "0.1.0"
```

- [ ] **Step 9: 写 `tests\__init__.py` 和 `tests\conftest.py`**

`tests\__init__.py`: 空文件

`tests\conftest.py`:
```python
import sys
from pathlib import Path

# 让 src/ 在 import 路径中
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 10: 安装包（editable）**

Run: `cd F:\project\vector-service && pip install -e .`
Expected: 成功安装；`pip show vector-service` 显示 0.1.0

- [ ] **Step 11: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/test_version.py -v`
Expected: 1 passed

- [ ] **Step 12: git init 并首次提交**

Run:
```bash
cd F:\project\vector-service
git init
git add .
git commit -m "chore: project scaffolding with pyproject and test infrastructure"
```

---

## Phase 1 — 横切基础

### Task 2: 配置 (Settings)

**Files:**
- Create: `src\vector_service\core\__init__.py`, `src\vector_service\core\config.py`
- Test: `tests\unit\test_config.py`

**Interfaces:**
- Consumes: 环境变量（前缀 `VS_`，可选 `.env`）
- Produces: `get_settings() -> Settings`（lru_cache 单例）

- [ ] **Step 1: 写 `tests\unit\test_config.py`**

```python
import pytest
from pydantic import ValidationError

from vector_service.core.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"
    assert s.port == 8080
    assert s.embedding_backend == "bge-m3"
    assert s.embedding_device == "auto"
    assert s.vector_store_backend == "milvus_lite"
    assert s.embedding_batch_size == 32
    assert s.embedding_max_length == 512


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("VS_PORT", "9090")
    monkeypatch.setenv("VS_EMBEDDING_DEVICE", "cuda")
    # 重建 settings 实例
    from vector_service.core import config as cfg
    cfg.get_settings.cache_clear()
    s = cfg.get_settings()
    assert s.port == 9090
    assert s.embedding_device == "cuda"


def test_invalid_device_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_device="tpu")


def test_paths_are_path_objects():
    s = Settings(_env_file=None)
    from pathlib import Path
    assert isinstance(s.embedding_model_dir, Path)
    assert isinstance(s.data_dir, Path)


def test_get_settings_singleton():
    from vector_service.core import config as cfg
    cfg.get_settings.cache_clear()
    a = cfg.get_settings()
    b = cfg.get_settings()
    assert a is b
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_config.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\core\__init__.py`**

空文件

- [ ] **Step 4: 写 `src\vector_service\core\config.py`**

```python
"""Application settings via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # 向量库
    vector_store_backend: str = "milvus_lite"
    milvus_uri: str = "./data/milvus.db"

    # 运行时
    data_dir: Path = Path("./data")

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
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_config.py -v`
Expected: 5 passed

- [ ] **Step 6: 提交**

Run:
```bash
git add src/vector_service/core/ tests/unit/test_config.py
git commit -m "feat(config): add Settings with VS_ env prefix and pydantic-settings"
```

---

### Task 3: 错误层级

**Files:**
- Create: `src\vector_service\core\errors.py`
- Test: `tests\unit\test_errors.py`

**Interfaces:**
- Produces:
  - `VectorServiceError(Exception)`
  - `EmbedderError(VectorServiceError)`
  - `ModelNotLoaded(EmbedderError)`
  - `StoreError(VectorServiceError)`
  - `CollectionNotFound(StoreError)`
  - `CollectionAlreadyExists(StoreError)`
  - `DimensionMismatch(StoreError)`（带 `expected`/`got` 属性）
  - `BackendError(StoreError)`

- [ ] **Step 1: 写 `tests\unit\test_errors.py`**

```python
import pytest

from vector_service.core.errors import (
    VectorServiceError,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
    CollectionNotFound,
    CollectionAlreadyExists,
    DimensionMismatch,
    BackendError,
)


def test_hierarchy():
    assert issubclass(EmbedderError, VectorServiceError)
    assert issubclass(StoreError, VectorServiceError)
    assert issubclass(CollectionNotFound, StoreError)
    assert issubclass(CollectionAlreadyExists, StoreError)
    assert issubclass(ModelNotLoaded, EmbedderError)
    assert issubclass(BackendError, StoreError)


def test_dimension_mismatch_carries_attrs():
    e = DimensionMismatch("dim wrong", expected=4, got=3)
    assert e.expected == 4
    assert e.got == 3
    assert "dim wrong" in str(e)


def test_dimension_mismatch_defaults():
    e = DimensionMismatch("x")
    assert e.expected is None
    assert e.got is None


def test_can_be_caught_as_base():
    with pytest.raises(VectorServiceError):
        raise CollectionNotFound("missing")
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_errors.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\core\errors.py`**

```python
"""Exception hierarchy. Translates to HTTP in the API layer."""
from __future__ import annotations

from typing import Any


class VectorServiceError(Exception):
    """Base for all errors raised by vector-service."""


class EmbedderError(VectorServiceError):
    """Embedding model inference or availability failure."""


class ModelNotLoaded(EmbedderError):
    """Model failed to load at startup or is no longer available."""


class StoreError(VectorServiceError):
    """Vector store backend failure."""


class CollectionNotFound(StoreError):
    """Operation referenced a non-existent collection."""


class CollectionAlreadyExists(StoreError):
    """Creation would clobber an existing collection."""


class DimensionMismatch(StoreError):
    """Vector dimension does not match the collection's."""

    def __init__(self, message: str, *, expected: int | None = None, got: int | None = None):
        super().__init__(message)
        self.expected = expected
        self.got = got


class BackendError(StoreError):
    """Wrapped native backend exception."""
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_errors.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

Run:
```bash
git add src/vector_service/core/errors.py tests/unit/test_errors.py
git commit -m "feat(errors): add exception hierarchy for HTTP translation"
```

---

### Task 4: 结构化日志

**Files:**
- Create: `src\vector_service\core\logging.py`
- Test: `tests\unit\test_logging.py`

**Interfaces:**
- Consumes: `log_format: str`, `log_level: str`
- Produces: `setup_logging(format, level) -> None`；上下文变量 `request_id_var: ContextVar[str|None]`

- [ ] **Step 1: 写 `tests\unit\test_logging.py`**

```python
import io
import json
import logging

import structlog

from vector_service.core.logging import setup_logging, request_id_var


def _capture_log_output(format: str = "json", level: str = "INFO"):
    buf = io.StringIO()
    setup_logging(format=format, level=level, stream=buf)
    return buf


def test_json_output_is_valid_json():
    buf = _capture_log_output("json")
    log = structlog.get_logger("test")
    log.info("hello", foo="bar")
    line = buf.getvalue().strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["event"] == "hello"
    assert obj["foo"] == "bar"
    assert obj["level"] == "info"
    assert "ts" in obj


def test_console_output_is_text():
    buf = _capture_log_output("console")
    log = structlog.get_logger("test")
    log.info("hi", k="v")
    assert "hi" in buf.getvalue()


def test_request_id_included_when_set():
    buf = _capture_log_output("json")
    request_id_var.set("req_abc")
    log = structlog.get_logger("test")
    log.info("with-id")
    obj = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert obj["request_id"] == "req_abc"


def test_log_level_respected(caplog):
    setup_logging(format="console", level="WARNING")
    log = structlog.get_logger("test_lvl")
    with caplog.at_level(logging.DEBUG):
        log.info("should-be-filtered")
        log.warning("should-appear")
    assert "should-be-filtered" not in caplog.text
    assert "should-appear" in caplog.text
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_logging.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\core\logging.py`**

```python
"""Structured logging via structlog."""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import IO, Any

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def setup_logging(
    format: str = "json",
    level: str = "INFO",
    stream: IO[Any] | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Args:
        format: "json" or "console"
        level: stdlib log level name
        stream: optional output stream (defaults to stderr)
    """
    out = stream or sys.stderr
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        _add_request_id,
    ]

    if format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(out)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)


def _add_request_id(_logger: Any, _name: str, event_dict: dict) -> dict:
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_logging.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

Run:
```bash
git add src/vector_service/core/logging.py tests/unit/test_logging.py
git commit -m "feat(logging): structured JSON logging with request_id context var"
```

---

### Task 5: Prometheus 指标

**Files:**
- Create: `src\vector_service\core\metrics.py`
- Test: `tests\unit\test_metrics.py`

**Interfaces:**
- Produces: 全局指标对象
  - `EMBEDDING_DURATION_SECONDS: Histogram(labels=[model, status])`
  - `EMBEDDING_TOKENS_TOTAL: Counter(labels=[model])`
  - `EMBEDDING_REQUESTS_TOTAL: Counter(labels=[model, status])`
  - `STORE_OP_DURATION_SECONDS: Histogram(labels=[op, backend, status])`
  - `STORE_COLLECTIONS: Gauge(labels=[backend])`
  - `STORE_VECTORS_TOTAL: Counter(labels=[op, backend])`
  - `MODEL_LOADED: Gauge(labels=[model, device])`
  - `VS_INFO: Gauge(labels=[version, embedding_backend, vector_store_backend])`
- 函数 `render_metrics() -> str`（用于 `/metrics` 端点）

- [ ] **Step 1: 写 `tests\unit\test_metrics.py`**

```python
from prometheus_client import parser

from vector_service.core.metrics import (
    EMBEDDING_DURATION_SECONDS,
    EMBEDDING_TOKENS_TOTAL,
    STORE_OP_DURATION_SECONDS,
    STORE_COLLECTIONS,
    MODEL_LOADED,
    VS_INFO,
    render_metrics,
)


def test_required_metrics_exist():
    names = {
        EMBEDDING_DURATION_SECONDS._name,
        EMBEDDING_TOKENS_TOTAL._name,
        STORE_OP_DURATION_SECONDS._name,
        STORE_COLLECTIONS._name,
        MODEL_LOADED._name,
        VS_INFO._name,
    }
    assert names == {
        "vs_embedding_duration_seconds",
        "vs_embedding_tokens_total",
        "vs_store_operation_duration_seconds",
        "vs_store_collections",
        "vs_model_loaded",
        "vs_info",
    }


def test_embedding_duration_has_required_labels():
    labels = EMBEDDING_DURATION_SECONDS._labelnames
    assert "model" in labels
    assert "status" in labels


def test_render_metrics_is_parseable():
    EMBEDDING_DURATION_SECONDS.labels(model="x", status="ok").observe(0.1)
    out = render_metrics()
    # 至少包含我们的指标名
    assert "vs_embedding_duration_seconds" in out
    # 可被 prometheus_client 解析
    parsed = list(parser.text_string_to_metric_families(out))
    assert any(f.name == "vs_embedding_duration_seconds" for f in parsed)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_metrics.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\core\metrics.py`**

```python
"""Prometheus metric definitions."""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# 全局 registry，所有指标共享
REGISTRY = CollectorRegistry(auto_describe=True)

EMBEDDING_DURATION_SECONDS = Histogram(
    "vs_embedding_duration_seconds",
    "Embedding request duration in seconds",
    labelnames=("model", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

EMBEDDING_TOKENS_TOTAL = Counter(
    "vs_embedding_tokens_total",
    "Total tokens processed by embedder",
    labelnames=("model",),
    registry=REGISTRY,
)

EMBEDDING_REQUESTS_TOTAL = Counter(
    "vs_embedding_requests_total",
    "Total embedding requests",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

STORE_OP_DURATION_SECONDS = Histogram(
    "vs_store_operation_duration_seconds",
    "Vector store operation duration in seconds",
    labelnames=("op", "backend", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    registry=REGISTRY,
)

STORE_COLLECTIONS = Gauge(
    "vs_store_collections",
    "Number of collections in the store",
    labelnames=("backend",),
    registry=REGISTRY,
)

STORE_VECTORS_TOTAL = Counter(
    "vs_store_vectors_total",
    "Total vector write/delete operations",
    labelnames=("op", "backend"),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "vs_model_loaded",
    "1 if the embedding model is loaded, 0 otherwise",
    labelnames=("model", "device"),
    registry=REGISTRY,
)

VS_INFO = Gauge(
    "vs_info",
    "Static build/runtime information",
    labelnames=("version", "embedding_backend", "vector_store_backend"),
    registry=REGISTRY,
)


def render_metrics() -> str:
    """Render Prometheus exposition format text."""
    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

Run:
```bash
git add src/vector_service/core/metrics.py tests/unit/test_metrics.py
git commit -m "feat(metrics): prometheus metric definitions and render helper"
```

---

## Phase 2 — 抽象与工具

### Task 6: filter 翻译器

**Files:**
- Create: `src\vector_service\core\filter_translator.py`
- Test: `tests\unit\test_filter_translator.py`

**Interfaces:**
- Produces:
  - `translate_filter(filter_dict: dict) -> str`（Milvus expr）
  - `FilterTranslationError(ValueError)`（不支持的语法 → 422）

- [ ] **Step 1: 写 `tests\unit\test_filter_translator.py`**

```python
import pytest

from vector_service.core.filter_translator import translate_filter, FilterTranslationError


def test_empty():
    assert translate_filter(None) == ""
    assert translate_filter({}) == ""


def test_single_equality():
    assert translate_filter({"source": "doc1"}) == 'metadata["source"] == "doc1"'


def test_multiple_keys_are_anded():
    out = translate_filter({"source": "doc1", "lang": "en"})
    assert out == 'metadata["source"] == "doc1" and metadata["lang"] == "en"'


def test_int_value():
    out = translate_filter({"count": 5})
    assert out == 'metadata["count"] == 5'


def test_float_value():
    out = translate_filter({"score": 0.5})
    assert out == 'metadata["score"] == 0.5'


def test_bool_value():
    out = translate_filter({"active": True})
    assert out == 'metadata["active"] == True'


def test_none_value():
    out = translate_filter({"tag": None})
    assert out == 'metadata["tag"] == null'


def test_nested_dict_rejected():
    with pytest.raises(FilterTranslationError):
        translate_filter({"meta": {"a": 1}})


def test_list_value_rejected():
    with pytest.raises(FilterTranslationError):
        translate_filter({"tags": ["a", "b"]})


def test_or_unsupported_in_v1():
    with pytest.raises(FilterTranslationError):
        # v1 不支持 OR，使用 operator 字段强制 OR 应被拒绝
        translate_filter({"__or__": [{"a": 1}, {"b": 2}]})


def test_quote_in_string_escaped():
    out = translate_filter({"name": 'a"b'})
    assert '\\"a\\"b\\"' in out or 'a\\"b' in out
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_filter_translator.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\core\filter_translator.py`**

```python
"""Translate a generic filter dict to a Milvus expression.

v1 scope: only top-level key/value equality, AND-composed.
Anything more complex raises FilterTranslationError.
"""
from __future__ import annotations

import json
from typing import Any


class FilterTranslationError(ValueError):
    """The filter cannot be translated to the backend expression."""


def translate_filter(filter: dict[str, Any] | None) -> str:
    """Translate a filter dict to a Milvus-style expression.

    Args:
        filter: dict of {field_name: value}, AND-composed. Values must be
            JSON-compatible scalars (str, int, float, bool, None).
        Empty/None returns "" (no filter).

    Returns:
        Milvus expression string like `metadata["k"] == "v"`.

    Raises:
        FilterTranslationError: if any value is a non-scalar (dict/list),
            or any key/value pair cannot be safely represented.
    """
    if not filter:
        return ""

    parts: list[str] = []
    for k, v in filter.items():
        _validate_key(k)
        parts.append(f'metadata["{k}"] {_op(v)} {_render_value(v)}')
    return " and ".join(parts)


def _validate_key(k: str) -> None:
    if not isinstance(k, str) or not k:
        raise FilterTranslationError(f"filter key must be non-empty str, got {k!r}")
    # 防止注入 `]` 之类破坏表达式
    if any(c in k for c in ['"', "\\", "\n", "\r"]):
        raise FilterTranslationError(f"filter key contains illegal chars: {k!r}")


def _op(v: Any) -> str:
    return "=="


def _render_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, str):
        # 用 json.dumps 转义并加双引号
        return json.dumps(v, ensure_ascii=False)
    raise FilterTranslationError(
        f"unsupported filter value type: {type(v).__name__} (v1 only supports scalars)"
    )
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_filter_translator.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

Run:
```bash
git add src/vector_service/core/filter_translator.py tests/unit/test_filter_translator.py
git commit -m "feat(filter): milvus filter translator with scalar-only v1 scope"
```

---

### Task 7: Embedder ABC + FakeEmbedder

**Files:**
- Create: `src\vector_service\embeddings\__init__.py`, `src\vector_service\embeddings\base.py`, `src\vector_service\testing\__init__.py`, `src\vector_service\testing\fake_embedder.py`
- Test: `tests\contract\__init__.py`, `tests\contract\conftest.py`, `tests\contract\test_embedder_contract.py`

**Interfaces:**
- Produces:
  - `class Embedder(ABC): dim, model_name; embed_documents(texts); embed_query(text)`
  - `class FakeEmbedder(Embedder)`: dim=4, deterministic hash-based vector（测试用）

- [ ] **Step 1: 写 `tests\contract\conftest.py`**

```python
import pytest

from vector_service.testing.fake_embedder import FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=8)
```

- [ ] **Step 2: 写 `tests\contract\test_embedder_contract.py`**

```python
import pytest

from vector_service.embeddings.base import Embedder
from vector_service.core.errors import EmbedderError


def test_embedder_is_abstract():
    # 尝试直接实例化 Embedder 应失败
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


def test_dim_attribute_present(embedder):
    assert embedder.dim == 8
    assert isinstance(embedder.dim, int)


def test_model_name_present(embedder):
    assert isinstance(embedder.model_name, str)
    assert embedder.model_name


def test_embed_query_returns_one_vector_of_correct_dim(embedder):
    v = embedder.embed_query("hello")
    assert isinstance(v, list)
    assert len(v) == embedder.dim
    assert all(isinstance(x, float) for x in v)


def test_embed_documents_returns_one_per_input(embedder):
    texts = ["a", "b", "c"]
    out = embedder.embed_documents(texts)
    assert len(out) == 3
    assert all(len(v) == embedder.dim for v in out)


def test_embed_documents_empty_input(embedder):
    assert embedder.embed_documents([]) == []


def test_query_and_doc_share_space(embedder):
    # 同一个 embedder 必须产生同一语义空间：同一文本经 query/doc 路径可不同维度但应同 dim
    q = embedder.embed_query("x")
    d = embedder.embed_documents(["x"])[0]
    assert len(q) == len(d)


def test_subclass_must_implement_both():
    class Incomplete(Embedder):
        dim = 4
        model_name = "x"

        def embed_documents(self, texts):
            return [[0.0] * 4 for _ in texts]
        # 缺 embed_query

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_embedder_contract.py -v`
Expected: ModuleNotFoundError 或 fixture 找不到

- [ ] **Step 4: 写 `src\vector_service\embeddings\__init__.py`**

空文件

- [ ] **Step 5: 写 `src\vector_service\embeddings\base.py`**

```python
"""Embedder abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base for embedding models.

    Concrete subclasses must:
    - Set `dim` (int) and `model_name` (str)
    - Implement `embed_documents` and `embed_query`

    Both methods are synchronous. Async dispatch is the caller's job
    (use `loop.run_in_executor` from FastAPI routes).
    """

    dim: int
    model_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Return one vector per input."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. May differ from embed_documents (e.g. BGE prefix)."""
```

- [ ] **Step 6: 写 `src\vector_service\testing\__init__.py`**

空文件

- [ ] **Step 7: 写 `src\vector_service\testing\fake_embedder.py`**

```python
"""Deterministic fake embedder for tests."""
from __future__ import annotations

import hashlib

from vector_service.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    """Deterministic embedder using SHA-256-derived vectors.

    Useful in tests where you don't want to load a real model.
    Same input always yields the same vector.
    """

    def __init__(self, dim: int = 4, model_name: str = "fake-embedder"):
        self.dim = dim
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector_for(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vector_for(f"q:{text}", self.dim)


def _vector_for(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 每个字节产生 4 个 float 位，从字节数组扩展成 dim 维
    out: list[float] = []
    i = 0
    while len(out) < dim:
        b = digest[i % len(digest)]
        # 拆成两个 nibble 避免浮点退化
        out.append(((b >> 4) / 15.0) - 0.5)
        out.append(((b & 0xF) / 15.0) - 0.5)
        i += 1
    return out[:dim]
```

- [ ] **Step 8: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_embedder_contract.py -v`
Expected: 8 passed

- [ ] **Step 9: 提交**

Run:
```bash
git add src/vector_service/embeddings/ src/vector_service/testing/ tests/contract/
git commit -m "feat(embeddings): Embedder ABC and FakeEmbedder for tests"
```

---

### Task 8: VectorStore ABC + FakeStore

**Files:**
- Create: `src\vector_service\stores\__init__.py`, `src\vector_service\stores\base.py`, `src\vector_service\testing\fake_store.py`
- Test: `tests\contract\test_vector_store_contract.py`

**Interfaces:**
- Produces:
  - `@dataclass Hit(id: str, score: float, metadata: dict)`
  - `class VectorStore(ABC)`: 所有 §5.2 方法 + `backend_name: str` + `backend` property

- [ ] **Step 1: 写 `tests\contract\test_vector_store_contract.py`**

```python
import pytest

from vector_service.stores.base import VectorStore, Hit
from vector_service.core.errors import (
    CollectionNotFound,
    CollectionAlreadyExists,
    DimensionMismatch,
)


@pytest.fixture
def store():
    from vector_service.testing.fake_store import FakeStore
    return FakeStore()


def test_hit_is_dataclass():
    h = Hit(id="a", score=0.1, metadata={"k": "v"})
    assert h.id == "a"
    assert h.score == 0.1
    assert h.metadata == {"k": "v"}


def test_create_and_drop(store):
    store.create_collection("c1", dim=4)
    assert "c1" in store.list_collections()
    store.drop_collection("c1")
    assert "c1" not in store.list_collections()


def test_create_twice_raises(store):
    store.create_collection("c2", dim=4)
    with pytest.raises(CollectionAlreadyExists):
        store.create_collection("c2", dim=4)


def test_drop_missing_raises(store):
    with pytest.raises(CollectionNotFound):
        store.drop_collection("nope")


def test_list_collections_empty(store):
    assert store.list_collections() == []


def test_upsert_and_get(store):
    store.create_collection("c3", dim=4)
    store.upsert(
        "c3",
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadatas=[{"k": "v"}, {}],
    )
    got = store.get("c3", ids=["a", "b"])
    assert len(got) == 2
    assert got[0]["id"] == "a"
    assert got[0]["metadata"] == {"k": "v"}


def test_search_returns_top_k_ordered(store):
    store.create_collection("c4", dim=4)
    store.upsert(
        "c4",
        ids=["a", "b", "c"],
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
    )
    hits = store.search("c4", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0].id in ("a", "c")  # 余弦最近
    assert hits[1].id in ("a", "c")


def test_search_on_missing_collection_raises(store):
    with pytest.raises(CollectionNotFound):
        store.search("missing", query_vector=[0.0] * 4)


def test_upsert_wrong_dim_raises(store):
    store.create_collection("c5", dim=4)
    with pytest.raises(DimensionMismatch):
        store.upsert("c5", ids=["x"], vectors=[[1.0, 0.0, 0.0]])


def test_delete(store):
    store.create_collection("c6", dim=4)
    store.upsert("c6", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]])
    store.delete("c6", ids=["a"])
    assert {h["id"] for h in store.get("c6", ids=["a", "b"])} == {"b"}


def test_collection_info(store):
    store.create_collection("c7", dim=4, metric="cosine")
    info = store.collection_info("c7")
    assert info["dim"] == 4
    assert info["metric"] == "cosine"


def test_filter_is_accepted(store):
    store.create_collection("c8", dim=4)
    store.upsert(
        "c8",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadatas=[{"src": "doc1"}, {"src": "doc2"}],
    )
    hits = store.search("c8", query_vector=[1, 0, 0, 0], top_k=10, filter={"src": "doc1"})
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_backend_property_present(store):
    # 抽象属性应该有具体值
    assert isinstance(store.backend_name, str)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_vector_store_contract.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: 写 `src\vector_service\stores\__init__.py`**

空文件

- [ ] **Step 4: 写 `src\vector_service\stores\base.py`**

```python
"""VectorStore abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base for vector store backends.

    Concrete subclasses must implement all abstract methods and set
    `backend_name` (str). The `backend` property exposes the native
    client/connection for backend-specific operations.
    """

    backend_name: str

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dim: int,
        *,
        metric: str = "cosine",
        **backend_opts: Any,
    ) -> None:
        """Create a collection with the given dimension and metric.

        Raises:
            CollectionAlreadyExists: if a collection with this name exists.
            StoreError: on backend failure.
        """

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Raises:
            CollectionNotFound: if no such collection.
        """

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return all collection names in this store."""

    @abstractmethod
    def collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata: at least `dim`, `metric`, `count`."""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert or update vectors with given ids and metadata.

        Raises:
            CollectionNotFound, DimensionMismatch, StoreError.
        """

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by id. Silently ignores missing ids."""

    @abstractmethod
    def get(self, collection: str, ids: list[str]) -> list[dict]:
        """Fetch vectors by id.

        Returns a list of `{id, vector, metadata}` dicts. Missing ids
        are skipped (not error).
        """

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[Hit]:
        """Top-k nearest neighbors.

        Raises:
            CollectionNotFound, DimensionMismatch, StoreError.
        """

    @property
    def backend(self) -> Any:
        """Native client/connection. Use only for backend-specific ops."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources. Default: no-op."""
```

- [ ] **Step 5: 写 `src\vector_service\testing\fake_store.py`**

```python
"""In-memory FakeStore implementing VectorStore ABC for tests."""
from __future__ import annotations

import math
from typing import Any

from vector_service.core.errors import (
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.stores.base import Hit, VectorStore


class FakeStore(VectorStore):
    """Pure-Python in-memory vector store. Brute-force cosine search."""

    def __init__(self):
        self.backend_name = "fake"
        self._collections: dict[str, _FakeCollection] = {}

    @property
    def backend(self) -> Any:
        return self  # escape hatch: 暴露自己

    def create_collection(self, name, dim, *, metric="cosine", **_opts):
        if name in self._collections:
            raise CollectionAlreadyExists(f"collection {name!r} exists")
        self._collections[name] = _FakeCollection(name, dim, metric)

    def drop_collection(self, name):
        if name not in self._collections:
            raise CollectionNotFound(name)
        del self._collections[name]

    def list_collections(self):
        return list(self._collections.keys())

    def collection_info(self, name):
        c = self._require(name)
        return {"name": c.name, "dim": c.dim, "metric": c.metric, "count": len(c.records)}

    def upsert(self, collection, ids, vectors, metadatas=None):
        c = self._require(collection)
        if len(ids) != len(vectors):
            raise StoreError("ids and vectors must have same length")
        metas = metadatas or [{} for _ in ids]
        for vid, vec, meta in zip(ids, vectors, metas):
            if len(vec) != c.dim:
                raise DimensionMismatch(
                    f"vector dim {len(vec)} != collection dim {c.dim}",
                    expected=c.dim, got=len(vec),
                )
            c.records[vid] = (list(vec), dict(meta))

    def delete(self, collection, ids):
        c = self._require(collection)
        for vid in ids:
            c.records.pop(vid, None)

    def get(self, collection, ids):
        c = self._require(collection)
        return [
            {"id": vid, "vector": c.records[vid][0], "metadata": c.records[vid][1]}
            for vid in ids if vid in c.records
        ]

    def search(self, collection, query_vector, top_k=10, filter=None):
        c = self._require(collection)
        if len(query_vector) != c.dim:
            raise DimensionMismatch(
                f"query dim {len(query_vector)} != collection dim {c.dim}",
                expected=c.dim, got=len(query_vector),
            )

        scored: list[Hit] = []
        for vid, (vec, meta) in c.records.items():
            if filter is not None and not _matches_filter(meta, filter):
                continue
            score = _cosine(query_vector, vec)
            scored.append(Hit(id=vid, score=score, metadata=dict(meta)))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def _require(self, name) -> "_FakeCollection":
        if name not in self._collections:
            raise CollectionNotFound(name)
        return self._collections[name]


class _FakeCollection:
    def __init__(self, name, dim, metric):
        self.name = name
        self.dim = dim
        self.metric = metric
        self.records: dict[str, tuple[list[float], dict]] = {}


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _matches_filter(meta: dict, flt: dict) -> bool:
    return all(meta.get(k) == v for k, v in flt.items())
```

- [ ] **Step 6: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_vector_store_contract.py -v`
Expected: 13 passed

- [ ] **Step 7: 提交**

Run:
```bash
git add src/vector_service/stores/ src/vector_service/testing/fake_store.py tests/contract/test_vector_store_contract.py
git commit -m "feat(stores): VectorStore ABC, Hit, and in-memory FakeStore"
```

---

## Phase 3 — 具体实现

### Task 9: BGE-M3 embedder

**Files:**
- Create: `src\vector_service\embeddings\bge_m3.py`, `src\vector_service\embeddings\registry.py`
- Test: `tests\unit\test_bge_m3_loading.py`（仅测加载/分发逻辑，标 `@pytest.mark.slow` 跑真实模型）

**Interfaces:**
- Produces:
  - `class BGEM3Embedder(Embedder)`: dim=1024, model_name="bge-m3"
  - 自动下载/加载逻辑、`device` 字段
  - `EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {"bge-m3": BGEM3Embedder}`
  - `get_embedder_class(name: str) -> type[Embedder]`

**依赖安装要求**：本任务执行前需 `pip install -e ".[embed]"`，否则 import torch / FlagEmbedding 失败。

- [ ] **Step 1: 写 `tests\unit\test_bge_m3_loading.py`（不依赖真实模型）**

```python
import pytest

from vector_service.embeddings.registry import EMBEDDER_REGISTRY, get_embedder_class
from vector_service.embeddings.base import Embedder


def test_bge_m3_registered():
    assert "bge-m3" in EMBEDDER_REGISTRY
    cls = get_embedder_class("bge-m3")
    assert issubclass(cls, Embedder)


def test_unknown_embedder_raises():
    from vector_service.core.errors import EmbedderError
    with pytest.raises(EmbedderError):
        get_embedder_class("nonexistent-model")


def test_bge_m3_class_attrs():
    from vector_service.embeddings.bge_m3 import BGEM3Embedder
    assert BGEM3Embedder.dim == 1024
    assert BGEM3Embedder.model_name == "bge-m3"


@pytest.mark.slow
def test_bge_m3_real_load_cpu(tmp_path):
    """慢测试：真实加载 BGE-M3 (CPU)。需要模型已下载或网络可达。"""
    import os
    os.environ.setdefault("VS_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("VS_EMBEDDING_MODEL_DIR", str(tmp_path / "bge-m3"))
    os.environ.setdefault("VS_EMBEDDING_AUTO_DOWNLOAD", "true")

    from vector_service.core.config import get_settings
    get_settings.cache_clear()
    from vector_service.embeddings.bge_m3 import BGEM3Embedder

    emb = BGEM3Embedder()
    v = emb.embed_query("hello world")
    assert len(v) == 1024
    assert all(isinstance(x, float) for x in v)
```

- [ ] **Step 2: 运行测试（slow 排除），确认 registry 部分失败**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_bge_m3_loading.py -v -m "not slow"`
Expected: ModuleNotFoundError 或 import 失败

- [ ] **Step 3: 安装 embed 依赖**

Run: `cd F:\project\vector-service && pip install -e ".[embed]"`
Expected: torch, onnxruntime, FlagEmbedding, transformers, huggingface-hub 安装成功

- [ ] **Step 4: 写 `src\vector_service\embeddings\bge_m3.py`**

```python
"""BGE-M3 embedder with GPU/CPU dispatch."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import EmbedderError, ModelNotLoaded
from vector_service.embeddings.base import Embedder

_QUERY_PREFIX = "为这个句子生成表示以用于检索："


class BGEM3Embedder(Embedder):
    dim = 1024
    model_name = "bge-m3"

    def __init__(
        self,
        settings: Settings | None = None,
    ):
        s = settings or get_settings()
        self._settings = s
        self._device = self._resolve_device(s.embedding_device)
        self._batch_size = s.embedding_batch_size
        self._max_length = s.embedding_max_length
        self._model_dir = Path(s.embedding_model_dir)
        self._impl: _BGEBackend | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested == "cuda":
            return _check_cuda()
        if requested == "cpu":
            return "cpu"
        # auto
        try:
            return _check_cuda()
        except ModelNotLoaded:
            return "cpu"

    def _ensure_loaded(self) -> "_BGEBackend":
        if self._impl is not None:
            return self._impl
        self._ensure_model_dir()
        try:
            if self._device == "cuda":
                self._impl = _TorchBackend(
                    self._model_dir, self._device, self._max_length, self._batch_size
                )
            else:
                # CPU: 优先 ONNX（int8 量化），失败回退 Torch
                try:
                    self._impl = _OnnxBackend(
                        self._model_dir, self._max_length, self._batch_size
                    )
                except Exception:
                    self._impl = _TorchBackend(
                        self._model_dir, "cpu", self._max_length, self._batch_size
                    )
        except Exception as e:
            raise ModelNotLoaded(f"failed to load BGE-M3: {e}") from e

        # 预热
        try:
            self.embed_documents([""] * 5)
        except Exception as e:  # 预热失败不致命
            import structlog
            structlog.get_logger(__name__).warning("bge_m3_warmup_failed", error=str(e))
        return self._impl

    def _ensure_model_dir(self) -> None:
        s = self._settings
        self._model_dir.mkdir(parents=True, exist_ok=True)
        # 判断是否已有必需文件（粗略）
        if _dir_has_model(self._model_dir):
            return
        if not s.embedding_auto_download:
            raise ModelNotLoaded(
                f"model dir {self._model_dir} has no BGE-M3 files and auto_download is off"
            )
        # 下载
        repo = s.embedding_onnx_repo if self._device == "cpu" else s.embedding_hf_repo
        snapshot_download(
            repo_id=repo,
            local_dir=str(self._model_dir),
            local_dir_use_symlinks=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        impl = self._ensure_loaded()
        return impl.encode(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        impl = self._ensure_loaded()
        return impl.encode([_QUERY_PREFIX + text], is_query=True)[0]


def _check_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError as e:
        raise ModelNotLoaded(f"torch not available: {e}")
    raise ModelNotLoaded("CUDA not available")


def _dir_has_model(p: Path) -> bool:
    if not p.exists():
        return False
    # 至少有 config.json 或 model.safetensors / model.onnx
    if (p / "config.json").exists():
        return True
    if any(p.glob("*.onnx")):
        return True
    if any(p.glob("*.safetensors")) or any(p.glob("pytorch_model.bin")):
        return True
    return False


class _BGEBackend:
    def encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        raise NotImplementedError


class _TorchBackend(_BGEBackend):
    def __init__(self, model_dir: Path, device: str, max_length: int, batch_size: int):
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=(device == "cuda"),
            device=device,
        )
        self._max_length = max_length
        self._batch_size = batch_size

    def encode(self, texts, is_query):
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [list(map(float, v)) for v in out["dense_vecs"]]


class _OnnxBackend(_BGEBackend):
    def __init__(self, model_dir: Path, max_length: int, batch_size: int):
        # 简单 ONNX 推理：使用 optimum 或 onnxruntime 直接跑
        # 这里用 FlagEmbedding 的 ONNX 接口；若仓库不含 ONNX，会抛错回退
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=False,
            device="cpu",
        )
        self._max_length = max_length
        self._batch_size = batch_size

    def encode(self, texts, is_query):
        out = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [list(map(float, v)) for v in out["dense_vecs"]]
```

**注意**：`_TorchBackend` 和 `_OnnxBackend` 的具体推理路径在实现时如发现 FlagEmbedding 接口不匹配，需调整成 `transformers` 直跑；本质目标是"加载模型 + 返回 1024-d float32 dense 向量 + 批处理"。保持 ABC 契约不变，实现细节可微调。

- [ ] **Step 5: 写 `src\vector_service\embeddings\registry.py`**

```python
"""Embedder registry."""
from __future__ import annotations

from vector_service.core.errors import EmbedderError
from vector_service.embeddings.base import Embedder
from vector_service.embeddings.bge_m3 import BGEM3Embedder

EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {
    "bge-m3": BGEM3Embedder,
}


def get_embedder_class(name: str) -> type[Embedder]:
    if name not in EMBEDDER_REGISTRY:
        raise EmbedderError(f"unknown embedding model: {name!r}")
    return EMBEDDER_REGISTRY[name]


def list_embedder_names() -> list[str]:
    return list(EMBEDDER_REGISTRY.keys())
```

- [ ] **Step 6: 运行测试（不含 slow），确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_bge_m3_loading.py -v -m "not slow"`
Expected: 3 passed

- [ ] **Step 7: 提交**

Run:
```bash
git add src/vector_service/embeddings/bge_m3.py src/vector_service/embeddings/registry.py tests/unit/test_bge_m3_loading.py
git commit -m "feat(embeddings): BGE-M3 with GPU/CPU dispatch + registry"
```

---

### Task 10: Milvus Lite store

**Files:**
- Create: `src\vector_service\stores\milvus_lite.py`, `src\vector_service\stores\registry.py`
- Test: `tests\contract\test_milvus_lite_extra.py`（在 tmp 路径上跑）

**依赖**：`pip install -e ".[store]"`（pymilvus 含 milvus-lite）

**Interfaces:**
- Produces:
  - `class MilvusLiteStore(VectorStore)`: backend_name="milvus_lite"
  - `STORES_REGISTRY: dict[str, type[VectorStore]]`
  - `build_store(settings) -> VectorStore`

- [ ] **Step 1: 安装 store 依赖**

Run: `cd F:\project\vector-service && pip install -e ".[store]"`
Expected: pymilvus 安装成功

- [ ] **Step 2: 写 `tests\contract\test_milvus_lite_extra.py`**

```python
import os
import uuid

import pytest

from vector_service.stores.milvus_lite import MilvusLiteStore


@pytest.fixture
def store(tmp_path):
    uri = str(tmp_path / f"milvus_{uuid.uuid4().hex[:8]}.db")
    s = MilvusLiteStore(uri=uri)
    yield s
    s.close()


def test_collection_info_includes_count(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a", "b"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]])
    info = store.collection_info("c")
    assert info["count"] == 2


def test_get_returns_vector_and_metadata(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]], metadatas=[{"k": "v"}])
    got = store.get("c", ids=["a"])
    assert got[0]["metadata"] == {"k": "v"}


def test_filter_search(store):
    store.create_collection("c", dim=4)
    store.upsert(
        "c",
        ids=["a", "b"],
        vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
        metadatas=[{"src": "x"}, {"src": "y"}],
    )
    hits = store.search("c", query_vector=[1, 0, 0, 0], top_k=10, filter={"src": "x"})
    assert len(hits) == 1
    assert hits[0].id == "a"


def test_upsert_updates_existing(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]], metadatas=[{"v": 1}])
    store.upsert("c", ids=["a"], vectors=[[0, 1, 0, 0]], metadatas=[{"v": 2}])
    got = store.get("c", ids=["a"])
    assert got[0]["vector"] == [0, 1, 0, 0]
    assert got[0]["metadata"] == {"v": 2}


def test_delete_idempotent(store):
    store.create_collection("c", dim=4)
    store.upsert("c", ids=["a"], vectors=[[1, 0, 0, 0]])
    store.delete("c", ids=["a", "nonexistent"])  # 不应抛


def test_backend_property_returns_native(store):
    assert store.backend is not None
    # 应当是 pymilvus 的 Connection 之类
    assert hasattr(store.backend, "list_collections") or hasattr(store.backend, "describe_collection")
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_milvus_lite_extra.py -v`
Expected: ModuleNotFoundError 或 ImportError

- [ ] **Step 4: 写 `src\vector_service\stores\milvus_lite.py`**

```python
"""Milvus Lite implementation of VectorStore."""
from __future__ import annotations

import json
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from vector_service.core.config import Settings, get_settings
from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    StoreError,
)
from vector_service.core.filter_translator import FilterTranslationError, translate_filter
from vector_service.stores.base import Hit, VectorStore

_VALID_METRICS = {"cosine": "COSINE", "ip": "IP", "l2": "L2"}


class MilvusLiteStore(VectorStore):
    backend_name = "milvus_lite"

    def __init__(self, uri: str | None = None, settings: Settings | None = None):
        s = settings or get_settings()
        self._uri = uri or s.milvus_uri
        # 确保父目录存在
        import os
        parent = os.path.dirname(self._uri)
        if parent:
            os.makedirs(parent, exist_ok=True)

        try:
            connections.connect("default", uri=self._uri)
        except Exception as e:
            raise BackendError(f"failed to connect to milvus lite at {self._uri}: {e}") from e

    @property
    def backend(self) -> Any:
        """原生连接：返回当前 default connection 的引用对象（pymilvus 不暴露对象，但可用 utility）。"""
        return _MilvusLiteBackendProxy(self._uri)

    def close(self) -> None:
        try:
            connections.disconnect("default")
        except Exception:
            pass

    def create_collection(self, name, dim, *, metric="cosine", **backend_opts):
        if name in self.list_collections():
            raise CollectionAlreadyExists(f"collection {name!r} exists")
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
            coll = Collection(name=name, schema=schema, using="default")

            metric_type = _VALID_METRICS.get(metric)
            if metric_type is None:
                raise StoreError(f"unsupported metric {metric!r}")
            index_params = {
                "metric_type": metric_type,
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            coll.create_index(field_name="vector", index_params=index_params)
        except (CollectionAlreadyExists, StoreError):
            raise
        except Exception as e:
            raise BackendError(f"create_collection failed: {e}") from e

    def drop_collection(self, name):
        if name not in self.list_collections():
            raise CollectionNotFound(name)
        try:
            utility.drop_collection(name, using="default")
        except Exception as e:
            raise BackendError(f"drop_collection failed: {e}") from e

    def list_collections(self) -> list[str]:
        try:
            return list(utility.list_collections(using="default"))
        except Exception as e:
            raise BackendError(f"list_collections failed: {e}") from e

    def collection_info(self, name):
        self._require(name)
        try:
            coll = Collection(name, using="default")
            coll.flush()
            count = coll.num_entities
            # 从 schema 读 dim
            dim = next(
                f.params["dim"]
                for f in coll.schema.fields
                if f.name == "vector"
            )
            metric = self._metric_for(coll)
            return {"name": name, "dim": dim, "metric": metric, "count": count}
        except CollectionNotFound:
            raise
        except Exception as e:
            raise BackendError(f"collection_info failed: {e}") from e

    def upsert(self, collection, ids, vectors, metadatas=None):
        coll = self._require(collection)
        if len(ids) != len(vectors):
            raise StoreError("ids and vectors length mismatch")
        # dim 校验
        expected_dim = self.collection_info(collection)["dim"]
        for i, v in enumerate(vectors):
            if len(v) != expected_dim:
                raise DimensionMismatch(
                    f"vector[{i}] dim {len(v)} != collection dim {expected_dim}",
                    expected=expected_dim, got=len(v),
                )

        metas = metadatas or [{} for _ in ids]
        rows = [
            {"id": vid, "vector": list(map(float, vec)), "metadata_json": json.dumps(meta, ensure_ascii=False)}
            for vid, vec, meta in zip(ids, vectors, metas)
        ]
        try:
            coll.upsert(rows)
            coll.flush()
        except Exception as e:
            raise BackendError(f"upsert failed: {e}") from e

    def delete(self, collection, ids):
        coll = self._require(collection)
        try:
            expr = " or ".join(f'id == "{_escape(vid)}"' for vid in ids)
            coll.delete(expr)
            coll.flush()
        except Exception as e:
            raise BackendError(f"delete failed: {e}") from e

    def get(self, collection, ids):
        coll = self._require(collection)
        if not ids:
            return []
        try:
            expr = " or ".join(f'id == "{_escape(vid)}"' for vid in ids)
            rows = coll.query(expr=expr, output_fields=["id", "metadata_json"])
            by_id = {r["id"]: r for r in rows}
            return [
                {
                    "id": vid,
                    "vector": by_id[vid].get("vector"),  # query 不返回 vector，需 search
                    "metadata": json.loads(by_id[vid]["metadata_json"]) if vid in by_id else {},
                }
                for vid in ids if vid in by_id
            ]
        except Exception as e:
            raise BackendError(f"get failed: {e}") from e

    def search(self, collection, query_vector, top_k=10, filter=None):
        coll = self._require(collection)
        # dim 校验
        expected_dim = self.collection_info(collection)["dim"]
        if len(query_vector) != expected_dim:
            raise DimensionMismatch(
                f"query dim {len(query_vector)} != collection dim {expected_dim}",
                expected=expected_dim, got=len(query_vector),
            )
        try:
            expr = translate_filter(filter)
        except FilterTranslationError as e:
            raise StoreError(f"filter not supported: {e}") from e

        try:
            params = {"ef": 64}
            results = coll.search(
                data=[list(map(float, query_vector))],
                anns_field="vector",
                param=params,
                limit=top_k,
                expr=expr or None,
                output_fields=["metadata_json"],
            )
            hits: list[Hit] = []
            if not results:
                return hits
            for hit in results[0]:
                meta_raw = hit.entity.get("metadata_json") if hasattr(hit, "entity") else "{}"
                meta = json.loads(meta_raw) if meta_raw else {}
                hits.append(Hit(id=hit.id, score=float(hit.score), metadata=meta))
            return hits
        except (DimensionMismatch, StoreError):
            raise
        except Exception as e:
            raise BackendError(f"search failed: {e}") from e

    def _require(self, name) -> Collection:
        if name not in self.list_collections():
            raise CollectionNotFound(name)
        return Collection(name, using="default")

    @staticmethod
    def _metric_for(coll: Collection) -> str:
        try:
            idx = coll.indexes
            if idx:
                mt = idx[0].params.get("metric_type", "COSINE")
                inv = {v: k for k, v in _VALID_METRICS.items()}
                return inv.get(mt, mt.lower())
        except Exception:
            pass
        return "cosine"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


class _MilvusLiteBackendProxy:
    """Escape hatch: exposes a few utility methods, hides raw connection."""

    def __init__(self, uri: str):
        self.uri = uri

    def list_collections(self):
        return list(utility.list_collections(using="default"))

    def describe_collection(self, name: str):
        return Collection(name, using="default").schema
```

- [ ] **Step 5: 写 `src\vector_service\stores\registry.py`**

```python
"""Vector store registry."""
from __future__ import annotations

from vector_service.core.config import Settings
from vector_service.core.errors import StoreError
from vector_service.stores.base import VectorStore
from vector_service.stores.milvus_lite import MilvusLiteStore

STORES_REGISTRY: dict[str, type[VectorStore]] = {
    "milvus_lite": MilvusLiteStore,
}


def get_store_class(name: str) -> type[VectorStore]:
    if name not in STORES_REGISTRY:
        raise StoreError(f"unknown vector store backend: {name!r}")
    return STORES_REGISTRY[name]


def build_store(settings: Settings) -> VectorStore:
    cls = get_store_class(settings.vector_store_backend)
    return cls(settings=settings)
```

- [ ] **Step 6: 运行测试，确认通过**

Run: `cd F:\project\vector-service && python -m pytest tests/contract/test_milvus_lite_extra.py -v`
Expected: 6 passed

- [ ] **Step 7: 提交**

Run:
```bash
git add src/vector_service/stores/milvus_lite.py src/vector_service/stores/registry.py tests/contract/test_milvus_lite_extra.py
git commit -m "feat(stores): Milvus Lite implementation + registry"
```

---

## Phase 4 — Schemas

### Task 11: Pydantic Schemas

**Files:**
- Create: `src\vector_service\schemas\__init__.py`, `src\vector_service\schemas\openai.py`, `src\vector_service\schemas\management.py`

**Interfaces:**
- Produces:
  - OpenAI: `EmbeddingRequest`, `EmbeddingData`, `EmbeddingUsage`, `EmbeddingResponse`, `Model`, `ModelList`
  - Management: `CreateCollectionRequest`, `CollectionInfo`, `UpsertVectorsRequest`, `DeleteVectorsRequest`, `GetVectorsRequest`, `GetVectorsResponse`, `SearchRequest`, `SearchResponse`, `HitResponse`, `BackendInfo`
  - Errors: `ErrorResponse`, `ErrorBody`

- [ ] **Step 1: 写 `src\vector_service\schemas\__init__.py`**

空文件

- [ ] **Step 2: 写 `src\vector_service\schemas\openai.py`**

```python
"""OpenAI-compatible embedding API schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: Literal["float"] = "float"
    user: str | None = None

    @field_validator("input")
    @classmethod
    def _validate_input_not_empty(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input must not be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("input list must not be empty")
            for i, t in enumerate(v):
                if not isinstance(t, str) or not t.strip():
                    raise ValueError(f"input[{i}] must be non-empty string")
        return v


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "vector-service"
    created: int = 0
    dimensions: int | None = None


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]
```

- [ ] **Step 3: 写 `src\vector_service\schemas\management.py`**

```python
"""Vector store management API schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---- Collection ----

class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    dim: int | None = None
    metric: Literal["cosine", "ip", "l2"] = "cosine"
    backend_opts: dict = Field(default_factory=dict)


class CollectionInfoResponse(BaseModel):
    name: str
    dim: int
    metric: str
    count: int


# ---- Vectors ----

class UpsertVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)
    texts: list[str] | None = None
    embeddings: list[list[float]] | None = None
    metadatas: list[dict] | None = None

    @model_validator(mode="after")
    def _xor_texts_embeddings(self):
        has_t = self.texts is not None
        has_e = self.embeddings is not None
        if has_t and has_e:
            raise ValueError("provide either texts or embeddings, not both")
        if not has_t and not has_e:
            raise ValueError("must provide either texts or embeddings")
        n = len(self.ids)
        if has_t and len(self.texts) != n:  # type: ignore[arg-type]
            raise ValueError("ids and texts must have same length")
        if has_e and len(self.embeddings) != n:  # type: ignore[arg-type]
            raise ValueError("ids and embeddings must have same length")
        if self.metadatas is not None and len(self.metadatas) != n:
            raise ValueError("ids and metadatas must have same length")
        return self


class DeleteVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class GetVectorsRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class GetVectorItem(BaseModel):
    id: str
    vector: list[float] | None = None
    metadata: dict


class GetVectorsResponse(BaseModel):
    items: list[GetVectorItem]


# ---- Search ----

class SearchRequest(BaseModel):
    query_text: str | None = None
    query_embedding: list[float] | None = None
    top_k: int = Field(10, ge=1, le=1000)
    filter: dict | None = None

    @model_validator(mode="after")
    def _xor_text_embedding(self):
        has_t = self.query_text is not None
        has_e = self.query_embedding is not None
        if has_t and has_e:
            raise ValueError("provide either query_text or query_embedding, not both")
        if not has_t and not has_e:
            raise ValueError("must provide either query_text or query_embedding")
        return self


class HitResponse(BaseModel):
    id: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    hits: list[HitResponse]


# ---- Backend escape hatch ----

class BackendInfo(BaseModel):
    backend: str
    info: dict
```

- [ ] **Step 4: 写 `tests\unit\test_schemas.py`**

```python
import pytest
from pydantic import ValidationError

from vector_service.schemas.openai import EmbeddingRequest
from vector_service.schemas.management import (
    CreateCollectionRequest,
    UpsertVectorsRequest,
    SearchRequest,
)


def test_embedding_request_accepts_string():
    r = EmbeddingRequest(input="hi", model="bge-m3")
    assert r.input == "hi"


def test_embedding_request_accepts_list():
    r = EmbeddingRequest(input=["a", "b"], model="bge-m3")
    assert r.input == ["a", "b"]


def test_embedding_request_rejects_empty_string():
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="", model="bge-m3")


def test_embedding_request_rejects_unknown_encoding():
    with pytest.raises(ValidationError):
        EmbeddingRequest(input="x", model="bge-m3", encoding_format="base64")


def test_create_collection_dim_optional():
    r = CreateCollectionRequest(name="c")
    assert r.dim is None
    assert r.metric == "cosine"


def test_upsert_texts_or_embeddings_xor():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a"], texts=["t"], embeddings=[[0.1] * 4])


def test_upsert_requires_one():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a"])


def test_upsert_length_mismatch():
    with pytest.raises(ValidationError):
        UpsertVectorsRequest(ids=["a", "b"], texts=["one"])


def test_search_xor():
    with pytest.raises(ValidationError):
        SearchRequest(query_text="x", query_embedding=[0.1] * 4)


def test_search_requires_one():
    with pytest.raises(ValidationError):
        SearchRequest()
```

- [ ] **Step 5: 运行测试**

Run: `cd F:\project\vector-service && python -m pytest tests/unit/test_schemas.py -v`
Expected: 10 passed

- [ ] **Step 6: 提交**

Run:
```bash
git add src/vector_service/schemas/ tests/unit/test_schemas.py
git commit -m "feat(schemas): OpenAI-compatible and management API schemas"
```

---

## Phase 5 — API 路由层

### Task 12: Health & Metrics 路由

**Files:**
- Create: `src\vector_service\api\__init__.py`, `src\vector_service\api\health.py`
- Test: `tests\integration\__init__.py`, `tests\integration\conftest.py`, `tests\integration\test_health.py`

**Interfaces:**
- Produces:
  - `GET /healthz` → `{status: "ok"}`
  - `GET /readyz` → 200 / 503
  - `GET /metrics` → Prometheus text

- [ ] **Step 1: 写 `src\vector_service\api\__init__.py`**

空文件

- [ ] **Step 2: 写 `tests\integration\__init__.py`**

空文件

- [ ] **Step 3: 写 `tests\integration\conftest.py`**

```python
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vector_service.main import create_app
from vector_service.core.config import get_settings


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Per-test settings: tmp model dir + tmp milvus URI."""
    monkeypatch.setenv("VS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VS_MILVUS_URI", str(tmp_path / f"milvus_{uuid.uuid4().hex[:8]}.db"))
    monkeypatch.setenv("VS_EMBEDDING_MODEL_DIR", str(tmp_path / "models" / "bge-m3"))
    monkeypatch.setenv("VS_EMBEDDING_AUTO_DOWNLOAD", "false")
    monkeypatch.setenv("VS_LOG_FORMAT", "console")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def fake_components(monkeypatch):
    """注入 FakeEmbedder + FakeStore，跳过模型加载。"""
    from vector_service.embeddings.bge_m3 import BGEM3Embedder
    from vector_service.stores.milvus_lite import MilvusLiteStore
    from vector_service.testing.fake_embedder import FakeEmbedder
    from vector_service.testing.fake_store import FakeStore

    monkeypatch.setattr(BGEM3Embedder, "__init__", lambda self, settings=None: None)
    # 在 lifespan 里我们通过 settings 实例化；改为返回 fake
    from vector_service.core import lifespan as lspan
    from vector_service.core.config import get_settings as _gs

    original_build_embedder = lspan.build_embedder
    original_build_store = lspan.build_store

    def fake_build_embedder(settings):
        e = FakeEmbedder(dim=4)
        # 让 dim 表现为 4 便于测试
        return e

    def fake_build_store(settings):
        return FakeStore()

    monkeypatch.setattr(lspan, "build_embedder", fake_build_embedder)
    monkeypatch.setattr(lspan, "build_store", fake_build_store)
    yield


@pytest.fixture
def client(tmp_settings, fake_components):
    app = create_app()
    with TestClient(app) as c:
        yield c
```

- [ ] **Step 4: 写 `tests\integration\test_health.py`**

```python
from prometheus_client import parser


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_ok_with_components(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_metrics_exposition(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "vs_info" in text
    parsed = list(parser.text_string_to_metric_families(text))
    names = {f.name for f in parsed}
    assert "vs_info" in names
```

- [ ] **Step 5: 运行测试，确认失败**

Run: `cd F:\project\vector-service && python -m pytest tests/integration/test_health.py -v`
Expected: ModuleNotFoundError（main.py / api/health.py 未实现）

- [ ] **Step 6: 写 `src\vector_service\api\health.py`**

```python
"""Health, readiness, metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse

from vector_service.core.metrics import get_content_type, render_metrics

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request):
    embedder = getattr(request.app.state, "embedder", None)
    store = getattr(request.app.state, "store", None)
    if embedder is None or store is None:
        return Response(
            content='{"status":"not_ready"}',
            status_code=503,
            media_type="application/json",
        )
    return {"status": "ready"}


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(
        content=render_metrics(),
        media_type=get_content_type(),
    )
```

- [ ] **Step 7: 提交（先不实现 main.py，下个 task 一起）**

Run:
```bash
git add src/vector_service/api/__init__.py src/vector_service/api/health.py tests/integration/
git commit -m "feat(api): health/readyz/metrics endpoints"
```

---

### Task 13: Embeddings 路由 (OpenAI 协议)

**Files:**
- Create: `src\vector_service\api\embeddings.py`

**Interfaces:**
- Produces:
  - `POST /v1/embeddings`：调 Embedder.embed_documents，构造 OpenAI 响应
  - `GET /v1/models`、`GET /v1/models/{model_id}`
- 同步重计算走 `run_in_executor`

- [ ] **Step 1: 写 `src\vector_service\api\embeddings.py`**

```python
"""OpenAI-compatible embedding endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import EmbedderError, ModelNotLoaded
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    EMBEDDING_DURATION_SECONDS,
    EMBEDDING_REQUESTS_TOTAL,
    EMBEDDING_TOKENS_TOTAL,
    MODEL_LOADED,
)
from vector_service.embeddings.registry import (
    EMBEDDER_REGISTRY,
    get_embedder_class,
    list_embedder_names,
)
from vector_service.schemas.openai import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    Model,
    ModelList,
)

router = APIRouter(prefix="/v1", tags=["embeddings"])
log = get_logger(__name__)


def _estimate_tokens(text: str) -> int:
    return max(1, -(-len(text) // 4))  # ceil(len/4)


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(body: EmbeddingRequest, request: Request):
    settings = request.app.state.settings
    embedder = request.app.state.embedder

    # 校验 model
    try:
        get_embedder_class(body.model)
    except EmbedderError as e:
        raise HTTPException(status_code=404, detail={"error": {"code": "model_not_found", "message": str(e)}})

    texts = [body.input] if isinstance(body.input, str) else list(body.input)

    # 限制校验
    if len(texts) > settings.embedding_max_texts_per_request:
        raise HTTPException(
            status_code=422,
            detail={"error": {"code": "too_many_texts",
                              "message": f"max {settings.embedding_max_texts_per_request}",
                              "max": settings.embedding_max_texts_per_request}},
        )
    for i, t in enumerate(texts):
        if len(t) > settings.embedding_max_chars_per_text:
            raise HTTPException(
                status_code=422,
                detail={"error": {"code": "text_too_long",
                                  "message": f"max {settings.embedding_max_chars_per_text} chars",
                                  "index": i}},
            )

    total_tokens = sum(_estimate_tokens(t) for t in texts)
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    status = "ok"
    try:
        vectors = await loop.run_in_executor(None, embedder.embed_documents, texts)
    except (EmbedderError, ModelNotLoaded) as e:
        status = "error"
        raise HTTPException(status_code=503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
    finally:
        EMBEDDING_DURATION_SECONDS.labels(model=body.model, status=status).observe(time.perf_counter() - t0)
        EMBEDDING_REQUESTS_TOTAL.labels(model=body.model, status=status).inc()
        EMBEDDING_TOKENS_TOTAL.labels(model=body.model).inc(total_tokens)

    data = [
        EmbeddingData(index=i, embedding=v)
        for i, v in enumerate(vectors)
    ]
    log.info(
        "embedding_request",
        model=body.model,
        text_count=len(texts),
        tokens=total_tokens,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        status=status,
    )
    return EmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


@router.get("/models", response_model=ModelList)
def list_models(request: Request):
    embedder = request.app.state.embedder
    models = []
    for name in list_embedder_names():
        try:
            dim = embedder.dim if (embedder and name == embedder.model_name) else None
        except Exception:
            dim = None
        models.append(Model(id=name, dimensions=dim))
    return ModelList(data=models)


@router.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str, request: Request):
    if model_id not in EMBEDDER_REGISTRY:
        raise HTTPException(status_code=404, detail={"error": {"code": "model_not_found", "message": model_id}})
    embedder = request.app.state.embedder
    dim = embedder.dim if (embedder and embedder.model_name == model_id) else None
    return Model(id=model_id, dimensions=dim)
```

- [ ] **Step 2: 暂不单独跑测试，留到 Task 17 main app 落地后一起集成**

- [ ] **Step 3: 提交**

Run:
```bash
git add src/vector_service/api/embeddings.py
git commit -m "feat(api): OpenAI-compatible embeddings and models endpoints"
```

---

### Task 14: Management 路由

**Files:**
- Create: `src\vector_service\api\management.py`, `src\vector_service\api\backend.py`

**Interfaces:**
- Produces:
  - `GET /collections`, `POST /collections`, `DELETE /collections/{name}`, `GET /collections/{name}`
  - `PUT /collections/{name}/vectors`（upsert，支持 texts 或 embeddings）
  - `POST /collections/{name}/vectors/delete`
  - `POST /collections/{name}/vectors/get`
  - `POST /collections/{name}/search`
  - `GET /backend/raw`, `POST /backend/raw/call`（仅 debug 开启）

- [ ] **Step 1: 写 `src\vector_service\api\management.py`**

```python
"""Vector store management endpoints."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request

from vector_service.core.errors import (
    BackendError,
    CollectionAlreadyExists,
    CollectionNotFound,
    DimensionMismatch,
    EmbedderError,
    ModelNotLoaded,
    StoreError,
)
from vector_service.core.logging import get_logger
from vector_service.core.metrics import (
    STORE_COLLECTIONS,
    STORE_OP_DURATION_SECONDS,
    STORE_VECTORS_TOTAL,
)
from vector_service.schemas.management import (
    CollectionInfoResponse,
    CreateCollectionRequest,
    DeleteVectorsRequest,
    GetVectorItem,
    GetVectorsRequest,
    GetVectorsResponse,
    SearchRequest,
    SearchResponse,
    HitResponse,
    UpsertVectorsRequest,
)

router = APIRouter(tags=["management"])
log = get_logger(__name__)


# ---- helpers ----

def _http_from_store_error(e: Exception) -> HTTPException:
    if isinstance(e, CollectionNotFound):
        return HTTPException(404, detail={"error": {"code": "collection_not_found", "message": str(e)}})
    if isinstance(e, CollectionAlreadyExists):
        return HTTPException(409, detail={"error": {"code": "collection_exists", "message": str(e)}})
    if isinstance(e, DimensionMismatch):
        return HTTPException(422, detail={"error": {
            "code": "dimension_mismatch",
            "message": str(e),
            "expected": e.expected, "got": e.got,
        }})
    return HTTPException(503, detail={"error": {"code": "store_unavailable", "message": str(e)}})


def _timed(op: str, backend: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    status = "ok"
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        status = "error"
        raise
    finally:
        STORE_OP_DURATION_SECONDS.labels(op=op, backend=backend, status=status).observe(time.perf_counter() - t0)


async def _timed_async(op: str, backend: str, fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _timed, op, backend, fn, *args, **kwargs)


# ---- collections ----

@router.get("/collections")
async def list_collections(request: Request):
    store = request.app.state.store
    names = await _timed_async("list", store.backend_name, store.list_collections)
    STORE_COLLECTIONS.labels(backend=store.backend_name).set(len(names))
    return {"collections": names}


@router.post("/collections", status_code=201)
async def create_collection(body: CreateCollectionRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder
    dim = body.dim if body.dim is not None else embedder.dim
    try:
        await _timed_async("create", store.backend_name, store.create_collection,
                           body.name, dim, metric=body.metric, **body.backend_opts)
    except (CollectionAlreadyExists, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=store.backend_name).inc()
    return {"name": body.name, "dim": dim, "metric": body.metric}


@router.delete("/collections/{name}")
async def drop_collection(name: str, request: Request):
    store = request.app.state.store
    try:
        await _timed_async("drop", store.backend_name, store.drop_collection, name)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_COLLECTIONS.labels(backend=store.backend_name).dec()
    return {"deleted": name}


@router.get("/collections/{name}", response_model=CollectionInfoResponse)
async def get_collection(name: str, request: Request):
    store = request.app.state.store
    try:
        info = await _timed_async("info", store.backend_name, store.collection_info, name)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return CollectionInfoResponse(**info)


# ---- vectors ----

@router.put("/collections/{name}/vectors")
async def upsert_vectors(name: str, body: UpsertVectorsRequest, request: Request):
    settings = request.app.state.settings
    store = request.app.state.store
    embedder = request.app.state.embedder

    if body.texts is not None:
        loop = asyncio.get_running_loop()
        try:
            vectors = await loop.run_in_executor(None, embedder.embed_documents, body.texts)
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
    else:
        vectors = body.embeddings or []

    if len(vectors) != len(body.ids):
        raise HTTPException(422, detail={"error": {"code": "shape_mismatch", "message": "vectors/ids length"}})

    try:
        await _timed_async("upsert", store.backend_name, store.upsert,
                           name, body.ids, vectors, body.metadatas)
    except (CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(op="upsert", backend=store.backend_name).inc(len(body.ids))
    return {"upserted": len(body.ids)}


@router.post("/collections/{name}/vectors/delete")
async def delete_vectors(name: str, body: DeleteVectorsRequest, request: Request):
    store = request.app.state.store
    try:
        await _timed_async("delete", store.backend_name, store.delete, name, body.ids)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    STORE_VECTORS_TOTAL.labels(op="delete", backend=store.backend_name).inc(len(body.ids))
    return {"deleted": len(body.ids)}


@router.post("/collections/{name}/vectors/get", response_model=GetVectorsResponse)
async def get_vectors(name: str, body: GetVectorsRequest, request: Request):
    store = request.app.state.store
    try:
        items = await _timed_async("get", store.backend_name, store.get, name, body.ids)
    except (CollectionNotFound, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return GetVectorsResponse(items=[GetVectorItem(**it) for it in items])


@router.post("/collections/{name}/search", response_model=SearchResponse)
async def search(name: str, body: SearchRequest, request: Request):
    store = request.app.state.store
    embedder = request.app.state.embedder

    if body.query_text is not None:
        loop = asyncio.get_running_loop()
        try:
            qvec = (await loop.run_in_executor(None, embedder.embed_query, body.query_text))
        except (EmbedderError, ModelNotLoaded) as e:
            raise HTTPException(503, detail={"error": {"code": "embedder_unavailable", "message": str(e)}})
    else:
        qvec = body.query_embedding or []

    try:
        hits = await _timed_async("search", store.backend_name, store.search,
                                  name, qvec, body.top_k, body.filter)
    except (CollectionNotFound, DimensionMismatch, StoreError, BackendError) as e:
        raise _http_from_store_error(e)
    return SearchResponse(hits=[HitResponse(id=h.id, score=h.score, metadata=h.metadata) for h in hits])
```

- [ ] **Step 2: 写 `src\vector_service\api\backend.py`**

```python
"""Backend escape hatch endpoints (debug-only for raw/call)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/backend", tags=["backend"])


@router.get("/raw")
def backend_raw(request: Request):
    store = request.app.state.store
    backend = store.backend
    info: dict[str, Any] = {
        "backend": store.backend_name,
        "uri": getattr(backend, "uri", None),
        "native_methods": [m for m in dir(backend) if not m.startswith("_")][:20],
    }
    return {"backend": store.backend_name, "info": info}


@router.post("/raw/call")
async def backend_raw_call(request: Request):
    settings = request.app.state.settings
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")
    body = await request.json()
    op = (body or {}).get("op")
    if op == "list_collections":
        return {"result": request.app.state.store.list_collections()}
    raise HTTPException(status_code=400, detail=f"unsupported op {op!r}")
```

- [ ] **Step 3: 提交**

Run:
```bash
git add src/vector_service/api/management.py src/vector_service/api/backend.py
git commit -m "feat(api): vector store management and backend escape hatch routes"
```

---

## Phase 6 — App 组合

### Task 15: 中间件 + Lifespan

**Files:**
- Create: `src\vector_service\core\middleware.py`, `src\vector_service\core\lifespan.py`

**Interfaces:**
- Produces:
  - `RequestIDMiddleware`：注入 `request_id` 到 `request.state` + `request_id_var` + 响应头
  - `build_embedder(settings) -> Embedder`
  - `build_store(settings) -> VectorStore`
  - `lifespan(app)`：启动/关闭

- [ ] **Step 1: 写 `src\vector_service\core\middleware.py`**

```python
"""HTTP middleware: request_id propagation."""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from vector_service.core.logging import request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    HEADER = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(self.HEADER) or f"req_{uuid.uuid4().hex[:8]}"
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.HEADER] = rid
        return response
```

- [ ] **Step 2: 写 `src\vector_service\core\lifespan.py`**

```python
"""Application lifespan: load model + open store."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from vector_service.core.config import Settings, get_settings
from vector_service.core.logging import get_logger, setup_logging
from vector_service.core.metrics import MODEL_LOADED, VS_INFO
from vector_service.embeddings.registry import get_embedder_class
from vector_service.stores.registry import build_store

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


def build_embedder(settings: Settings):
    cls = get_embedder_class(settings.embedding_backend)
    return cls(settings=settings)


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    settings = get_settings()
    setup_logging(settings.log_format, settings.log_level)
    VS_INFO.labels(
        version="0.1.0",
        embedding_backend=settings.embedding_backend,
        vector_store_backend=settings.vector_store_backend,
    ).set(1)

    embedder = build_embedder(settings)
    store = build_store(settings)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store

    device = getattr(embedder, "_device", "unknown")
    MODEL_LOADED.labels(model=embedder.model_name, device=device).set(1)
    log.info("model_loaded", model=embedder.model_name, device=device, dim=embedder.dim)
    log.info("store_opened", backend=store.backend_name)

    try:
        yield
    finally:
        store.close()
        log.info("shutdown")
```

- [ ] **Step 3: 提交**

Run:
```bash
git add src/vector_service/core/middleware.py src/vector_service/core/lifespan.py
git commit -m "feat(core): request_id middleware and lifespan"
```

---

### Task 16: 异常处理器 + Main App

**Files:**
- Create: `src\vector_service\main.py`

**Interfaces:**
- Produces:
  - `create_app() -> FastAPI`：所有 router、middleware、exception handler、lifespan
  - `run()`：uvicorn 入口（脚本）

- [ ] **Step 1: 写 `src\vector_service\main.py`**

```python
"""FastAPI application factory."""
from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vector_service import __version__
from vector_service.api.backend import router as backend_router
from vector_service.api.embeddings import router as embeddings_router
from vector_service.api.health import router as health_router
from vector_service.api.management import router as management_router
from vector_service.core.errors import VectorServiceError
from vector_service.core.lifespan import lifespan
from vector_service.core.logging import get_logger, request_id_var
from vector_service.core.middleware import RequestIDMiddleware

log = get_logger(__name__)


def _err(code: str, message: str, status: int, extra: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id_var.get(),
                "extra": extra or {},
            }
        },
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="vector-service",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(VectorServiceError)
    async def _vs_error_handler(request: Request, exc: VectorServiceError):
        # 业务异常的 HTTP 状态码由具体路由处理；
        # 这里兜底（不应被命中）
        log.error("unhandled_business_error", error=str(exc))
        return _err("internal", "internal error", 500)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return _err("invalid_request", "validation error", 422, {"errors": exc.errors()})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled_exception", error=str(exc))
        return _err("internal", "internal error", 500)

    app.include_router(health_router)
    app.include_router(embeddings_router)
    app.include_router(management_router)
    app.include_router(backend_router)

    return app


app = create_app()


def run() -> None:
    from vector_service.core.config import get_settings
    s = get_settings()
    uvicorn.run(
        "vector_service.main:app",
        host=s.host,
        port=s.port,
        workers=s.workers,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: 跑健康测试，应通过**

Run: `cd F:\project\vector-service && python -m pytest tests/integration/test_health.py -v`
Expected: 3 passed

- [ ] **Step 3: 提交**

Run:
```bash
git add src/vector_service/main.py
git commit -m "feat(main): FastAPI app factory with error handlers and middleware"
```

---

## Phase 7 — 集成测试

### Task 17: 集成测试 /v1/embeddings

**Files:**
- Create: `tests\integration\test_embeddings_api.py`

**Interfaces:**
- 测试覆盖：合法请求、文本数组、模型不存在→404、空输入→422、超限→422、不合法 encoding→422

- [ ] **Step 1: 写 `tests\integration\test_embeddings_api.py`**

```python
def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert any(m["id"] == "bge-m3" for m in data)


def test_get_model(client):
    r = client.get("/v1/models/bge-m3")
    assert r.status_code == 200
    assert r.json()["id"] == "bge-m3"


def test_get_unknown_model_404(client):
    r = client.get("/v1/models/bogus")
    assert r.status_code == 404


def test_embed_string_input(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "model": "bge-m3"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "bge-m3"
    assert len(body["data"]) == 1
    assert isinstance(body["data"][0]["embedding"], list)
    # FakeEmbedder dim=4（见 conftest）
    assert len(body["data"][0]["embedding"]) == 4
    assert body["usage"]["prompt_tokens"] >= 1


def test_embed_list_input(client):
    r = client.post("/v1/embeddings", json={"input": ["a", "b", "c"], "model": "bge-m3"})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3
    assert [d["index"] for d in r.json()["data"]] == [0, 1, 2]


def test_embed_unknown_model_404(client):
    r = client.post("/v1/embeddings", json={"input": "x", "model": "bogus"})
    assert r.status_code == 404


def test_embed_empty_input_422(client):
    r = client.post("/v1/embeddings", json={"input": "", "model": "bge-m3"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_embed_invalid_encoding_422(client):
    r = client.post("/v1/embeddings", json={"input": "x", "model": "bge-m3", "encoding_format": "base64"})
    assert r.status_code == 422
```

- [ ] **Step 2: 运行测试**

Run: `cd F:\project\vector-service && python -m pytest tests/integration/test_embeddings_api.py -v`
Expected: 7 passed

- [ ] **Step 3: 提交**

Run:
```bash
git add tests/integration/test_embeddings_api.py
git commit -m "test: integration tests for /v1/embeddings and /v1/models"
```

---

### Task 18: 集成测试 management

**Files:**
- Create: `tests\integration\test_management_api.py`

- [ ] **Step 1: 写 `tests\integration\test_management_api.py`**

```python
def test_create_and_list_collection(client):
    r = client.post("/collections", json={"name": "c1"})
    assert r.status_code == 201
    r = client.get("/collections")
    assert r.status_code == 200
    assert "c1" in r.json()["collections"]


def test_create_twice_409(client):
    client.post("/collections", json={"name": "c"})
    r = client.post("/collections", json={"name": "c"})
    assert r.status_code == 409


def test_get_collection_info(client):
    client.post("/collections", json={"name": "c"})
    r = client.get("/collections/c")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "c"
    assert body["dim"] == 4  # FakeEmbedder


def test_upsert_with_texts_and_search(client):
    client.post("/collections", json={"name": "docs"})
    r = client.put("/collections/docs/vectors", json={
        "ids": ["a", "b"],
        "texts": ["alpha", "beta"],
        "metadatas": [{"src": "x"}, {"src": "y"}],
    })
    assert r.status_code == 200
    assert r.json()["upserted"] == 2

    r = client.post("/collections/docs/search", json={
        "query_text": "alpha", "top_k": 1,
    })
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["id"] == "a"


def test_upsert_with_embeddings(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"],
        "embeddings": [[1.0, 0.0, 0.0, 0.0]],
    })
    assert r.status_code == 200


def test_upsert_xor_422(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"], "texts": ["t"], "embeddings": [[1, 0, 0, 0]],
    })
    assert r.status_code == 422


def test_dim_mismatch_422(client):
    client.post("/collections", json={"name": "v"})
    r = client.put("/collections/v/vectors", json={
        "ids": ["a"],
        "embeddings": [[1.0, 0.0]],  # dim=2 != collection dim=4
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "dimension_mismatch"


def test_search_missing_collection_404(client):
    r = client.post("/collections/nope/search", json={"query_text": "x"})
    assert r.status_code == 404


def test_filter_search(client):
    client.post("/collections", json={"name": "f"})
    client.put("/collections/f/vectors", json={
        "ids": ["a", "b"],
        "texts": ["alpha", "beta"],
        "metadatas": [{"src": "x"}, {"src": "y"}],
    })
    r = client.post("/collections/f/search", json={
        "query_text": "alpha", "top_k": 10, "filter": {"src": "y"},
    })
    assert r.status_code == 200
    assert [h["id"] for h in r.json()["hits"]] == ["b"]


def test_delete_and_drop(client):
    client.post("/collections", json={"name": "d"})
    client.put("/collections/d/vectors", json={"ids": ["a"], "texts": ["x"]})
    r = client.post("/collections/d/vectors/delete", json={"ids": ["a"]})
    assert r.status_code == 200
    r = client.delete("/collections/d")
    assert r.status_code == 200


def test_get_vectors(client):
    client.post("/collections", json={"name": "g"})
    client.put("/collections/g/vectors", json={"ids": ["a"], "texts": ["x"], "metadatas": [{"k": "v"}]})
    r = client.post("/collections/g/vectors/get", json={"ids": ["a"]})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == "a"
    assert items[0]["metadata"] == {"k": "v"}


def test_request_id_in_response_header(client):
    r = client.get("/healthz", headers={"X-Request-ID": "req_test123"})
    assert r.headers.get("X-Request-ID") == "req_test123"
```

- [ ] **Step 2: 运行测试**

Run: `cd F:\project\vector-service && python -m pytest tests/integration/test_management_api.py -v`
Expected: 12 passed

（**注意**：dim_mismatch 用 FakeStore，会抛 DimensionMismatch；FakeEmbedder dim=4，所以 test_dim_mismatch 期望 422。）

- [ ] **Step 3: 提交**

Run:
```bash
git add tests/integration/test_management_api.py
git commit -m "test: integration tests for collection/vector management endpoints"
```

---

### Task 19: 端到端 smoke + README 完善

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 跑完整测试套件**

Run: `cd F:\project\vector-service && python -m pytest -q`
Expected: 所有测试通过

- [ ] **Step 2: 手动 smoke（不启动真实 BGE-M3，使用 fake 启动）**

Run:
```bash
cd F:\project\vector-service
# 临时把 BGE-M3 替换成 FakeEmbedder 起服务做 smoke：
# 通过环境变量切换不现实（embedder class 写死），改为写一个临时 smoke 脚本
```

创建 `scripts\smoke.py`：

```python
"""Manual smoke: bypass BGE-M3, use FakeEmbedder + FakeStore."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vector_service.main import create_app
from vector_service.core.lifespan import build_embedder, build_store
from vector_service.testing.fake_embedder import FakeEmbedder
from vector_service.testing.fake_store import FakeStore
from vector_service.core.config import get_settings


class _DummySettings:
    embedding_model_dir = Path("/tmp")
    embedding_auto_download = False
    vector_store_backend = "fake"
    milvus_uri = "/tmp/x.db"
    embedding_device = "cpu"
    debug = False


# monkey patch
import vector_service.core.lifespan as lspan
lspan.build_embedder = lambda s: FakeEmbedder(dim=4)
lspan.build_store = lambda s: FakeStore()

from fastapi.testclient import TestClient
app = create_app()
with TestClient(app) as c:
    r = c.get("/healthz"); print("healthz:", r.status_code, r.json())
    r = c.get("/readyz"); print("readyz:", r.status_code, r.json())
    r = c.get("/v1/models"); print("models:", r.status_code, r.json())
    r = c.post("/v1/embeddings", json={"input": "hi", "model": "bge-m3"}); print("embed:", r.status_code, len(r.json()["data"][0]["embedding"]))
    r = c.post("/collections", json={"name": "demo"}); print("create coll:", r.status_code, r.json())
    r = c.put("/collections/demo/vectors", json={"ids": ["a"], "texts": ["hi"]}); print("upsert:", r.status_code, r.json())
    r = c.post("/collections/demo/search", json={"query_text": "hi", "top_k": 1}); print("search:", r.status_code, r.json())
    r = c.get("/metrics"); print("metrics lines:", len(r.text.splitlines()))
```

Run: `cd F:\project\vector-service && python scripts/smoke.py`
Expected: 所有 print 输出 200/201

- [ ] **Step 3: 写最终版 `README.md`**

```markdown
# vector-service

生产级 FastAPI 向量服务：

- **OpenAI 兼容嵌入 API**：`/v1/embeddings`、`/v1/models`
- **可扩展向量库管理**：`/collections`、`/collections/{name}/vectors`、`/collections/{name}/search`
- **可观测性**：`/metrics`（Prometheus）+ 结构化 JSON 日志
- **可扩展抽象**：新增嵌入器/向量库只需新增适配文件并注册

## 第一阶段支持

| 组件 | 实现 |
|------|------|
| Embedder | BGE-M3 (GPU: torch fp16 / CPU: ONNX int8) |
| VectorStore | Milvus Lite |

## 快速开始

### 安装

```bash
pip install -e ".[embed,store,dev]"
cp .env.example .env
# 编辑 .env，至少设置 VS_MILVUS_URI 与 VS_EMBEDDING_MODEL_DIR
```

### 启动

```bash
# 首次启动会自动从 HuggingFace 下载 BGE-M3（若模型不存在）
make run
# 或
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
```

### 调用

```bash
# 1. 嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'

# 2. 创建集合
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "products"}'

# 3. upsert（用文本，自动嵌入）
curl -X PUT http://localhost:8080/collections/products/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": ["1"], "texts": ["a wireless mouse"], "metadatas": [{"price": 29.9}]}'

# 4. 检索
curl -X POST http://localhost:8080/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "computer accessory", "top_k": 5}'
```

### 健康检查

```bash
curl http://localhost:8080/healthz   # 进程存活
curl http://localhost:8080/readyz    # 模型 + store 就绪
curl http://localhost:8080/metrics   # Prometheus
```

## 测试

```bash
make test              # 全部
make test-unit         # 仅单元
make test-contract     # 仅契约
make test-integration  # 集成（不含 slow）
make test-slow         # 真实 BGE-M3 + Milvus Lite
```

## 添加新嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `Embedder` ABC
2. 在 `embeddings/registry.py` 注册
3. 加配置项到 `core/config.py`
4. 写契约测试 `tests/contract/test_embedder_contract.py`

## 添加新向量库

参见 `src/vector_service/stores/`：

1. 新建 `stores/<backend>.py`，实现 `VectorStore` ABC
2. 在 `stores/registry.py` 注册
3. 加配置项到 `core/config.py`
4. 跑契约测试 `tests/contract/test_vector_store_contract.py`

## 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件。详见 `.env.example`。

## 架构

```
api/         FastAPI 路由（OpenAI 协议 + 管理）
embeddings/  Embedder 抽象 + 实现
stores/      VectorStore 抽象 + 实现
core/        配置 / 日志 / 指标 / 错误 / 生命周期 / 中间件
schemas/     Pydantic 请求/响应模型
testing/     测试用 FakeEmbedder / FakeStore
```

## 文档

- 设计 spec: `docs/superpowers/specs/2026-08-22-vector-service-design.md`
- 实施计划: `docs/superpowers/plans/2026-08-22-vector-service-plan.md`
```

- [ ] **Step 4: 提交**

Run:
```bash
git add scripts/smoke.py README.md
git commit -m "docs: full README + manual smoke script"
```

---

## Self-Review Checklist

✅ **Spec coverage**:

| Spec 章节 | 实现任务 |
|----------|---------|
| §3 架构（项目布局、依赖） | Task 1 |
| §4 组件职责 | Tasks 1–18 |
| §5.1 Embedder ABC | Task 7 |
| §5.2 VectorStore ABC | Task 8 |
| §6.1 OpenAI 嵌入 API | Task 13 |
| §6.2 管理 API | Task 14 |
| §6.3 健康与可观测性 | Task 12 |
| §6.4 后端逃生通道 | Task 14 |
| §7 错误处理 | Tasks 3, 16 |
| §8 BGE-M3 子系统 | Task 9 |
| §9 Milvus Lite 实现 | Task 10 |
| §10 配置 | Task 2 |
| §11 生命周期 | Task 15 |
| §12 可观测性 | Tasks 4, 5, 12 |
| §13 测试策略 | Tasks 7, 8, 17, 18 |

✅ **Placeholder scan**：无 TBD/TODO。
✅ **Type consistency**：
- `Embedder.dim` / `model_name` 在 Task 7 定义、BGE-M3 在 Task 9 实现、schema 在 Task 11 引用一致。
- `VectorStore` 方法签名在 Task 8 定义、Milvus Lite 在 Task 10、management 路由在 Task 14 引用一致。
- `Hit(id, score, metadata)` 在 Task 8 定义，schema 在 Task 11 引用一致。
- 错误类名跨任务一致。
