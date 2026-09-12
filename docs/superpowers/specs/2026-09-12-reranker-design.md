# 向量服务重排序 (reranker) 子系统设计规格

- **日期**：2026-09-12
- **状态**：待用户审阅
- **作者**：通过 brainstorming 流程与用户协作产出
- **范围**：在现有 vector-service 上新增 reranker 子系统，独立 `POST /v1/rerank` 端点 + 配套抽象、配置、指标、测试。不改变现有 `/v1/embeddings`、`/v1/databases/.../search` 行为。

## 1. 概述

为 vector-service 增加 cross-encoder 风格的本地重排序能力。Reranker 接收一段 query 和一组候选 documents，返回按相关性分数排序的索引列表（不回传原文）。客户端可在自己已有的 RAG 流程里把检索召回的 top-k 候选过一遍 reranker，再截断到最终 top-n。

### 1.1 范围

**包含（v1）**：
- 新模块 `src/vector_service/rerankers/`，与 `embeddings/`、`stores/` 平级
- `Reranker` ABC + `ScoredHit` dataclass + 模块级 `RERANKER_REGISTRY` 工厂模式
- `CrossEncoderReranker` 实现：sentence-transformers `CrossEncoder`，默认模型 `BAAI/bge-reranker-v2-m3`
- 配置：`RerankerSettings`（嵌套于 `Settings.reranker`），环境变量前缀 `VS_RERANKER__`
- 路由：`POST /v1/rerank`、`GET /v1/rerank/models`
- 健康：`/readyz` 增 `reranker` 诊断字段，不阻塞 ready
- 错误：新增 `RerankerError`、`RerankerNotLoaded`，映射到 503
- 指标：`RERANK_DURATION_SECONDS`、`RERANK_REQUESTS_TOTAL`
- 测试：4 个单元测试文件 + 1 个契约测试文件（默认跳过）
- 依赖：仅在 `pyproject.toml` 的 `[embed]` extra 显式声明 `sentence-transformers>=2.6` 最低版本

**不包含（v1）**：
- 嵌入到现有 `/search` 路由做后置 rerank（独立端点为主）
- 远程 rerank 后端（Jina、Cohere、SiliconFlow 等）
- 鉴权 / 速率限制 / 计费
- 多 reranker 实例并存（v1 单实例，进程级单例）
- 运行期热切换模型

### 1.2 与已有模块的关系

| 现有模块 | 与 reranker 的关系 |
|---|---|
| `embeddings/` | 同级独立包；不互相 import |
| `stores/` | 同级独立包；`/search` 路由不动 |
| `core/config.py` | 新增嵌套 `Settings.reranker` 字段 |
| `core/lifespan.py` | 在 `lifespan()` 中追加 `build_reranker + load()` 步骤 |
| `core/errors.py` | 新增 `RerankerError`、`RerankerNotLoaded` |
| `core/metrics.py` | 新增两个指标；`MODEL_LOADED` 改为带 `kind` 标签 |
| `core/middleware.py` | 不变（`RequestIDMiddleware` 直接复用） |
| `api/embeddings.py` | 不动 |
| `api/management.py` | 不动（`/search` 路由不接 rerank） |
| `api/health.py` | `/readyz` 增 `reranker` 字段；`/healthz`、`/metrics` 不变 |
| `main.py` | 注册新 router、OPENAPI tag、异常 handler |
| `schemas/errors.py` | 不动（沿用现有 `ErrorEnvelope`） |

## 2. 目标与决策摘要

| 维度 | 决策 |
|---|---|
| 端点形态 | 独立 `POST /v1/rerank`（不嵌入 `/search`） |
| 后端 | sentence-transformers `CrossEncoder`，本地权重 |
| 默认模型 | `BAAI/bge-reranker-v2-m3` |
| 配置 | 纯环境变量 `VS_RERANKER__*`，启动时加载，不支持运行期热改 |
| 启动行为 | `VS_RERANKER__BACKEND` 必填具体后端名；加载失败 → lifespan 抛错退出 |
| 响应粒度 | 只回 `{index, score}`，不返回原文、不返回 metadata |
| 输入限制 | 与 embeddings 对齐：256 docs / 8192 chars / 2048 chars query / top_n ∈ [1, 64] |
| 抽象层级 | `Reranker` ABC + `RERANKER_REGISTRY`（与 embedder/store 模式 1:1 对称） |
| 异步 | `loop.run_in_executor` 派发，与 `/v1/embeddings` 模式一致 |
| 可观测性 | `RERANK_DURATION_SECONDS{model}`、`RERANK_REQUESTS_TOTAL{model,status}`、结构化日志、`/readyz` 增 `reranker` 字段 |
| 测试 | 不引入共享 `conftest.py`；契约测试在 Windows 默认跳过 |

## 3. 架构

### 3.1 组件依赖

```
api/rerank.py   → rerankers/    → core/
                              ↘ (sentence-transformers, torch)

rerankers/  与 embeddings/、stores/ 互不依赖、互不感知。
组合发生在 api 层。
```

### 3.2 新增模块布局

```
src/vector_service/
├── rerankers/
│   ├── __init__.py
│   ├── base.py              # Reranker ABC + ScoredHit dataclass
│   ├── cross_encoder.py     # CrossEncoderReranker
│   └── registry.py          # RERANKER_REGISTRY + get_reranker_class + list_reranker_names
├── api/
│   └── rerank.py            # POST /v1/rerank, GET /v1/rerank/models
└── schemas/
    └── rerank.py            # RerankRequest / RerankResultItem / RerankResponse / RerankerInfo / RerankModelsResponse
```

### 3.3 `Reranker` ABC 形状

```python
# rerankers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class ScoredHit:
    index: int   # 原始 documents 列表中的位置
    score: float # reranker 输出的相关性分数（不归一化）

class Reranker(ABC):
    model_name: str  # class attr, e.g. "bge-reranker-v2-m3"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[ScoredHit]: ...
    # 返回结果已按 score 降序，已截 top_n（若给出）
```

### 3.4 `CrossEncoderReranker` 形状

```python
# rerankers/cross_encoder.py
class CrossEncoderReranker(Reranker):
    model_name = "bge-reranker-v2-m3"

    def __init__(self, settings: Settings | None = None):
        s = (settings or get_settings()).reranker
        self._device = self._resolve_device(s.device)
        self._batch_size = s.batch_size
        self._max_length = s.max_length
        self._model_dir = s.model_dir
        self._auto_download = s.auto_download
        self._download_source = s.download_source
        self._hf_repo = s.hf_repo
        self._ms_repo = s.ms_repo
        self._impl: CrossEncoder | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._impl is not None:
                return
            self._ensure_model_dir()
            self._impl = CrossEncoder(
                self._model_dir,
                max_length=self._max_length,
                device=self._device,
            )
            self._impl.predict([("warmup", "")] * 5, batch_size=self._batch_size,
                               show_progress_bar=False)

    def rerank(self, query, documents, top_n=None):
        self._ensure_loaded()
        pairs = [(query, d) for d in documents]
        scores = self._impl.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        ).tolist()
        hits = sorted(
            [ScoredHit(index=i, score=float(s)) for i, s in enumerate(scores)],
            key=lambda h: h.score, reverse=True,
        )
        if top_n is not None:
            hits = hits[:top_n]
        return hits
```

`_resolve_device` 复用 BGE-M3 的 `auto|cpu|cuda` 回落逻辑；`_ensure_model_dir` 复用 `_dir_has_model` + `modelscope`/`huggingface` snapshot_download 模式（`_dir_has_model` 检查 `config.json`、`tokenizer_config.json`、`model.safetensors` 中任一存在即可）。

### 3.5 `RERANKER_REGISTRY` 起始

```python
# rerankers/registry.py
from vector_service.rerankers.base import Reranker
from vector_service.rerankers.cross_encoder import CrossEncoderReranker

RERANKER_REGISTRY: dict[str, type[Reranker]] = {
    "bge-reranker-v2-m3": CrossEncoderReranker,
}

def get_reranker_class(name: str) -> type[Reranker]:
    cls = RERANKER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(name)
    return cls

def list_reranker_names() -> list[str]:
    return list(RERANKER_REGISTRY)
```

## 4. 数据流

### 4.0 `GET /v1/rerank/models`

列出 `RERANKER_REGISTRY` 中所有已注册的 reranker 后端名。返回 `RerankModelsResponse { data: [{name}, ...] }`，不依赖 reranker 实例已加载（注册表是模块级常量）。无路径参数，无错误码。

### 4.1 `POST /v1/rerank` 完整路径

```
POST /v1/rerank
  body: RerankRequest { query, documents, top_n?, model? }

RequestIDMiddleware → 注入 request_id
  ↓
api/rerank.py::rerank(request, settings, reranker)
  1. model = req.model or settings.reranker.backend
  2. registry 校验: model ∈ list_reranker_names()       → 404 model_not_found
  3. settings 校验:
     - len(documents) ≤ max_documents_per_request       → 422 too_many_documents
     - 任一 len(d) ≤ max_chars_per_doc                  → 422 document_too_long
     - len(query) ≤ max_query_chars                     → 422 query_too_long
     - top_n ∈ [1, max_top_n]                           → 422 invalid_top_n
  4. RERANK_REQUESTS_TOTAL{model, "received"}.inc()
  5. T0 = perf_counter()
     loop.run_in_executor(reranker.rerank, query, documents, top_n)
       → cross_encoder.predict([(query, d) for d in documents])
       → list[ScoredHit] (已按 score 降序、已截 top_n)
  6. T1 = perf_counter()
     RERANK_DURATION_SECONDS{model}.observe(T1 - T0)
     RERANK_REQUESTS_TOTAL{model, "ok"|"error"}.inc()
  7. log.info("rerank_completed", model, n_docs, top_n, latency_ms, request_id)
  8. return RerankResponse { model, results, request_id }
```

### 4.2 响应形状

```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 3, "score": 0.873},
    {"index": 0, "score": 0.412},
    {"index": 5, "score": 0.198}
  ],
  "request_id": "req_a3f1b2c4"
}
```

`results` 始终按 score 降序；长度 = `min(top_n, len(documents))`。

### 4.3 错误码矩阵

| HTTP | code | 触发 |
|---|---|---|
| 404 | `model_not_found` | `model` 不在 registry |
| 422 | `invalid_request` | Pydantic schema 拒绝 |
| 422 | `too_many_documents` | `len(documents) > max_documents_per_request` |
| 422 | `document_too_long` | 任一 `len(d) > max_chars_per_doc` |
| 422 | `query_too_long` | `len(query) > max_query_chars` |
| 422 | `invalid_top_n` | `top_n` 不在 `[1, max_top_n]` 范围或非正整数 |
| 503 | `reranker_not_loaded` | lifespan 未加载成功（兜底） |
| 503 | `reranker_error` | 推理时异常（GPU OOM、模型损坏等） |
| 500 | `internal` | 兜底 |

错误信封沿用现有 `ErrorEnvelope`，不新发明。

## 5. 配置

### 5.1 `RerankerSettings`

```python
# core/config.py 增量
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VS_RERANKER__", extra="ignore")

    backend: str = Field(..., description="Reranker backend name; must exist in RERANKER_REGISTRY")

    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_dir: str = "./models/bge-reranker-v2-m3"
    auto_download: bool = True
    download_source: Literal["huggingface", "modelscope"] = "modelscope"
    hf_repo: str = "BAAI/bge-reranker-v2-m3"
    ms_repo: str = "BAAI/bge-reranker-v2-m3"

    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(32, ge=1, le=512)
    max_length: int = Field(512, ge=1, le=8192)

    max_documents_per_request: int = Field(256, ge=1, le=4096)
    max_chars_per_doc: int = Field(8192, ge=1, le=32768)
    max_query_chars: int = Field(2048, ge=1, le=8192)
    max_top_n: int = Field(64, ge=1, le=1024)
    top_n_default: int = Field(10, ge=1, le=1024)

    @field_validator("top_n_default")
    @classmethod
    def _top_n_default_le_max(cls, v, info):
        max_top_n = info.data.get("max_top_n", 64)
        if v > max_top_n:
            raise ValueError(f"top_n_default ({v}) must be <= max_top_n ({max_top_n})")
        return v

class Settings(BaseSettings):  # 在现有类里追加
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
```

### 5.2 `.env` 示例

```bash
VS_RERANKER__BACKEND=bge-reranker-v2-m3
VS_RERANKER__MODEL_DIR=./models/bge-reranker-v2-m3
VS_RERANKER__DEVICE=auto
VS_RERANKER__BATCH_SIZE=32
VS_RERANKER__MAX_DOCUMENTS_PER_REQUEST=256
VS_RERANKER__MAX_QUERY_CHARS=2048
VS_RERANKER__MAX_CHARS_PER_DOC=8192
VS_RERANKER__TOP_N_DEFAULT=10
VS_RERANKER__MAX_TOP_N=64
```

## 6. Schemas

```python
# schemas/rerank.py
from pydantic import BaseModel, ConfigDict, Field

class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = Field(None, ge=1)
    model: str | None = Field(None, description="Backend name; defaults to settings.reranker.backend")

class RerankResultItem(BaseModel):
    index: int = Field(..., ge=0)
    score: float

class RerankResponse(BaseModel):
    model: str
    results: list[RerankResultItem]
    request_id: str

class RerankerInfo(BaseModel):
    name: str

class RerankModelsResponse(BaseModel):
    data: list[RerankerInfo]
```

Pydantic schema 只做 `min_length=1`（documents 非空、query 非空）；长度上限、字符上限、top_n 范围在 route 层基于 settings 返回 422，明确错误 `code`。

## 7. 生命周期

### 7.1 `lifespan()` 增量

```python
# core/lifespan.py
from vector_service.rerankers.registry import get_reranker_class

def build_reranker(settings: Settings) -> Reranker:
    cls = get_reranker_class(settings.reranker.backend)
    return cls(settings=settings)

# lifespan() 内：
try:
    reranker = build_reranker(settings)
    reranker.load()
    app.state.reranker = reranker
    MODEL_LOADED.labels(kind="reranker").set(1)
    log.info("reranker_loaded", model=reranker.model_name, backend=settings.reranker.backend)
except (RerankerNotLoaded, RerankerError, KeyError) as exc:
    MODEL_LOADED.labels(kind="reranker").set(0)
    log.error("reranker_load_failed", backend=settings.reranker.backend, error=str(exc))
    raise  # lifespan 退出，启动失败
```

启动顺序：`load embedder → open store → load reranker`。reranker 放最后是因为它不参与 search 路径，延后加载可让 embedder/store 先就绪。

### 7.2 异常处理（main.py 增量）

```python
# main.py
@app.exception_handler(RerankerNotLoaded)
async def _rerank_not_loaded(request, exc):
    return _err("reranker_not_loaded", str(exc), 503, exc=exc)

@app.exception_handler(RerankerError)
async def _rerank_error(request, exc):
    return _err("reranker_error", str(exc) or "reranker failed", 503, exc=exc)
```

## 8. 可观测性

### 8.1 指标（core/metrics.py 增量）

```python
RERANK_DURATION_SECONDS = Histogram(
    "vs_rerank_duration_seconds",
    "Rerank latency in seconds (route handler end-to-end).",
    labels=["model"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

RERANK_REQUESTS_TOTAL = Counter(
    "vs_rerank_requests_total",
    "Total rerank requests, labelled by model and outcome.",
    labels=["model", "status"],  # status ∈ {"received", "ok", "error"}
)
```

### 8.2 `MODEL_LOADED` 标签变更

当前 `MODEL_LOADED`（core/metrics.py）无标签，仅标记 embedder 状态。改为带 `labels=["kind"]`：
- `MODEL_LOADED.labels(kind="embedder").set(0|1)`
- `MODEL_LOADED.labels(kind="reranker").set(0|1)`

`kind` 标签初始化时同时注册 `embedder=0, reranker=0`，保证 `/readyz` 与 metrics 输出始终可见。

### 8.3 `/readyz` 增量

```python
# api/health.py 增量（在原响应里加 reranker 字段）
{
  "status": "ready" | "degraded" | "not_ready",
  "store": "ready" | "...",
  "embedder": "ready" | "not_loaded",
  "reranker": "ready" | "not_loaded"  # 新增；不阻塞 status
}
```

`status` 计算仍只看 embedder + store；`reranker` 字段是诊断信号（便于 ops 在 reranker 加载失败时快速定位，但不让 k8s readiness probe 抖动）。

### 8.4 日志

成功：
```json
{"event": "rerank_completed", "model": "bge-reranker-v2-m3", "n_docs": 12,
 "top_n": 5, "latency_ms": 87, "request_id": "req_a3f1b2c4", "level": "info"}
```

失败：
```json
{"event": "rerank_failed", "model": "...", "n_docs": 12, "top_n": 5,
 "error_type": "RuntimeError", "error": "...", "request_id": "...", "level": "warning"}
```

## 9. 测试

### 9.1 单元测试（不依赖真实模型）

| 文件 | 覆盖 |
|---|---|
| `tests/unit/test_reranker_load.py` | `Reranker` 是 ABC（`TypeError`）；`_FakeReranker` 验证 `_impl` 状态、load 幂等、`RerankerNotLoaded` 传播；lifespan 集成 `/readyz` 含 `reranker` 字段、`MODEL_LOADED(kind="reranker")` 同步置 1/0 |
| `tests/unit/test_rerank_route.py` | `api/rerank.py` 路由：404 `model_not_found`、4 个 422 错误码、`top_n` 默认值生效、`results` 按 score 降序、`results` 长度 = `min(top_n, len(docs))`、`request_id` 回写、metrics `received`+`ok` 计数、Pydantic 拒绝空 documents |
| `tests/unit/test_reranker_settings.py` | `RerankerSettings` 校验：`backend` 必填、`top_n_default > max_top_n` 触发 `ValueError`、`device/batch_size/max_*` 范围越界报错、`download_source` literal 校验 |
| `tests/unit/test_rerank_schemas.py` | `RerankRequest` `extra="forbid"`、`query min_length=1`、`documents min_length=1`、`top_n ge=1` |

每个测试文件自带 `_make_app(reranker=...)` 工厂，沿用项目"无共享 conftest"风格。

### 9.2 `_FakeReranker`（test_reranker_load.py 内私有）

```python
class _FakeReranker(Reranker):
    model_name = "fake-reranker"
    def __init__(self, settings=None):
        self._impl = None
    def load(self):
        self._impl = "ready"
    def rerank(self, query, documents, top_n=None):
        n = len(documents) if top_n is None else min(top_n, len(documents))
        # 分数单调递减、可预测；让测试断言排序与截断
        scored = sorted(
            [ScoredHit(index=i, score=1.0 / (i + 1)) for i in range(len(documents))],
            key=lambda h: h.score, reverse=True,
        )
        return scored[:n]
```

### 9.3 契约测试（默认跳过）

`tests/contract/test_cross_encoder_reranker.py`：标 `@pytest.mark.contract` + `@pytest.mark.slow`，沿用 `b86e077` 风格在 Windows 默认跳过。

```python
@pytest.mark.contract
@pytest.mark.slow
def test_bge_reranker_v2_m3_orders_relevant_first():
    settings = Settings(reranker=RerankerSettings(
        backend="bge-reranker-v2-m3", model_dir="./models/bge-reranker-v2-m3"))
    r = CrossEncoderReranker(settings=settings)
    r.load()
    docs = ["巴黎是法国首都", "苹果是一种水果", "北京是中华人民共和国的首都"]
    hits = r.rerank("中国首都", docs, top_n=3)
    assert hits[0].index == 2
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
```

### 9.4 测试规模估算

- 4 个单元测试文件，合计 ~15 个测试函数
- 1 个契约测试文件，1 个测试函数 + 2~3 个 fixture 参数化
- 单测总耗时 < 2s；契约测试 ~30s

## 10. 依赖

```toml
# pyproject.toml [embed] extra 显式声明 sentence-transformers 最低版本
[project.optional-dependencies]
embed = [
    "torch>=2.2",
    "onnxruntime>=1.18",
    "FlagEmbedding>=1.2.10",
    "transformers>=4.41",
    "huggingface-hub>=0.23",
    "modelscope>=1.9",
    "sentence-transformers>=2.6",  # 新增
]
```

不引入新顶级依赖；`sentence-transformers` 已在 venv 中，列为最低版本要求。

## 11. 运行与运维

### 11.1 启动

```bash
# 最小配置（默认模型 + 默认限制）
VS_RERANKER__BACKEND=bge-reranker-v2-m3 \
uv run vector-service
```

启动流程：加载 BGE-M3 embedder → 连接 Milvus store → 加载 bge-reranker-v2-m3 → 监听 :8080。任意一步失败 lifespan 抛错退出。

### 11.2 验证

```bash
# 健康
curl localhost:8080/healthz   # 进程存活
curl localhost:8080/readyz    # 检查 embedder / store / reranker

# 列出模型
curl localhost:8080/v1/rerank/models

# 重排序
curl -X POST localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "中国首都",
    "documents": ["巴黎是法国首都", "苹果是一种水果", "北京是中华人民共和国的首都"],
    "top_n": 3
  }'
```

### 11.3 指标

```bash
curl localhost:8080/metrics | grep -E 'vs_rerank|MODEL_LOADED'
```

输出含 `vs_rerank_duration_seconds_bucket{model=...}`、`vs_rerank_requests_total{model=...,status=...}`、`vs_model_loaded{kind="reranker"} 1.0`。

## 12. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `sentence-transformers` 与 BGE-M3 依赖冲突 | 现有 BGE-M3 走 `FlagEmbedding`，不直接 import sentence-transformers；新加的 `sentence-transformers` 仅被 `CrossEncoderReranker` 使用，依赖交集为空 |
| 模型权重首次下载阻塞 lifespan | 默认 `download_source=modelscope`（国内源）；`auto_download=False` 时目录缺关键文件 → `RerankerNotLoaded`，lifespan 退出由用户决定 |
| `CrossEncoder.predict` 在大 batch 时内存峰值 | `batch_size` 默认 32，可通过 `VS_RERANKER__BATCH_SIZE` 调小 |
| Windows 上 `torch`+CUDA 不可用 | 沿用 BGE-M3 模式，`device="auto"` 回落 CPU |
| `MODEL_LOADED` 改标签是破坏性 metrics 变更 | 项目起步阶段，无 dashboard/alert 依赖；变更同时初始化 `embedder=0, reranker=0` |
| reranker 与 embedder 同时加载，首启时间翻倍 | 接受；未来可拆 background warmup task，但 v1 不做 |

## 13. 验收标准

1. `uv run vector-service` 在最小配置下成功启动并监听 :8080
2. `POST /v1/rerank` 在正确输入下返回 `RerankResponse`，`results` 按 score 降序
3. 4 类 422 错误码（`too_many_documents`、`document_too_long`、`query_too_long`、`invalid_top_n`）均可触发并返回正确 `code`
4. `model="unknown"` 时返回 404 `model_not_found`
5. `VS_RERANKER__BACKEND` 缺失或加载失败时 lifespan 抛错退出（非静默降级）
6. `/readyz` 响应包含 `reranker` 字段；reranker 加载失败时该字段为 `not_loaded` 且 status 仍按 embedder/store 判定
7. 单元测试 4 个文件全部通过；契约测试在 Linux/CI 通过、Windows 默认跳过
8. `pyproject.toml` 安装 `[embed]` extra 后无新增顶级依赖冲突
