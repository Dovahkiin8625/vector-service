# 向量服务 (vector-service) 设计规格

- **日期**：2026-08-22
- **状态**：待用户审阅
- **作者**：通过 brainstorming 流程与用户协作产出

## 1. 概述

构建一个生产级 FastAPI 向量服务，提供两类能力：

1. **向量嵌入**：OpenAI 协议兼容的 embedding API（`/v1/embeddings` 等）。
2. **向量管理**：以统一抽象封装底层向量库，提供集合与向量的 CRUD + 检索。

### 范围

**包含**：
- FastAPI 应用，单进程部署
- BGE-M3 嵌入模型，支持 GPU（PyTorch fp16）与 CPU（ONNX int8）两种后端
- Milvus Lite 向量库后端（第一阶段唯一）
- 可插拔的嵌入器和向量库抽象，未来可扩展更多实现
- 结构化日志 + Prometheus 指标
- 配置走环境变量 / `.env`（pydantic-settings）

**不包含（v1）**：
- 多租户隔离 / 鉴权
- BGE-M3 的 sparse / ColBERT 多向量输出
- 多 worker 横向扩展的安全保证
- 性能基准与混沌测试

### 后续扩展（不在本规格，但接口预留）

- 嵌入器：OpenAI 嵌入 API、TEI、FlagEmbedding、vLLM 等
- 向量库：Milvus（standalone/cluster）、Chroma、FAISS、zvec

## 2. 目标与决策摘要

| 维度 | 决策 |
|------|------|
| 部署拓扑 | 单进程 FastAPI |
| 向量库抽象 | 统一最小 API + 各后端原生能力逃生通道 |
| 集合/租户 | 平面命名集合，无租户概念，调用方自管数据隔离 |
| BGE-M3 输出 | 仅 dense（1024 维） |
| 鉴权 | v1 不做，靠网关隔离 |
| 模型加载 | 本地路径 + 可选启动时下载 HF ONNX/Torch |
| 异步模型 | FastAPI async 路由 + 重计算丢 `run_in_executor` |
| 可观测性 | 结构化 JSON 日志 + Prometheus `/metrics` |
| 配置 | 环境变量 + `.env`（pydantic-settings，统一前缀 `VS_`） |

## 3. 架构

### 3.1 组件依赖

```
api/         → embeddings/  → core/
            → stores/       → core/

embeddings/  与 stores/ 互不依赖、互不感知。
组合发生在 api 层。
```

### 3.2 项目布局

```
F:\project\vector-service\
├── src/vector_service/
│   ├── api/
│   │   ├── embeddings.py     # /v1/embeddings, /v1/models
│   │   └── management.py     # /collections, /vectors, /search
│   ├── embeddings/
│   │   ├── base.py           # Embedder ABC
│   │   ├── bge_m3.py         # BGE-M3 ONNX/Torch 实现
│   │   └── registry.py       # 模型名 → 实现工厂
│   ├── stores/
│   │   ├── base.py           # VectorStore ABC
│   │   ├── milvus_lite.py    # 第一阶段实现
│   │   └── registry.py
│   ├── core/
│   │   ├── config.py         # pydantic-settings
│   │   ├── logging.py        # structlog JSON
│   │   ├── metrics.py        # Prometheus
│   │   ├── errors.py         # 异常层级
│   │   ├── lifespan.py       # 启动/关闭
│   │   └── middleware.py     # request_id 中间件
│   ├── schemas/              # Pydantic 请求/响应
│   └── main.py               # FastAPI app 工厂
├── tests/{unit,contract,integration}/
├── models/                   # 本地模型文件（gitignored）
├── data/                     # Milvus Lite 文件（gitignored）
├── docs/superpowers/specs/
├── .env.example
├── pyproject.toml
├── README.md
└── Makefile
```

### 3.3 数据流

**嵌入请求**：
```
POST /v1/embeddings
  → EmbeddingsRouter（异步）
    → loop.run_in_executor(None, embedder.embed_documents, texts)
      → BGE-M3（ONNX 或 Torch）
    → 构造 OpenAI 格式响应
  → JSON
```

**管理请求**（以 `upsert texts` 为例）：
```
PUT /collections/{name}/vectors  body={ids, texts}
  → ManagementRouter
    → loop.run_in_executor(embedder.embed_documents, texts)
    → loop.run_in_executor(store.upsert, name, ids, vectors, metadatas)
      → Milvus Lite
  → JSON
```

## 4. 组件职责

| 模块 | 职责 | 不做什么 |
|------|------|----------|
| `api/embeddings.py` | OpenAI 协议路由、输入校验、调 Embedder、转响应 | 不存向量、不管集合 |
| `api/management.py` | 向量库管理路由、调 Store、可选自动嵌入 | 不直接读模型 |
| `embeddings/base.py` | `Embedder` ABC | 不做异步、不做缓存 |
| `embeddings/bge_m3.py` | BGE-M3 实现：ONNX session（CPU）或 Torch 模型（GPU）、批大小、max_length | 不感知 HTTP/路由 |
| `embeddings/registry.py` | `EMBEDDER_REGISTRY: dict[str, type[Embedder]]`、`get_embedder(name)` | 不做实例化（lifespan 负责） |
| `stores/base.py` | `VectorStore` ABC | 不做嵌入、不做鉴权 |
| `stores/milvus_lite.py` | Milvus Lite 实现：连接、schema 映射、索引参数；`self.backend` 作为原生逃生通道 | 不感知 HTTP |
| `core/config.py` | `Settings` (BaseSettings)，所有可调参数 | 不做 IO |
| `core/lifespan.py` | 启动：加载模型 + 打开 store；关闭：释放资源 | 不注册路由 |
| `core/errors.py` | 异常类；FastAPI handler 映射 HTTP 状态 | 不做日志 |
| `core/metrics.py` | 暴露指标定义 | 不做业务逻辑 |
| `core/logging.py` | 配置 structlog；输出 JSON | 不管业务 |

## 5. 抽象契约

### 5.1 Embedder

```python
class Embedder(ABC):
    dim: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

- `embed_query` 与 `embed_documents` 必须为同一语义空间（一致维度）。
- 同步方法；异步化由调用方用 `run_in_executor` 完成。
- 不抛 HTTP 异常，业务异常用 `EmbedderError`。

### 5.2 VectorStore

```python
@dataclass
class Hit:
    id: str
    score: float
    metadata: dict

class VectorStore(ABC):
    backend_name: str

    def create_collection(
        self, name: str, dim: int, *, metric: str = "cosine", **backend_opts
    ) -> None: ...
    def drop_collection(self, name: str) -> None: ...
    def list_collections(self) -> list[str]: ...
    def collection_info(self, name: str) -> dict: ...
    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None: ...
    def delete(self, collection: str, ids: list[str]) -> None: ...
    def get(self, collection: str, ids: list[str]) -> list[dict]: ...
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[Hit]: ...

    @property
    def backend(self) -> Any:
        """原生客户端/连接，仅用于调用后端独有 API。"""
        ...
```

- ABC 内方法返标准化结构；后端独有能力通过 `backend_opts` 透传 + `self.backend` 暴露。
- `metric` 取值集合：`"cosine"`, `"ip"`, `"l2"`（各后端内部映射）。
- `filter` 为统一 dict，**实现层负责翻译**为后端原生表达式（如 Milvus expr）。

## 6. API 表面

### 6.1 嵌入接口（OpenAI 兼容）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/embeddings` | OpenAI 协议 |
| `GET`  | `/v1/models` | 列出已注册模型 |
| `GET`  | `/v1/models/{model_id}` | 单模型元信息 |

**`/v1/embeddings` 请求**：
```json
{
  "input": "string | string[]",
  "model": "bge-m3",
  "encoding_format": "float",   // 可选，仅接受 "float"
  "user": "optional-trace-id"   // 可选
}
```

**`/v1/embeddings` 响应**：
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.034, ...]}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 128, "total_tokens": 128}
}
```

`prompt_tokens` 计算方式：按字符 ÷ 4 估算（粗略，足够计量）。后续可换成 tiktoken 或 model 自带 tokenizer。

### 6.2 管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`    | `/collections` | 列出集合 |
| `POST`   | `/collections` | 创建 |
| `DELETE` | `/collections/{name}` | 删除 |
| `GET`    | `/collections/{name}` | 元信息 |
| `PUT`    | `/collections/{name}/vectors` | upsert |
| `POST`   | `/collections/{name}/vectors/delete` | 删除 |
| `POST`   | `/collections/{name}/vectors/get` | 按 id 取 |
| `POST`   | `/collections/{name}/search` | 检索 |

**`POST /collections`** 请求：
```json
{ "name": "products", "dim": null, "metric": "cosine", "backend_opts": {} }
```
`dim` 为 `null` 时使用当前 embedder 的 `dim`。

**`PUT /collections/{name}/vectors`** 请求：
```json
{
  "ids": ["a", "b"],
  "texts": ["hello", "world"],
  "embeddings": null,
  "metadatas": [{"src": "doc1"}, {}]
}
```
`texts` 与 `embeddings` 二选一；同时给或都不给 → 422。

**`POST /collections/{name}/search`** 请求：
```json
{
  "query_text": "find me similar docs",
  "query_embedding": null,
  "top_k": 10,
  "filter": {"source": "docs"}
}
```
`query_text` 与 `query_embedding` 二选一；同时给或都不给 → 422。

**`POST /collections/{name}/search`** 响应：
```json
{
  "hits": [
    {"id": "a", "score": 0.92, "metadata": {"src": "doc1"}}
  ]
}
```

### 6.3 健康与可观测性

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 进程存活 |
| `GET` | `/readyz` | 模型已加载 + store 已打开 |
| `GET` | `/metrics` | Prometheus 文本 |

### 6.4 后端逃生通道

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/backend/raw` | 返 `{backend: "milvus_lite", info: {...}}`，不暴露客户端对象 |
| `POST` | `/backend/raw/call` | **仅在 `VS_DEBUG=true` 时启用**，透传原生调用（请求体带 `op` 名） |

## 7. 错误处理

### 7.1 异常层级

```python
class VectorServiceError(Exception):
    """基类。"""

class EmbedderError(VectorServiceError):
    """推理失败或模型不可用。"""

class ModelNotLoaded(EmbedderError):
    """lifespan 阶段加载失败或被关闭。"""

class StoreError(VectorServiceError):
    """向量库 IO 失败。"""

class CollectionNotFound(StoreError):
    """→ 404。"""

class CollectionAlreadyExists(StoreError):
    """→ 409。"""

class DimensionMismatch(StoreError):
    """→ 422。"""

class BackendError(StoreError):
    """后端原生异常包装。"""
```

### 7.2 HTTP 映射

| 异常类 | HTTP | 响应 `error.code` |
|--------|------|-------------------|
| `EmbedderError`, `ModelNotLoaded` | 503 | `embedder_unavailable` |
| `StoreError`, `BackendError` | 503 | `store_unavailable` |
| `CollectionNotFound` | 404 | `collection_not_found` |
| `CollectionAlreadyExists` | 409 | `collection_exists` |
| `DimensionMismatch` | 422 | `dimension_mismatch` |
| `pydantic.ValidationError` | 422 | （标准 Pydantic 错误） |
| 未捕获 | 500 | `internal`（不泄露栈） |

错误响应统一格式：
```json
{
  "error": {
    "code": "string",
    "message": "human-readable",
    "request_id": "req_xxx",
    "extra": {}
  }
}
```

### 7.3 不变式

- 不向客户端泄露栈/路径/敏感信息。
- 每条错误响应携带 `request_id`，与日志同源。
- 后端原生异常必须翻译，不允许原始堆栈透传。

## 8. BGE-M3 嵌入子系统

### 8.1 加载策略

- **配置**：
  - `VS_EMBEDDING_BACKEND=bge-m3`
  - `VS_EMBEDDING_DEVICE=auto|cpu|cuda`
  - `VS_EMBEDDING_MODEL_DIR=./models/bge-m3`
  - `VS_EMBEDDING_AUTO_DOWNLOAD=true`
  - `VS_EMBEDDING_HF_REPO=BAAI/bge-m3`
  - `VS_EMBEDDING_ONNX_REPO=BAAI/bge-m3-onnx`
- **设备自动选择**：CUDA 可用 → GPU（Torch）；否则 → CPU（ONNX）。
- **GPU 路径**：`torch` 加载 BGE-M3，fp16 推理。
- **CPU 路径**：`onnxruntime` + int8 ONNX 模型（若 ONNX 仓库可用）。否则降级为 Torch + CPU。
- **缺失处理**：
  - 路径不存在或关键文件缺失：
    - `VS_EMBEDDING_AUTO_DOWNLOAD=true` → 从 `VS_EMBEDDING_HF_REPO` 下载，按 device 选择 Torch 或 ONNX 资产
    - `false` → 启动失败，错误信息明确指出期望路径
- **预热**：启动后跑 5 条空字符串触发算子/JIT 编译。

### 8.2 批处理与限制

- `VS_EMBEDDING_BATCH_SIZE=32`：CPU 路径默认；GPU 路径启动时探测显存自动放大（≤64）。
- `VS_EMBEDDING_MAX_LENGTH=512`：超过截断（尾部）+ 日志 warning。
- `VS_EMBEDDING_MAX_TEXTS_PER_REQUEST=256`：超过 → 422。
- `VS_EMBEDDING_MAX_CHARS_PER_TEXT=8192`：单文本字符上限（粗略防护），超过 → 422。

### 8.3 文本归一化

- `embed_query(text)`：在传入文本前加 BGE 官方前缀 `为这个句子生成表示以用于检索：`
- `embed_documents(texts)`：不加前缀

### 8.4 输出

- 仅 dense 向量，1024 维 float32 → `list[float]`。
- 响应 `embedding` 序列化为 JSON 数组。`encoding_format` 仅接受 `"float"`，未来可扩展 `"base64"`（v1 不实现）。

### 8.5 token 计数

- 粗略估算：`tokens ≈ ceil(chars / 4)`
- 仅用于 `usage.prompt_tokens` / `usage.total_tokens`，不参与计费。

## 9. 向量库：Milvus Lite 实现要点

### 9.1 连接

- `connections.connect("default", uri=VS_MILVUS_URI)`
- `VS_MILVUS_URI` 默认 `./data/milvus.db`，启动时父目录自动创建。

### 9.2 Schema

- `id`: `VARCHAR(64)` primary key
- `vector`: `FLOAT_VECTOR(1024)`（dim 来自 embedder）
- `metadata`: `JSON` 字段（非 dynamic field，更稳）

### 9.3 索引

- 默认：`HNSW`，`M=16`、`efConstruction=200`
- `metric_type`: COSINE
- search 参数：`ef=64`

### 9.4 metadata 序列化

- dict → `json.dumps(...)` → 写入 Milvus
- 读时 `json.loads(...)` 还原

### 9.5 filter 翻译

- v1 范围：仅支持顶层 key/value 等值比较（`metadata["k"] == "v"`），AND 组合
- 复杂表达式（OR、嵌套）→ 422 + 提示"filter 表达式能力依赖后端，请使用 `/backend/raw/call`"
- 翻译器为 Milvus-specific 实现，存于 `stores/milvus_lite.py` 内部，不污染 ABC

## 10. 配置

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # 服务
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"
    log_format: str = "json"     # json | console
    debug: bool = False

    # 嵌入
    embedding_backend: str = "bge-m3"
    embedding_model_dir: Path = Path("./models/bge-m3")
    embedding_auto_download: bool = True
    embedding_device: str = "auto"  # auto | cpu | cuda
    embedding_batch_size: int = 32
    embedding_max_length: int = 512
    embedding_max_texts_per_request: int = 256
    embedding_max_chars_per_text: int = 8192
    embedding_hf_repo: str = "BAAI/bge-m3"
    embedding_onnx_repo: str = "BAAI/bge-m3-onnx"

    # 向量库
    vector_store_backend: str = "milvus_lite"
    milvus_uri: str = "./data/milvus.db"

    # 运行时
    data_dir: Path = Path("./data")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VS_", extra="ignore")
```

`.env.example` 列出所有项及默认值。

## 11. 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_format, settings.log_level)
    setup_metrics()

    embedder = build_embedder(settings)
    store = build_store(settings)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.store = store

    try:
        yield
    finally:
        store.close()
```

`build_embedder` 与 `build_store` 根据 `Settings` 决定具体实现（注册表 + factory）。

## 12. 可观测性

### 12.1 结构化日志

- `structlog` 输出 JSON
- 中间件注入 `request_id`（优先 `X-Request-ID` 头，否则生成 `req_<uuid8>`）
- 关键事件：
  - `startup`, `shutdown`
  - `model_loaded`（带 `model`, `device`, `duration_ms`）
  - `store_opened`（带 `backend`, `uri`）
  - `embedding_request`（`request_id`, `model`, `text_count`, `tokens`, `duration_ms`, `status`）
  - `store_operation`（`op`, `backend`, `collection`, `duration_ms`, `status`）

### 12.2 Prometheus 指标

| 指标 | 类型 | 标签 | 说明 |
|------|------|------|------|
| `vs_embedding_duration_seconds` | Histogram | `model`, `status` | 嵌入耗时 |
| `vs_embedding_tokens_total` | Counter | `model` | 处理 token 数 |
| `vs_embedding_requests_total` | Counter | `model`, `status` | 请求计数 |
| `vs_store_operation_duration_seconds` | Histogram | `op`, `backend`, `status` | 向量库操作耗时 |
| `vs_store_collections` | Gauge | `backend` | 当前集合数 |
| `vs_store_vectors_total` | Counter | `op`, `backend` | upsert/delete 计数 |
| `vs_model_loaded` | Gauge | `model`, `device` | 1 = 已加载 |
| `vs_info` | Gauge | `version`, `embedding_backend`, `vector_store_backend` | 静态信息 |

Histogram 桶：`[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]`。

### 12.3 健康检查

- `GET /healthz`：返 200 OK，不依赖 store/model。
- `GET /readyz`：检查 `app.state.embedder` 和 `app.state.store` 均非 None。200/503。

## 13. 测试策略

### 13.1 分层

| 层级 | 目录 | 范围 |
|------|------|------|
| 单元 | `tests/unit/` | 配置默认值、错误映射、Pydantic schema、filter 翻译器（mock）、日志格式 |
| 契约 | `tests/contract/` | 任何 `VectorStore` / `Embedder` 实现必须通过 |
| 集成 | `tests/integration/` | TestClient + 真实（或 tmp 路径）Milvus Lite + 真实 BGE-M3（标记 `@pytest.mark.slow`） |

### 13.2 契约测试示例

```python
def test_create_and_drop(store: VectorStore):
    store.create_collection("c1", dim=4)
    assert "c1" in store.list_collections()
    store.drop_collection("c1")

def test_upsert_search_roundtrip(store: VectorStore):
    store.create_collection("c2", dim=4)
    store.upsert("c2", ids=["a", "b"],
                 vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
                 metadatas=[{"k": "v"}, {}])
    hits = store.search("c2", query_vector=[1, 0, 0, 0], top_k=2)
    assert hits[0].id == "a"

def test_dimension_mismatch(store: VectorStore):
    store.create_collection("c3", dim=4)
    with pytest.raises(DimensionMismatch):
        store.upsert("c3", ids=["x"], vectors=[[1, 0, 0]])

def test_filter_metadata(store: VectorStore): ...
def test_delete(store: VectorStore): ...
def test_collection_not_found_raises(store: VectorStore): ...
```

### 13.3 集成测试覆盖

- `/v1/embeddings` 返 OpenAI 格式
- `/v1/embeddings` 输入超限 → 422
- `/v1/embeddings` 模型不存在 → 404
- 完整 upsert/search/delete/drop 流程（用真实 BGE-M3 + Milvus Lite）
- `/metrics` 返回合法 Prometheus 文本（用 `prometheus_client.parser` 解析）

### 13.4 运行

```bash
pytest -q                          # 全部
pytest -q tests/unit/              # 仅单元
pytest -q tests/contract/          # 契约
pytest -q tests/integration/ -m "not slow"  # 集成，去掉 -m 跑全量
```

### 13.5 不在 v1 范围

- 性能压测 / 基准
- 故障注入 / 混沌测试
- 多 worker 并发安全验证

## 14. 依赖（pyproject.toml）

核心：
- `fastapi[standard]`（含 uvicorn）
- `pydantic>=2`，`pydantic-settings`
- `httpx`（测试用 TestClient + 未来调 OpenAI）
- `structlog`，`prometheus-client`

嵌入：
- `torch`（GPU 路径）
- `onnxruntime`（CPU 路径）
- `sentence-transformers`（作为 BGE-M3 入口的简化抽象，可选）
- `FlagEmbedding`（BGE-M3 原作者库，提供官方 ONNX 导出，作为第一阶段首选；Torch 路径走 `transformers` 备份）
- `huggingface-hub`（模型下载）

向量库：
- `pymilvus`（含 milvus-lite）

开发：
- `pytest`，`pytest-asyncio`，`pytest-cov`
- `ruff`（lint + format）
- `mypy`（类型检查）

## 15. 启动

```bash
# 安装
pip install -e .

# 启动（VS_* 环境变量或 .env）
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
# 或
python -m vector_service
```

## 16. 添加新嵌入器 / 新向量库

**新嵌入器**（如 OpenAI、TEI）：
1. `embeddings/<backend>.py` 实现 `Embedder`
2. `embeddings/registry.py` 加注册项
3. `Settings.embedding_backend` 校验包含新值
4. 写契约测试 + 集成测试

**新向量库**（如 chroma/faiss/zvec/milvus-standalone）：
1. `stores/<backend>.py` 实现 `VectorStore`
2. `stores/registry.py` 加注册项
3. 新增 `Settings` 配置块（如 `chroma_host`, `faiss_index_type`）
4. 跑契约测试确保通过
5. 集成测试接真实后端

## 17. 风险与缓解

| 风险 | 缓解 |
|------|------|
| BGE-M3 ONNX 官方导出仓库不完整或失效 | 启动时校验必需文件；缺失则 fallback 到 Torch + CPU；自动下载失败时给出明确错误并保留路径供手动下载 |
| Milvus Lite 写盘并发风险 | v1 单 worker；文档明示；后续加锁 |
| metadata 嵌套结构丢失 | v1 仅 flat dict 顶层；嵌套在日志中 warn，后续可加 BSON/Protobuf |
| GPU 显存不足 | 启动时探测，根据显存自动降 batch；OOM 异常包装为 `EmbedderError` |
| 嵌入器与向量库 dim 不一致 | `create_collection(dim=None)` 强制跟随当前 embedder；`upsert` 显式校验，写时检查 |

## 18. 验收标准

满足以下条件视为 v1 完成：

1. `pytest tests/unit tests/contract -q` 全过。
2. 集成测试在真实 BGE-M3 + Milvus Lite 下通过（可标 slow，默认 CI 不跑）。
3. `uvicorn vector_service.main:app` 启动后：
   - `GET /healthz` 返 200
   - `GET /readyz` 返 200
   - `GET /v1/models` 列出 `bge-m3`
   - `POST /v1/embeddings` 返回 OpenAI 格式 1024 维向量
   - `POST /collections` → `PUT /collections/X/vectors`（with texts）→ `POST /collections/X/search` 端到端走通
   - `GET /metrics` 含 `vs_*` 指标
4. `.env.example` 列出所有可调项；README 含启动、调用、扩展说明。