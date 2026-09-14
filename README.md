# vector-service

生产级 FastAPI 向量服务。

- **OpenAI 兼容嵌入 API**：`/v1/embeddings`、`/v1/models`
- **图像嵌入 API**：`/v1/image_embeddings`（OpenCLIP ViT-L/14，把 base64 图片转成 768 维向量）
- **图文跨模态嵌入 API**：`/v1/multimodal_embeddings`（Chinese-CLIP ViT-B/16，文本与图片共享 512 维投影空间，可用于文搜图 / 图搜文）
- **重排序 API**：`/v1/rerank`（cross-encoder，对向量检索结果二次重排）
- **多 database 管理**：`/v1/databases` 增删改查 database，每个 database 下挂若干 collection
- **database-scoped 向量库管理**：`/v1/databases/{db}/collections/{coll}/vectors|search|...`
- **直连 Milvus server**：通过 `pymilvus` 直接连接独立部署的 Milvus，无中间代理
- **可观测性**：`/metrics`（Prometheus）+ 结构化 JSON 日志
- **可扩展抽象**：增加新嵌入器只需新建适配文件并注册；增加新向量库后端在 `stores/` 下实现 `VectorStore` 接口并在 `registry.py` 分支

## 内容索引

- [架构定位](#架构定位)
- [快速上手](#快速上手)
- [项目结构](#项目结构)
- [配置](#配置)
- [API 表面](#api-表面)
- [嵌入与重排子系统](#嵌入与重排子系统)
  - [文本嵌入（BGE-M3）](#文本嵌入bge-m3)
  - [图像嵌入（OpenCLIP）](#图像嵌入openclip)
  - [图文跨模态嵌入（Chinese-CLIP）](#图文跨模态嵌入chinese-clip)
  - [重排序（cross-encoder）](#重排序cross-encoder)
- [向量库管理（Milvus CRUD）](#向量库管理milvus-crud)
- [统一错误信封](#统一错误信封)
- [测试](#测试)
- [添加新后端](#添加新后端)

## 第一阶段支持

| 组件 | 实现 |
|------|------|
| Embedder | BGE-M3 (GPU: torch fp16 / CPU: ONNX int8) |
| ImageEmbedder | OpenCLIP ViT-L/14 (openai 预训练权重, 768 维) |
| MultimodalEmbedder | Chinese-CLIP ViT-B/16 (OFA-Sys, 512 维共享投影空间) |
| Reranker | BGE-Reranker-v2-M3 (cross-encoder) |
| VectorStore | Milvus server（直连 pymilvus） |

---

## 架构定位

```
client ──HTTP──▶ vector-service ──gRPC──▶ Milvus server
                  ├─ BGE-M3 文本嵌入
                  ├─ OpenCLIP 图像嵌入
                  ├─ Chinese-CLIP 跨模态嵌入
                  ├─ BGE-Reranker 重排序
                  └─ Milvus store 客户端（pymilvus 直连）
```

`vector-service` 自己的职责有四块：

1. **文本嵌入**：用 BGE-M3 把文本变成 `list[float]`。
2. **图像嵌入**：用 OpenCLIP 把 base64 图片变成 `list[float]`。
3. **跨模态嵌入**：用 Chinese-CLIP 同时支持中文文本与图片，输出同一空间向量。
4. **向量库 CRUD**：通过 `pymilvus` 直接对 Milvus 做 database / collection / 向量管理。

---

## 快速上手

### 0. 前置条件

- Python ≥ 3.11
- 一台可访问的 Milvus server（standalone / cluster），版本 ≥ 2.4（推荐启用 native database）
- 可选：GPU + CUDA（如需用 torch fp16 后端跑 BGE-M3；否则自动落到 ONNX int8 CPU）

### 1. 准备 `.env`

```bash
cp .env.example .env
```

至少确认：

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `VS_HOST` / `VS_PORT` | 监听地址 | `0.0.0.0` / `8080` |
| `VS_LOG_LEVEL` / `VS_LOG_FORMAT` | 日志级别 / 格式（`json` / `console`） | `INFO` / `json` |
| `VS_EMBEDDING_BACKEND` | 文本嵌入器后端 | `bge-m3` |
| `VS_EMBEDDING_MODEL_DIR` | 模型本地目录 | `./models/bge-m3` |
| `VS_EMBEDDING_AUTO_DOWNLOAD` | 首次启动自动下载模型权重 | `true` |
| `VS_VECTOR_STORE_BACKEND` | 向量库后端（目前只支持 `milvus`） | `milvus` |
| `VS_MILVUS_URI` | Milvus server 地址（gRPC） | `http://localhost:19530` |
| `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD` | 账号密码（启用 auth 时填） | （空） |
| `VS_MILVUS_TOKEN` | 可选 token（优先级高于 user/password） | （空） |
| `VS_MILVUS_TIMEOUT` | 连接 / RPC 超时（秒） | `30` |
| `VS_RERANKER__BACKEND` | reranker 后端 | `bge-reranker-v2-m3` |
| `VS_IMAGE_EMBEDDING__BACKEND` | 图像嵌入器后端 | `openclip-vit-l-14` |
| `VS_MULTIMODAL_EMBEDDING__BACKEND` | 跨模态嵌入器后端 | `chinese-clip-vit-base-patch16` |

完整字段见 `.env.example`。

### 2. 启动 Milvus server

参考 [Milvus 官方文档](https://milvus.io/docs/install_standalone-docker.md) 起一个 standalone：

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.10/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
```

确认 `localhost:19530` 可达。

### 3. 安装依赖

```bash
uv venv --python 3.11
uv pip install -e ".[all]"
```

`[all]` 同时装 BGE-M3、pymilvus、open_clip_torch 与 transformers。也可以分开装：`uv pip install -e ".[embed,store,image-embed,multimodal-embed]"`。

### 4. 启动服务

```bash
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
```

或直接装 console_script：

```bash
vector-service
```

访问：

| 路径 | 内容 |
|------|------|
| `http://127.0.0.1:8080/dashboard` | 自带调试页面（每个 tab 都能点按钮调接口） |
| `http://127.0.0.1:8080/docs` | Swagger UI |
| `http://127.0.0.1:8080/redoc` | ReDoc |
| `http://127.0.0.1:8080/openapi.json` | OpenAPI 文档 |
| `http://127.0.0.1:8080/healthz` | 进程存活 |
| `http://127.0.0.1:8080/readyz` | 嵌入器加载 + Milvus 可达 |
| `http://127.0.0.1:8080/metrics` | Prometheus |

---

## 项目结构

```
vector-service/
├── pyproject.toml              # 项目元数据 + 依赖 + pytest 配置
├── README.md                   # 本文件
├── .env.example                # VS_* 环境变量示例
├── .gitignore
├── src/vector_service/
│   ├── main.py                 # FastAPI app 工厂 + 异常处理 + lifespan
│   ├── api/                    # FastAPI 路由
│   │   ├── health.py           # /healthz /readyz /metrics
│   │   ├── models.py           # /v1/models /v1/models/{id}
│   │   ├── embeddings.py       # /v1/embeddings
│   │   ├── image_embeddings.py # /v1/image_embeddings
│   │   ├── multimodal_embeddings.py # /v1/multimodal_embeddings
│   │   ├── rerank.py           # /v1/rerank
│   │   ├── management.py       # /v1/databases + /v1/databases/{db}/collections/*
│   │   ├── backend.py          # /backend/raw（运维诊断）
│   │   └── dashboard.py       # /dashboard 内嵌调试页面
│   ├── core/                   # 基础设施
│   │   ├── config.py           # pydantic-settings：VS_* + 子系统嵌套设置
│   │   ├── errors.py           # 异常体系（EmbedderError / StoreError / RerankerError / ...）
│   │   ├── lifespan.py         # 启动期加载模型、连接 Milvus
│   │   ├── logging.py          # structlog 结构化日志
│   │   ├── metrics.py          # Prometheus 指标
│   │   └── middleware.py       # RequestID 中间件
│   ├── embeddings/             # 文本/图像/跨模态嵌入器抽象与实现
│   │   ├── base.py             # Embedder ABC
│   │   ├── bge_m3.py           # BGE-M3 文本嵌入
│   │   ├── image_base.py       # ImageEmbedder ABC
│   │   ├── image_decoding.py   # base64 + MIME 校验
│   │   ├── openclip_vit_l14.py # OpenCLIP 图像嵌入
│   │   ├── multimodal_base.py  # MultimodalEmbedder ABC
│   │   ├── chinese_clip_multimodal.py  # Chinese-CLIP 跨模态
│   │   ├── registry.py         # EMBEDDER_REGISTRY
│   │   ├── image_registry.py   # IMAGE_EMBEDDER_REGISTRY
│   │   └── multimodal_registry.py      # MULTIMODAL_EMBEDDER_REGISTRY
│   ├── rerankers/              # 重排序
│   │   ├── base.py             # Reranker ABC
│   │   ├── cross_encoder.py    # BGE-Reranker-v2-M3
│   │   └── registry.py         # RERANKER_REGISTRY
│   ├── schemas/                # Pydantic 请求/响应模型
│   │   ├── openai.py           # Embedding / EmbeddingResponse / Model
│   │   ├── image_embeddings.py
│   │   ├── multimodal_embeddings.py
│   │   ├── rerank.py
│   │   ├── management.py       # database/collection/vector CRUD
│   │   └── errors.py           # ErrorEnvelope
│   └── stores/                 # 向量库抽象与实现
│       ├── base.py             # VectorStore ABC
│       ├── milvus.py           # MilvusStore（直连 pymilvus）
│       ├── _milvus_adapter.py  # pymilvus 适配层（内部实现细节）
│       └── registry.py         # build_store(settings)
├── tests/                      # 单元 + contract 测试
│   ├── unit/                   # 不依赖外部服务的快速测试
│   └── contract/               # 需要真实 Milvus / 模型（标记 contract）
├── docs/                       # 设计文档与计划
│   └── superpowers/{plans,specs}/
└── models/                     # 模型权重目录（运行时下载；已在 .gitignore）
```

---

## 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件。pydantic-settings 使用 `__`（双下划线）作为嵌套分隔符，因此子系统的字段形如 `VS_RERANKER__BACKEND`。

详见 [`.env.example`](.env.example)。子系统配置块：

- `VS_EMBEDDING_*`：文本嵌入（BGE-M3）
- `VS_IMAGE_EMBEDDING__*`：图像嵌入（OpenCLIP）
- `VS_MULTIMODAL_EMBEDDING__*`：跨模态嵌入（Chinese-CLIP）
- `VS_RERANKER__*`：重排序（BGE-Reranker-v2-M3）
- `VS_MILVUS_*`：向量库连接

---

## API 表面

### Health / Metrics

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 进程存活（不依赖任何后端） |
| `GET` | `/readyz` | 文本/图像嵌入器加载 + Milvus 可达；任一失败则 503 `degraded` / `not_ready` |
| `GET` | `/metrics` | Prometheus 文本格式 |
| `GET` | `/dashboard` | 内嵌调试页面（每类接口一个 tab） |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### Model registry

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列已注册的全部后端（embedder / image_embedder / multimodal_embedder / reranker），用 `type` 区分 |
| `GET` | `/v1/models/{model_id}` | 单个模型元信息 |

### 嵌入

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/embeddings` | OpenAI 兼容文本嵌入 |
| `POST` | `/v1/image_embeddings` | base64 图片嵌入 |
| `POST` | `/v1/multimodal_embeddings` | 中文文本 + 图片跨模态嵌入 |

### 重排序

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/rerank` | cross-encoder 重排，返回按相关性降序的 `{index, score}` 列表 |

### 向量库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/databases` | 列 databases |
| `POST` | `/v1/databases` | 建（body: `{"name": "..."}`） |
| `GET` | `/v1/databases/{name}` | 查元信息 |
| `DELETE` | `/v1/databases/{name}` | 删（含其下所有 collection） |
| `GET` | `/v1/databases/{db}/collections` | 列 collection |
| `POST` | `/v1/databases/{db}/collections` | 建（body 见下文「Collection schema」） |
| `GET` | `/v1/databases/{db}/collections/{coll}` | 查元信息 |
| `DELETE` | `/v1/databases/{db}/collections/{coll}` | 删 |
| `PUT` | `/v1/databases/{db}/collections/{coll}/vectors` | upsert（texts / vectors / images 三选一） |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/delete` | 按 id 删 |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/get` | 按 id 取 |
| `POST` | `/v1/databases/{db}/collections/{coll}/search` | k-NN 检索（query_text / query_vector / query_image 三选一） |

### Backend 诊断（仅运维）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/backend/raw` | 返回当前 backend 名称与 URI |
| `POST` | `/backend/raw/call` | debug-only 透传调底层 store 方法（`VS_DEBUG=true` 时启用，否则 404） |

### 调用示例

```bash
# 1. 嵌入
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'

# 2. 创建 database
curl -X POST http://localhost:8080/v1/databases \
  -H "Content-Type: application/json" \
  -d '{"name": "tenant-a"}'

# 3. 在指定 database 下创建 collection
curl -X POST http://localhost:8080/v1/databases/tenant-a/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "products",
    "primary_field": "id",
    "scalar_fields": [
      {"name": "id", "dtype": "varchar", "is_primary": true, "max_length": 64},
      {"name": "category", "dtype": "varchar", "max_length": 64},
      {"name": "price", "dtype": "float"}
    ],
    "vector_field": {"name": "vector", "dim": 1024, "metric_type": "cosine"},
    "index_params": [
      {"field_name": "vector", "metric_type": "cosine", "index_type": "HNSW",
       "params": {"M": 16, "efConstruction": 200}}
    ]
  }'

# 4. upsert（自动嵌入）
curl -X PUT http://localhost:8080/v1/databases/tenant-a/collections/products/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "ids": ["1"],
    "texts": ["a wireless mouse"],
    "fields": [{"category": "mouse", "price": 29.9}]
  }'

# 5. 检索
curl -X POST http://localhost:8080/v1/databases/tenant-a/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "query_text": "computer accessory",
    "top_k": 5,
    "filter_expr": "category == '\''mouse'\'' and price < 100"
  }'

# 6. 列 databases / 列 collection / drop
curl http://localhost:8080/v1/databases
curl http://localhost:8080/v1/databases/tenant-a/collections
curl -X DELETE http://localhost:8080/v1/databases/tenant-a/collections/products
curl -X DELETE http://localhost:8080/v1/databases/tenant-a
```

---

## 嵌入与重排子系统

### 文本嵌入（BGE-M3）

`POST /v1/embeddings`：OpenAI 兼容协议，接受字符串或字符串列表，自动选择 BGE-M3（GPU 走 torch fp16，CPU 走 ONNX int8）。

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "bge-m3"}'
```

返回示例：

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]}],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

错误码：`model_not_found`（404）/ `too_many_texts` / `text_too_long` / `invalid_request`（422）/ `embedder_unavailable`（503）。

---

### 图像嵌入（OpenCLIP）

`POST /v1/image_embeddings`：当前内置 `openclip-vit-l-14`（OpenCLIP ViT-L/14，openai 预训练权重，768 维）。

```bash
curl -X POST localhost:8080/v1/image_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openclip-vit-l-14",
    "input": {
      "data": "'$(base64 -w0 cat.png)'",
      "mime": "image/png"
    }
  }'
```

返回示例：

```json
{
  "object": "list",
  "data": [
    {"object": "image_embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]}
  ],
  "model": "openclip-vit-l-14",
  "usage": {"prompt_tokens": 1, "total_tokens": 1}
}
```

#### 启动

首次启动会自动从 open_clip 的 CDN（`openaipublic.azureedge.net`）下载 openai 预训练的 `ViT-L-14` 权重到 `./models/openclip-vit-l-14/`（`OPEN_CLIP_DOWNLOAD_PATH` 环境变量也会被指向同一目录）。离线环境把 `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_IMAGE_EMBEDDING__MODEL_DIR` 指定的目录。

所有 `VS_IMAGE_EMBEDDING__*` 配置项见 `.env.example` 的 `Image embedding` 段。

#### 图搜图：upsert + search 接入

`PUT /v1/databases/{db}/collections/{coll}/vectors` 和 `POST .../search` 同时接受文本、向量和图像三种输入（三选一）。图像模式下：

- `PUT`：body 用 `images`（base64 列表）+ `image_mimes`（并行）+ `model`（图像嵌入器 id）代替 `texts`。
- `search`：body 用 `query_image`（base64 单张）+ `query_image_mime` + `model` 代替 `query_text`。

```bash
# 1) upsert 图片（collection dim 必须等于 768）
curl -X PUT localhost:8080/v1/databases/tenant-a/collections/products/vectors \
  -H 'Content-Type: application/json' \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "ids": ["sku-1"],
    "images": ["'"$(base64 -w0 mouse.png)"'"],
    "image_mimes": ["image/png"],
    "model": "openclip-vit-l-14"
  }'

# 2) 图搜图
curl -X POST localhost:8080/v1/databases/tenant-a/collections/products/search \
  -H 'Content-Type: application/json' \
  -d '{
    "primary_field": "id",
    "vector_field": "vector",
    "query_image": "'"$(base64 -w0 query.png)"'",
    "query_image_mime": "image/png",
    "model": "openclip-vit-l-14",
    "top_k": 5
  }'
```

错误码：`model_not_found`（404）/ `image_decode_failed` / `image_too_large` / `unsupported_mime` / `too_many_images`（422）/ `image_embedder_unavailable`（503）。

---

### 图文跨模态嵌入（Chinese-CLIP）

`POST /v1/multimodal_embeddings`：跨模态嵌入子系统同时接受中文文本和 base64 图片，统一在 **512 维共享投影空间**中输出向量。两端的向量可以直接做 cos similarity，是 **文搜图** 和 **图搜文** 检索的基础。当前内置 `chinese-clip-vit-base-patch16`（Chinese-CLIP ViT-B/16，OFA-Sys 预训练权重）。

> 与图像嵌入 (`openclip-vit-l-14`, 768 维) 不同，跨模态模型的两个塔必须使用同一个 `MultimodalEmbedder` 实例、产生同维度向量，才能保证跨模态相似度有意义。

```bash
# 纯文本
curl -X POST localhost:8080/v1/multimodal_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chinese-clip-vit-base-patch16",
    "input": [{"text": "一只猫"}, {"text": "一只狗"}]
  }'

# 混合输入 — 返回顺序与请求顺序一一对应
curl -X POST localhost:8080/v1/multimodal_embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chinese-clip-vit-base-patch16",
    "input": [
      {"text": "一只小猫在窗台上晒太阳"},
      {"image": {"data": "'$(base64 -w0 cat.png)'", "mime": "image/png"}},
      {"text": "一只小狗在草地上奔跑"}
    ]
  }'
```

返回示例：

```json
{
  "object": "list",
  "data": [
    {"object": "multimodal_embedding", "index": 0, "embedding": [0.0123, -0.0456, ...]},
    {"object": "multimodal_embedding", "index": 1, "embedding": [0.0789, -0.1011, ...]},
    {"object": "multimodal_embedding", "index": 2, "embedding": [0.1314, -0.1718, ...]}
  ],
  "model": "chinese-clip-vit-base-patch16",
  "usage": {"prompt_tokens": 3, "total_tokens": 3}
}
```

#### 启动

首次启动会从 HuggingFace（默认 `OFA-Sys/chinese-clip-vit-base-patch16`）下载权重到 `./models/chinese-clip-vit-base-patch16/`。离线环境把 `VS_MULTIMODAL_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_MULTIMODAL_EMBEDDING__MODEL_DIR` 指定的目录。`VS_MULTIMODAL_EMBEDDING__*` 配置项见 `.env.example` 的 `Multimodal embedding` 段。

#### 文搜图 / 图搜文

由于文本塔和图像塔输出同空间向量，把文本向量与图片向量存进同一个 `dim=512` 的 Milvus collection 之后，就可以直接做 cos 相似度检索（Milvus 端无需区分 query 端是文本还是图片 — 上层把文本 query 通过同一 embedder 转成向量即可）。Milvus upsert / search 端点打通跨模态路由在后续版本提供；本轮先把图文嵌入能力上线。

错误码：`model_not_found`（404）/ `image_decode_failed` / `image_too_large` / `unsupported_mime` / `text_too_long` / `too_many_items`（422）/ `multimodal_embedder_unavailable`（503）。

---

### 重排序（cross-encoder）

`POST /v1/rerank`：把向量检索回来的候选 documents 交给 cross-encoder 重排，按相关性得分降序输出 `{index, score}` 列表。当前内置 `bge-reranker-v2-m3`（`BAAI/bge-reranker-v2-m3`）。

```bash
curl -X POST localhost:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "中国首都",
    "documents": ["巴黎是法国首都", "苹果是一种水果", "北京是中华人民共和国的首都"],
    "top_n": 3
  }'
```

返回示例：

```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 2, "score": 0.9876},
    {"index": 0, "score": 0.0432}
  ],
  "request_id": "8f4e1c2a-9b1d-4f0e-9c1a-2b3c4d5e6f70"
}
```

#### 启动

reranker 默认随主进程一起启动。在 `.env` 里设好 `VS_RERANKER__BACKEND`，然后像往常一样跑：

```bash
VS_RERANKER__BACKEND=bge-reranker-v2-m3 vector-service
```

首次启动会自动从 ModelScope 下载 `BAAI/bge-reranker-v2-m3` 权重到 `./models/bge-reranker-v2-m3/`；离线环境可以把 `VS_RERANKER__AUTO_DOWNLOAD=false` 然后手工把权重放到 `VS_RERANKER__MODEL_DIR` 指定的目录。

错误码：`model_not_found`（404）/ `invalid_request` / `too_many_documents` / `document_too_long` / `query_too_long` / `invalid_top_n`（422）/ `reranker_not_loaded` / `reranker_error`（503）。

---

## 向量库管理（Milvus CRUD）

### Collection schema

与早期版本不同，本服务的 collection **schema 完全由调用方定义**：调用方提供主键字段、其他标量字段、向量字段和索引参数；服务不再注入任何字段。

约束：

- 恰好一个 `VARCHAR` 主键字段（通过 `primary_field` 指定名字，必须在 `scalar_fields` 中出现并 `is_primary=true`）
- 至少一个 `FLOAT_VECTOR` 字段（通过 `vector_field` 指定）
- 至少一个索引覆盖 `vector_field`（通过 `index_params` 提供）

支持的标量类型：`bool` / `int8` / `int16` / `int32` / `int64` / `float` / `double` / `varchar` / `json`。

支持的距离度量：`cosine` / `ip` / `l2`。

完整请求示例：

```json
{
  "name": "products",
  "primary_field": "id",
  "scalar_fields": [
    {"name": "id", "dtype": "varchar", "is_primary": true, "max_length": 64},
    {"name": "category", "dtype": "varchar", "max_length": 64},
    {"name": "price", "dtype": "float"}
  ],
  "vector_field": {"name": "vector", "dim": 1024, "metric_type": "cosine"},
  "index_params": [
    {"field_name": "vector", "metric_type": "cosine", "index_type": "HNSW",
     "params": {"M": 16, "efConstruction": 200}}
  ]
}
```

### Upsert / Search

- `PUT .../vectors`：`ids`、`texts` 或 `vectors` 或 `images` 三选一；可选 `fields`（与 `ids` 对齐的标量字段值字典列表）。
- `POST .../search`：`query_text` / `query_vector` / `query_image` 三选一；可选 `filter_expr`（Milvus 原生布尔表达式，例如 `category == 'mouse' and price < 100`）、`top_k`（默认 10）、`output_fields`（要回传的标量字段名列表）。

`filter_expr` 会被原样转发给 pymilvus，字段名必须匹配创建 collection 时声明的标量字段名。

### 运维诊断：`/backend/raw`

```bash
curl localhost:8080/backend/raw
# {"backend": "milvus", "info": {"backend": "milvus", "uri": "http://localhost:19530", ...}}
```

`/backend/raw/call` 在 `VS_DEBUG=true` 时提供对底层 store 方法的有限透传，仅供联调；线上请勿启用。

---

## 模型热加载 / 热卸载

服务对每个模型族（text embedder / image embedder / multimodal embedder / reranker）只保留一个常驻实例挂在 `app.state.<family>` 上。无需重启进程就能换模型或腾显存。

### 启动策略：默认不加载

生产默认行为：**进程启动时所有模型族都不构造、不加载**。`app.state.<family>` 全为 `None`，推理路由（`/v1/embeddings`、`/v1/image_embeddings`、`/v1/multimodal_embeddings`、`/v1/rerank`）立即返回 503。运维在启动后通过 dashboard 面板或 API 显式加载需要的模型。

`/readyz` 的 gate 已收敛到**仅 store 可达性**——模型未加载时 `/readyz` 仍然返回 200，body 中按家族报告 `not_loaded`，方便 K8s probe 不会因为运维还没点 Load 而把流量切走。仅当 vector store 不可达时 `/readyz` 才返 503 `degraded`。

如需保留旧的 eager-load 行为，在 `.env` 设：

```bash
VS_EMBEDDING_AUTO_LOAD=true             # 文本嵌入
VS_IMAGE_EMBEDDING__AUTO_LOAD=true      # 图像嵌入
VS_MULTIMODAL_EMBEDDING__AUTO_LOAD=true  # 图文嵌入
VS_RERANKER__AUTO_LOAD=true             # 重排
```

每个开关独立，不开 eager 的族保持默认的「未加载」状态。

### 加载 / 卸载端点

- `POST /v1/models/{model_id}/load` — 构造并加载该 id 对应的后端实例，替换同族当前实例。返回 `{id, type, status: "loaded", dimensions}`。同 id 重发是幂等操作（不会重建实例）。
- `POST /v1/models/{model_id}/unload` — 释放同族实例并把 `app.state.<family>` 置 `None`，后续该族推理返回 503。返回 `{id, type, status: "unloaded"}`。

```bash
# 加载 BGE-M3
curl -X POST localhost:8080/v1/models/bge-m3/load
# {"id":"bge-m3","type":"embedder","status":"loaded","dimensions":1024}

# 推理现在可用
curl -X POST localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "hello", "model": "bge-m3"}'

# 卸载后内存即释放，再次推理会得到 503 embedder_unavailable
curl -X POST localhost:8080/v1/models/bge-m3/unload
# {"id":"bge-m3","type":"embedder","status":"unloaded"}

# 重新加载
curl -X POST localhost:8080/v1/models/bge-m3/load
```

### Dashboard 面板

打开 <http://localhost:8080/dashboard>，切到「模型 → 加载/卸载」子标签：

- 顶部「刷新状态」按钮调 `GET /v1/models` 拉已注册 id 并按家族分组。
- 每行展示：模型 id + 类型徽标 + 维度、`已加载`/`未加载` pill、Load / Unload 按钮。
- 「自动刷新」开关（默认开）每 5s 轮询一次，仅在当前 tab 可见时拉接口。
- Load/Unload 后自动同步刷新「嵌入」「图像嵌入」「图文嵌入」「重排」各页签下的模型下拉。

### 错误码

- 同族并发 load/unload → 409 `model_busy`（不排队，路由立即返回）。
- 同族已加载了不同 id → 409 `conflict_loaded`（先 unload 再换）。
- load 路由的 factory 或实例 `load()` 抛异常 → 503 `model_load_failed`，slot 保持原状。
- unload 一个空 slot → 409 `not_loaded`。
- 未知 model_id → 404 `model_not_found`。
- 未加载即推理 → 503 `embedder_unavailable` / `image_embedder_unavailable` / `multimodal_embedder_unavailable` / `reranker_not_loaded`。

### 实现细节

每个模型族对应 `core/model_lifecycle.py` 里的 `ModelSlot`，挂到 `app.state._slot_<family>`；lifespan 在启动后按 `auto_load` 决定是否构造 + 加载实例并通过 `set_instance()` 注入对应 slot，热加载/卸载路由与推理路由都基于该 slot 同步状态。卸载在子类里释放原生资源（`_impl`/`_model`/`_preprocess`/`tokenizer` 等）并 best-effort 调用 `torch.cuda.empty_cache()`，base class 提供 no-op 默认实现。

---

## 统一错误信封

所有非 2xx 响应统一信封：

```json
{
  "error": {
    "code": "database_not_found",
    "message": "database 'tenant-a' does not exist",
    "request_id": "8f4e1c2a-9b1d-4f0e-9c1a-2b3c4d5e6f70",
    "extra": { "name": "tenant-a" }
  }
}
```

`extra` 中的 `exception_type` / `exception_cause` 字段在底层异常被包装时由 `main.py` 自动注入，方便排障。

错误码：

| code | 含义 | HTTP |
|------|------|------|
| `database_not_found` | database 不存在 | 404 |
| `database_exists` | 同名 database 已存在 | 409 |
| `collection_not_found` | collection 不存在 | 404 |
| `collection_exists` | 同名 collection 已存在 | 409 |
| `dimension_mismatch` | 向量维度与 collection 不一致 | 422 |
| `model_not_found` | 嵌入器 / reranker 未注册 | 404 |
| `embedder_unavailable` | 文本嵌入器未就绪 / 推理失败 | 503 |
| `image_embedder_unavailable` | 图像嵌入器未就绪 / 推理失败 | 503 |
| `multimodal_embedder_unavailable` | 跨模态嵌入器未就绪 / 推理失败 | 503 |
| `reranker_not_loaded` | reranker 未加载 | 503 |
| `reranker_error` | reranker 推理失败 | 503 |
| `store_unavailable` | Milvus 不可达 / 返回 5xx | 503 |
| `shape_mismatch` | ids/vectors 长度不一致 | 422 |
| `too_many_texts` / `text_too_long` | 文本输入超限 | 422 |
| `too_many_documents` / `document_too_long` | rerank 输入超限 | 422 |
| `query_too_long` | rerank query 过长 | 422 |
| `invalid_top_n` | rerank top_n 超过上限 | 422 |
| `too_many_images` / `image_too_large` / `image_decode_failed` / `unsupported_mime` | 图像输入超限 | 422 |
| `too_many_items` | 跨模态输入 item 数超限 | 422 |
| `invalid_request` | pydantic 校验失败 | 422 |
| `internal` | 未捕获异常 | 500 |

---

## 测试

```bash
# 单元测试（不需要 Milvus / 模型权重）
uv run pytest -m "not contract and not integration" -q

# 跑全部测试（contract 测试需要真实 Milvus ≥ 2.4）
uv run pytest -q
```

测试组织：

- `tests/unit/`：纯单元测试，使用 fake / mock embedder，秒级可跑完。
- `tests/contract/`：contract 测试，需要真实 Milvus server，通过 `@pytest.mark.contract` 标记。

合约测试示例：连接 `VS_MILVUS_URI` 上的真实 Milvus，验证 database/collection/upsert/search 端到端可用。

---

## 添加新后端

### 新文本嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `Embedder` ABC（`embed_documents` + `dim` + `model_name`）。
2. 在 `embeddings/registry.py` 的 `EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `Settings.embedding_*`。

### 新图像嵌入器

1. 新建 `embeddings/<backend>.py`，实现 `ImageEmbedder` ABC（`embed_images` + `dim` + `model_name`）。
2. 在 `embeddings/image_registry.py` 的 `IMAGE_EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `ImageEmbeddingSettings`。

### 新跨模态嵌入器

1. 新建 `embeddings/<backend>.py`，实现 `MultimodalEmbedder` ABC（`embed_text` + `embed_images` + `dim` + `model_name`）。
2. 在 `embeddings/multimodal_registry.py` 的 `MULTIMODAL_EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `MultimodalEmbeddingSettings`。

### 新 reranker 后端

1. 新建 `rerankers/<backend>.py`，实现 `Reranker` ABC。
2. 在 `rerankers/cross_encoder.py` 同目录下追加 `RERANKER_REGISTRY["<name>"] = <Class>`。
3. 如需新配置项，扩展 `core/config.py` 的 `RerankerSettings`。

### 新向量库后端

1. 新建 `stores/<backend>.py`，实现 `VectorStore` ABC。
2. 在 `stores/registry.py` 的 `build_store()` 函数加 `elif` 分支。
3. 如需新配置项，扩展 `core/config.py` 的 `Settings`（参考 `milvus_*`）。
