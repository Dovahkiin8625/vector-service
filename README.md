# vector-service

生产级 FastAPI 向量服务：

- **OpenAI 兼容嵌入 API**：`/v1/embeddings`、`/v1/models`
- **图像嵌入 API**：`/v1/image_embeddings`（OpenCLIP ViT-L/14，把 base64 图片转成 768 维向量）
- **多 database 管理**：`/v1/databases` 增删改查 database，每个 database 下挂若干 collection
- **database-scoped 向量库管理**：`/v1/databases/{db}/collections/{coll}/vectors|search|...`
- **直连 Milvus server**：通过 `pymilvus` 直接连接独立部署的 Milvus，无中间代理
- **可观测性**：`/metrics`（Prometheus）+ 结构化 JSON 日志
- **可扩展抽象**：增加新嵌入器只需新建适配文件并注册；增加新向量库后端在 `stores/` 下实现 `VectorStore` 接口并在 `registry.py` 分支

## 第一阶段支持

| 组件 | 实现 |
|------|------|
| Embedder | BGE-M3 (GPU: torch fp16 / CPU: ONNX int8) |
| ImageEmbedder | OpenCLIP ViT-L/14 (openai/ViT-L-14, 768 维) |
| VectorStore | Milvus server（直连 pymilvus） |

---

## 架构定位

```
client ──HTTP──▶ vector-service ──gRPC──▶ Milvus server
                  ├─ BGE-M3 嵌入
                  ├─ OpenCLIP 图像嵌入
                  └─ Milvus store 客户端（pymilvus 直连）
```

`vector-service` 自己的职责有三块：

1. **文本嵌入**：用 BGE-M3 把文本变成 `list[float]`。
2. **图像嵌入**：用 OpenCLIP 把 base64 图片变成 `list[float]`。
3. **向量库 CRUD**：通过 `pymilvus` 直接对 Milvus 做 database / collection / 向量管理。

---

## 启动服务

### 0. 前置条件

- Python ≥ 3.11
- 一台可访问的 Milvus server（standalone / cluster），版本 ≥ 2.4（推荐启用 native database）
- 可选：GPU + CUDA（如需用 torch fp16 后端跑 BGE-M3；否则自动落到 ONNX int8 CPU）

### 1. 准备 .env

```bash
cp .env.example .env
```

至少确认：

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `VS_HOST` / `VS_PORT` | 监听地址 | `0.0.0.0` / `8080` |
| `VS_LOG_LEVEL` / `VS_LOG_FORMAT` | 日志级别 / 格式（`json` / `console`） | `INFO` / `json` |
| `VS_EMBEDDING_BACKEND` | 嵌入器后端 | `bge-m3` |
| `VS_EMBEDDING_MODEL_DIR` | 模型本地目录 | `./models/bge-m3` |
| `VS_EMBEDDING_AUTO_DOWNLOAD` | 首次启动自动下载模型权重 | `true` |
| `VS_VECTOR_STORE_BACKEND` | 向量库后端（目前只支持 `milvus`） | `milvus` |
| `VS_MILVUS_URI` | Milvus server 地址（gRPC） | `http://localhost:19530` |
| `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD` | 账号密码（启用 auth 时填） | （空） |
| `VS_MILVUS_TOKEN` | 可选 token（优先级高于 user/password） | （空） |
| `VS_MILVUS_TIMEOUT` | 连接 / RPC 超时（秒） | `30` |

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

`[all]` 同时装 BGE-M3 和 pymilvus。也可以分开装：`uv pip install -e ".[embed,store]"`。

### 4. 启动服务

```bash
uvicorn vector_service.main:app --host 0.0.0.0 --port 8080
```

访问：

| 路径 | 内容 |
|------|------|
| `http://127.0.0.1:8080/playground` | 自带调试页面（每个 tab 都能点按钮调接口） |
| `http://127.0.0.1:8080/docs` | Swagger UI |
| `http://127.0.0.1:8080/healthz` | 进程存活 |
| `http://127.0.0.1:8080/readyz` | 嵌入器加载 + Milvus 可达 |
| `http://127.0.0.1:8080/metrics` | Prometheus |

---

## 调用示例

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
  -d '{"name": "products"}'

# 4. upsert（自动嵌入）
curl -X PUT http://localhost:8080/v1/databases/tenant-a/collections/products/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids": ["1"], "texts": ["a wireless mouse"], "metadatas": [{"price": 29.9}]}'

# 5. 检索
curl -X POST http://localhost:8080/v1/databases/tenant-a/collections/products/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "computer accessory", "top_k": 5}'

# 6. 列 databases / 列 collection / drop
curl http://localhost:8080/v1/databases
curl http://localhost:8080/v1/databases/tenant-a/collections
curl -X DELETE http://localhost:8080/v1/databases/tenant-a/collections/products
curl -X DELETE http://localhost:8080/v1/databases/tenant-a
```

---

## API 表面

### Database

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/databases` | 列 databases |
| `POST` | `/v1/databases` | 建（body: `{"name": "..."}`） |
| `GET` | `/v1/databases/{name}` | 查元信息 |
| `DELETE` | `/v1/databases/{name}` | 删（含其下所有 collection） |

### Collection（database-scoped）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/databases/{db}/collections` | 列 collection |
| `POST` | `/v1/databases/{db}/collections` | 建（body: `{"name", "dim"?, "metric", "backend_opts"}`） |
| `GET` | `/v1/databases/{db}/collections/{coll}` | 查元信息 |
| `DELETE` | `/v1/databases/{db}/collections/{coll}` | 删 |

### 向量（database-scoped）

| 方法 | 路径 | 说明 |
|------|------|------|
| `PUT` | `/v1/databases/{db}/collections/{coll}/vectors` | upsert（texts 或 vectors 二选一） |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/delete` | 按 id 删 |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/get` | 按 id 取 |
| `POST` | `/v1/databases/{db}/collections/{coll}/search` | k-NN 检索（query_text 或 query_vector 二选一） |

### 嵌入（OpenAI 协议）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/embeddings` | OpenAI 兼容嵌入 |
| `GET` | `/v1/models` | 列已注册的嵌入器和 reranker（用 `type` 区分） |
| `GET` | `/v1/models/{model_id}` | 单个模型元信息（嵌入器或 reranker） |

### 可观测性

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 进程存活 |
| `GET` | `/readyz` | 嵌入器加载 + Milvus 可达（否则 503 + `"degraded"`） |
| `GET` | `/metrics` | Prometheus 文本 |

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

错误码：

| code | 含义 | HTTP |
|------|------|------|
| `database_not_found` | database 不存在 | 404 |
| `database_exists` | 同名 database 已存在 | 409 |
| `collection_not_found` | collection 不存在 | 404 |
| `collection_exists` | 同名 collection 已存在 | 409 |
| `dimension_mismatch` | dim 与 collection 不一致 | 422 |
| `model_not_found` | 嵌入器未注册 | 404 |
| `embedder_unavailable` | 嵌入器未就绪 / 推理失败 | 503 |
| `store_unavailable` | Milvus 不可达 / 返回 5xx | 503 |
| `shape_mismatch` | ids/vectors 长度不一致 | 422 |
| `too_many_texts` / `text_too_long` | 输入超限 | 422 |
| `invalid_request` | pydantic 校验失败 | 422 |
| `internal` | 未捕获异常 | 500 |

---

## Milvus 字段约定

每个 collection 在创建时建三个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `VARCHAR(64)` primary key | 调用方在 upsert 时提供 |
| `vector` | `FLOAT_VECTOR(dim)` | 由 embedder 或调用方提供 |
| `metadata_json` | `VARCHAR(65535)` | 任意 JSON 字符串，`/search` 的 `metadata` 原样回传 |

`create_collection` 还会在 `vector` 上建 HNSW 索引（`M=16`, `efConstruction=200`，metric 跟随 `metric` 入参 `cosine/ip/l2`）。

`filter` 参数走 `vector_service.core.filter_translator.translate_filter`，把 JSON 字典翻成 Milvus 表达式。例如 `{"price": {"$lt": 100}}` → `` metadata["price"] < 100 ``。

---

## 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件，详见 `.env.example`。

## 架构

```
api/         FastAPI 路由（OpenAI 协议 + database / collection / vector 管理）
embeddings/  Embedder 抽象 + 实现（BGE-M3）
stores/      VectorStore 抽象 + Milvus 直连实现（pymilvus）
core/        配置 / 日志 / 指标 / 错误 / 生命周期 / 中间件
schemas/     Pydantic 请求/响应模型
```

## 添加新嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `Embedder` ABC
2. 在 `embeddings/registry.py` 注册
3. 加配置项到 `core/config.py`

## 添加新向量库后端

参见 `src/vector_service/stores/`：

1. 新建 `stores/<backend>.py`，实现 `VectorStore` ABC
2. 在 `stores/registry.py` 加分支
3. 加配置项到 `core/config.py`

---

## 重排序（Rerank）

Reranker 子系统把向量检索回来的候选 documents 交给 cross-encoder 重排，按相关性得分降序输出 `{index, score}` 列表。当前内置 `bge-reranker-v2-m3`（`BAAI/bge-reranker-v2-m3`）。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/rerank` | 重排序：接收 `query` + `documents`，返回按相关性降序的 `{index, score}` 列表 |

> 已注册的 reranker 后端通过 `GET /v1/models` 暴露，按 `type=reranker` 过滤即可。

### 启动

reranker 默认随主进程一起启动。在 `.env` 里设好 `VS_RERANKER__BACKEND`，然后像往常一样跑：

```bash
VS_RERANKER__BACKEND=bge-reranker-v2-m3 uv run vector-service
```

首次启动会自动从 ModelScope 下载 `BAAI/bge-reranker-v2-m3` 权重到 `./models/bge-reranker-v2-m3/`；离线环境可以把 `VS_RERANKER__AUTO_DOWNLOAD=false` 然后手工把权重放到 `VS_RERANKER__MODEL_DIR` 指定的目录。

所有 `VS_RERANKER__*` 配置项见 `.env.example` 的 `Reranker` 段（`backend` / `model_dir` / `device` / `batch_size` / `max_documents_per_request` / `max_chars_per_doc` / `max_query_chars` / `max_top_n` / `top_n_default` / `download_source`）。

### 示例

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

### 错误码

所有非 2xx 响应共用前文 "API 表面" 一节展示的统一 error 信封。rerank 端点的 9 个错误码：

| HTTP | code | 触发 |
|------|------|------|
| 404 | `model_not_found` | 后端名未在 `RERANKER_REGISTRY` 注册 |
| 422 | `invalid_request` | pydantic 校验失败（如 `query`/`documents` 为空、`top_n < 1`） |
| 422 | `too_many_documents` | `documents` 长度超过 `VS_RERANKER__MAX_DOCUMENTS_PER_REQUEST` |
| 422 | `document_too_long` | 任一 document 超过 `VS_RERANKER__MAX_CHARS_PER_DOC` |
| 422 | `query_too_long` | `query` 长度超过 `VS_RERANKER__MAX_QUERY_CHARS` |
| 422 | `invalid_top_n` | `top_n` 超过 `VS_RERANKER__MAX_TOP_N` |
| 503 | `reranker_not_loaded` | 启动期权重加载失败（模型目录缺失且未启用 auto_download） |
| 503 | `reranker_error` | 重排序推理失败（sentence-transformers 抛错等） |
| 500 | `internal` | 未捕获的兜底异常 |

---

## 图像嵌入（Image Embeddings）

图像嵌入子系统把 base64 编码的图片转成 `list[float]`。当前内置 `openclip-vit-l-14`（OpenCLIP ViT-L/14，openai 预训练权重，768 维）。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/image_embeddings` | 图像嵌入：接收 base64 图片，返回 `{index, embedding}` 列表 |

> 已注册的图像嵌入器通过 `GET /v1/models` 暴露，按 `type=image_embedder` 过滤即可。

### 启动

首次启动会自动从 HuggingFace 下载 `openai/ViT-L-14` 权重到 `./models/openclip-vit-l-14/`；离线环境把 `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD=false`，手工把权重放到 `VS_IMAGE_EMBEDDING__MODEL_DIR` 指定的目录。

所有 `VS_IMAGE_EMBEDDING__*` 配置项见 `.env.example` 的 `Image embedding` 段（`backend` / `model_dir` / `auto_download` / `device` / `batch_size` / `max_images_per_request` / `max_image_bytes` / `hf_repo` / `allowed_mime`）。

### 示例

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

### 图搜图：upsert + search 接入

`PUT /v1/databases/{db}/collections/{coll}/vectors` 和 `POST .../search` 同时接受文本、向量和图像三种输入（三选一）。图像模式下：

- `PUT`：body 用 `images`（base64 列表）+ `image_mimes`（并行）+ `model`（图像嵌入器 id）代替 `texts`。
- `search`：body 用 `query_image`（base64 单张）+ `query_image_mime` + `model` 代替 `query_text`。

```bash
# 1) 建 collection（dim 必须等于 768）
curl -X POST localhost:8080/v1/databases/tenant-a/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "products",
    "primary_field": "id",
    "scalar_fields": [{"name": "id", "dtype": "varchar", "is_primary": true, "max_length": 64}],
    "vector_field": {"name": "vector", "dim": 768, "metric_type": "cosine"}
  }'

# 2) upsert 图片
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

# 3) 图搜图
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

### 错误码

所有非 2xx 响应共用前文 "API 表面" 一节展示的统一 error 信封。图像嵌入端点的错误码：

| HTTP | code | 触发 |
|------|------|------|
| 404 | `model_not_found` | 图像嵌入器 id 未在 `IMAGE_EMBEDDER_REGISTRY` 注册 |
| 422 | `image_decode_failed` | base64 解码失败 |
| 422 | `image_too_large` | 解码后字节数超过 `VS_IMAGE_EMBEDDING__MAX_IMAGE_BYTES` |
| 422 | `too_many_images` | 输入图片数超过 `VS_IMAGE_EMBEDDING__MAX_IMAGES_PER_REQUEST` |
| 422 | `unsupported_mime` | MIME 不在 `VS_IMAGE_EMBEDDING__ALLOWED_MIME` |
| 503 | `image_embedder_unavailable` | 启动期权重加载失败 / 推理失败 |
| 500 | `internal` | 未捕获的兜底异常 |

## 添加新图像嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `ImageEmbedder` ABC
2. 在 `embeddings/image_registry.py` 注册
3. 如需新配置项，扩展 `core/config.py` 的 `ImageEmbeddingSettings`

---

## 添加新 reranker 后端

参见 `src/vector_service/rerankers/`：

1. 新建 `rerankers/<backend>.py`，实现 `Reranker` ABC
2. 在文件末尾 `RERANKER_REGISTRY["<name>"] = <Class>`
3. 如需新配置项，扩展 `core/config.py` 的 `RerankerSettings`