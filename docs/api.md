# API 表面

所有端点的请求/响应模型用 Pydantic 校验，定义在 `src/vector_service/schemas/`。非 2xx 响应统一信封见 [errors.md](errors.md)。

> 各推理端点的详细参数、配置项和错误码见 [embedding-subsystems.md](embedding-subsystems.md)；向量库端点的 schema/upsert/search 规则见 [vector-store.md](vector-store.md)；模型热加载/卸载端点见 [model-lifecycle.md](model-lifecycle.md)。

## Health / Metrics

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/healthz` | 进程存活（不依赖任何后端） |
| `GET` | `/readyz` | 文本/图像嵌入器加载 + Milvus 可达；任一失败则 503 `degraded` / `not_ready` |
| `GET` | `/metrics` | Prometheus 文本格式 |
| `GET` | `/dashboard` | 内嵌调试页面（每类接口一个 tab） |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

## Model registry

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列已注册的全部后端（embedder / image_embedder / multimodal_embedder / reranker），用 `type` 区分 |
| `GET` | `/v1/models/{model_id}` | 单个模型元信息 |
| `POST` | `/v1/models/{model_id}/load` | 显式加载模型（详见 [model-lifecycle.md](model-lifecycle.md)） |
| `POST` | `/v1/models/{model_id}/unload` | 显式卸载模型（同上） |

## 嵌入

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/embeddings` | OpenAI 兼容文本嵌入（BGE-M3） |
| `POST` | `/v1/image_embeddings` | base64 图片嵌入（OpenCLIP ViT-L/14） |
| `POST` | `/v1/multimodal_embeddings` | 中文文本 + 图片跨模态嵌入（Chinese-CLIP） |

参数、请求体格式、错误码见 [embedding-subsystems.md](embedding-subsystems.md)。

## 重排序

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/rerank` | cross-encoder 重排，返回按相关性降序的 `{index, score}` 列表 |

参数与错误码见 [embedding-subsystems.md § 重排序](embedding-subsystems.md#重排序cross-encoder)。

## 向量库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/databases` | 列 databases |
| `POST` | `/v1/databases` | 建（body: `{"name": "..."}`） |
| `GET` | `/v1/databases/{name}` | 查元信息 |
| `DELETE` | `/v1/databases/{name}` | 删（含其下所有 collection） |
| `GET` | `/v1/databases/{db}/collections` | 列 collection |
| `POST` | `/v1/databases/{db}/collections` | 建（body 见 [vector-store.md § Collection schema](vector-store.md#collection-schema)） |
| `GET` | `/v1/databases/{db}/collections/{coll}` | 查元信息 |
| `DELETE` | `/v1/databases/{db}/collections/{coll}` | 删 |
| `PUT` | `/v1/databases/{db}/collections/{coll}/vectors` | upsert（texts / vectors / images 三选一） |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/delete` | 按 id 删 |
| `POST` | `/v1/databases/{db}/collections/{coll}/vectors/get` | 按 id 取 |
| `POST` | `/v1/databases/{db}/collections/{coll}/search` | k-NN 检索（query_text / query_vector / query_image 三选一） |

详细 schema 约束、字段类型、距离度量、upsert/search 规则见 [vector-store.md](vector-store.md)。

## Backend 诊断（仅运维）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/backend/raw` | 返回当前 backend 名称与 URI |
| `POST` | `/backend/raw/call` | debug-only 透传调底层 store 方法（`VS_DEBUG=true` 时启用，否则 404） |

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
