# 向量库管理（Milvus CRUD）

服务通过 `pymilvus` 直连 Milvus server（standalone 或 cluster），由 `VS_MILVUS_URI` / `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD` / `VS_MILVUS_TOKEN` 控制连接。完整端点列表见 [api.md § 向量库管理](api.md#向量库管理)。

## Collection schema

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

## Upsert / Search

- `PUT .../vectors`：`ids`、`texts` 或 `vectors` 或 `images` 三选一；可选 `fields`（与 `ids` 对齐的标量字段值字典列表）。
- `POST .../search`：`query_text` / `query_vector` / `query_image` 三选一；可选 `filter_expr`（Milvus 原生布尔表达式，例如 `category == 'mouse' and price < 100`）、`top_k`（默认 10）、`output_fields`（要回传的标量字段名列表）。

`filter_expr` 会被原样转发给 pymilvus，字段名必须匹配创建 collection 时声明的标量字段名。

## 运维诊断：`/backend/raw`

```bash
curl localhost:8080/backend/raw
# {"backend": "milvus", "info": {"backend": "milvus", "uri": "http://localhost:19530", ...}}
```

`/backend/raw/call` 在 `VS_DEBUG=true` 时提供对底层 store 方法的有限透传，仅供联调；线上请勿启用。
