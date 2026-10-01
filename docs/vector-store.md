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

dense 与 sparse 搜索固定以 **Strong 一致性**发出：Milvus 默认 Bounded 带约数秒陈旧窗口，摄取 / 修复刚提交的行立刻检索可能返回零结果（重建行数校验通过、三路召回却全空）。Strong 让服务端按最新时间戳执行，代价是搜索不能复用更旧的可读快照。

## 多向量集合（multi-vector）

Milvus 的 collection 可声明**多个 `FLOAT_VECTOR` 字段**。store 接口
（Python 层）通过两个可选参数支持：

- `create_collection(..., extra_vector_fields=[FieldSpec(name, dtype="float_vector", dim=...), ...])`
  —— 在主向量之外追加向量字段，各自独立维度；
- `upsert(..., extra_vectors={"field_name": [[...], ...]})` —— 按字段名
  传入与 `ids` 逐行对齐的额外向量。

校验规则：

- 额外字段必须是 `float_vector` 且 dim 为正；字段名不得与主键 / 标量 /
  主向量重名；
- 每个额外向量字段都要有索引，只接受 dense 度量（cosine / ip / l2）；
- Milvus 向量字段不可空、无默认值——**每行都必须携带全部向量**，缺向量
  的行无法写入（调用方需自行兜底，例如用原文嵌入）；
- `search(field=...)` 可指定任一已声明向量字段，按该字段自己的 dim 校验；
- browse 默认投影排除**所有**向量字段，`include_vectors` 时保留 dense
  向量。

## Sparse 向量与 BM25（客户端编码）

Milvus 只持有 sparse **向量本身**（`SPARSE_FLOAT_VECTOR`，
`SPARSE_INVERTED_INDEX` + `IP`），**不做服务端分词、不注册 BM25 Function、
不存切片正文**：

- sparse 字段的索引度量只接受 `ip`（不再有 `"bm25"` 度量）；
- 服务侧 `SparseBM25`（`corpus/bm25.py`）用 jieba 分析器在客户端编码：
  ingest 后对整个语料 fit 统计，文档编码为 `{term_id: weight}` 字典写入；
- BM25 统计按 `(database, collection)` 持久化在 `data/corpus/bm25/`，
  与 Milvus 行一样是**派生物**——文件丢失时从 SQLite 全量重建，空语料
  查询报 RuntimeError；
- 查询时 `encode_query` 生成 sparse 向量，`store.search_sparse(...)` 检索；
  查询编码结果为空（词项全不在词表）则跳过该路 RPC；
- sparse 向量无法取回（Milvus `not allowed to retrieve raw data of field
  sparse`），browse/投影中不出现，这是向量字段本身的限制，与编码方式无关。

> 通用 collection 管理 HTTP 端点目前只暴露一个 `vector_field`；多向量的
> 实际消费者是摄取管线的固定 schema——`summary_vector` 摘要向量
> （见 [ingest-pipeline.md § Collection schema](ingest-pipeline.md#collection-schema写入约定)）。

## 运维诊断：`/backend/raw`

```bash
curl localhost:8080/backend/raw
# {"backend": "milvus", "info": {"backend": "milvus", "uri": "http://localhost:19530", ...}}
```

`/backend/raw/call` 在 `VS_DEBUG=true` 时提供对底层 store 方法的有限透传，仅供联调；线上请勿启用。
