# 统一错误信封

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

## 错误码

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
| `model_busy` | 同族并发 load/unload（含后台加载进行中再发 load/unload） | 409 |
| `conflict_loaded` | 同族已加载了不同 id | 409 |
| `model_load_failed` | 后台加载时 factory 或实例 `load()` 抛异常；**不再同步返回**，改为在 `GET /v1/models` 该 id 的 `load_status="failed"` / `load_error` 中观测（POST 返回 202） | 202 → 轮询 |
| `not_loaded` | 对一个空 slot 做 unload | 409 |
| `unsupported_mime` | `/v1/parse` 或 `/v1/jobs/ingest` 上传了不支持的 MIME（支持 PDF / Office / HTML / jpg·png·tiff·webp·bmp / md / txt） | 415 |
| `file_too_large` | 上传文件超过 `VS_PARSER__MAX_FILE_SIZE_MB` | 413 |
| `invalid_profile` | `profile` 字段不在 `auto` / `standard` / `native` / `vlm` 中；预检阶段返回，错误体含 `got` / `allowed` | 400 |
| `parser_failed` | Docling 解析失败 | 500 |
| `parser_unavailable` | Docling converter 构建失败（依赖缺失等） | 503 |
| `chunk_failed` | 分片器内部错误 | 500 |
| `retrieval_empty_query` | 检索 query 为空 | 422 |
| `retrieval_channel_unsupported` | 请求的召回路在集合上不可用：bm25 需要 `sparse` 字段、摘要路需要 `summary_vector` 字段（错误体带 `channels`） | 422 |
| `retrieval_unknown_index_ref` | 请求 pin 的 `index_ref` 既非该逻辑 collection 的 active 也非 canary | 422 |
| `reindex_no_canary` | promote 时 binding 上没有停放 canary | 409 |
| `rebuild_empty` | 逻辑 collection 下没有叶子切片，无可重建内容 | 400 |
| `rebuild_count_mismatch` | 重建后物理行数与 SQLite 叶子数不一致（5xx，worker 可重试；错误体带 `expected` / `got`） | 500 |
| `sparse_encode_failed` | 重建时拟合全新 BM25 统计失败 | 503 |
| `graph_not_built` | 检索请求开启 `graph` 但该 collection 无图谱行或实体/社区向量 collection 缺失；先 POST `.../graph/build` | 422 |
| `graph_empty` | 逻辑 collection 下没有叶子切片，图谱构建无从抽取 | 400 |
| `graph_count_mismatch` | 图谱构建写入的实体/社区向量行数与预期不一致（5xx，worker 可重试） | 500 |
| `llm_unavailable` | 查询改写 / 图谱构建需要的 LLM 未配置（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`） | 503 |
| `internal` | 未捕获异常 | 500 |
