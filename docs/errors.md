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
| `model_busy` | 同族并发 load/unload | 409 |
| `conflict_loaded` | 同族已加载了不同 id | 409 |
| `model_load_failed` | load 路由的 factory 或实例 `load()` 抛异常 | 503 |
| `not_loaded` | 对一个空 slot 做 unload | 409 |
| `internal` | 未捕获异常 | 500 |
