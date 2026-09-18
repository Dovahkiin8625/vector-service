# 添加新后端

## 新文本嵌入器

参见 `src/vector_service/embeddings/`：

1. 新建 `embeddings/<backend>.py`，实现 `Embedder` ABC（`embed_documents` + `dim` + `model_name`）。
2. 在 `embeddings/registry.py` 的 `EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `Settings.embedding_*`。

## 新图像嵌入器

1. 新建 `embeddings/<backend>.py`，实现 `ImageEmbedder` ABC（`embed_images` + `dim` + `model_name`）。
2. 在 `embeddings/image_registry.py` 的 `IMAGE_EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `ImageEmbeddingSettings`。

## 新跨模态嵌入器

1. 新建 `embeddings/<backend>.py`，实现 `MultimodalEmbedder` ABC（`embed_text` + `embed_images` + `dim` + `model_name`）。
2. 在 `embeddings/multimodal_registry.py` 的 `MULTIMODAL_EMBEDDER_REGISTRY` 注册。
3. 如需新配置项，扩展 `core/config.py` 的 `MultimodalEmbeddingSettings`。

## 新 reranker 后端

1. 新建 `rerankers/<backend>.py`，实现 `Reranker` ABC。
2. 在 `rerankers/cross_encoder.py` 同目录下追加 `RERANKER_REGISTRY["<name>"] = <Class>`。
3. 如需新配置项，扩展 `core/config.py` 的 `RerankerSettings`。

## 新向量库后端

1. 新建 `stores/<backend>.py`，实现 `VectorStore` ABC。
2. 在 `stores/registry.py` 的 `build_store()` 函数加 `elif` 分支。
3. 如需新配置项，扩展 `core/config.py` 的 `Settings`（参考 `milvus_*`）。

## 新文档解析器

`parsers/` 提供 PDF / DOCX / PPTX / HTML / Markdown / Text 的解析，统一转为 Markdown 输出。

1. 新建 `parsers/<name>_parser.py`，实现 `DocumentParser` ABC（`parse_bytes(data, mime) -> ParsedDocument`）。
2. 在 `parsers/base.py` 注册新解析器（按 MIME 关联到对应实现）。
3. 如需新配置项，扩展 `core/config.py` 的 `ParserSettings`。
4. 在 `api/parse.py` 与 `api/ingest.py` 的 MIME 白名单中加入新类型。

> Docling 是默认的二进制格式解析器（PDF/DOCX/PPTX/HTML），新增专门解析器通常只在以下场景：
> 自定义二进制格式、专门的 OCR 后端、或对 Markdown 有特殊语义提取。

## 新分片器

`chunking/recursive_chunker.py` 是默认的递归 markdown 分片器。新增专门分片器：

1. 新建 `chunking/<name>_chunker.py`，实现一个 `chunk(markdown, chunk_size, chunk_overlap, metadata) -> list[ChunkItem]` 函数。
2. 在 `api/chunk.py` 与 `api/ingest.py` 通过请求参数（如 `chunker: "recursive" | "my_chunker"`）路由。
3. Pydantic schemas（`schemas/ingest.py::ChunkItem`）已固定字段，新分片器只需填充这些字段即可。
