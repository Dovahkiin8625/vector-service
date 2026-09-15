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
