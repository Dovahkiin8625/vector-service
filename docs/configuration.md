# 配置

所有配置通过 `VS_*` 环境变量或 `.env` 文件。pydantic-settings 使用 `__`（双下划线）作为嵌套分隔符，因此子系统的字段形如 `VS_RERANKER__BACKEND`。

字段全集见 [`.env.example`](../.env.example)。

## 子系统配置块

| 前缀 | 适用子系统 |
|------|------------|
| `VS_EMBEDDING_*` | 文本嵌入（BGE-M3） |
| `VS_IMAGE_EMBEDDING__*` | 图像嵌入（OpenCLIP） |
| `VS_MULTIMODAL_EMBEDDING__*` | 跨模态嵌入（Chinese-CLIP） |
| `VS_RERANKER__*` | 重排序（BGE-Reranker-v2-M3） |
| `VS_MILVUS_*` | 向量库连接 |
| `VS_VECTOR_STORE_BACKEND` | 向量库后端选择（目前只支持 `milvus`） |

## 关键开关速查

| 变量 | 默认 | 含义 |
|------|------|------|
| `VS_HOST` / `VS_PORT` | `0.0.0.0` / `8080` | 监听地址 |
| `VS_LOG_LEVEL` / `VS_LOG_FORMAT` | `INFO` / `json` | 日志级别 / 格式（`json` 或 `console`） |
| `VS_DEBUG` | `false` | 是否开放 `/backend/raw/call` debug 透传 |
| `VS_EMBEDDING_AUTO_LOAD` | `false` | 进程启动时是否自动加载文本嵌入器 |
| `VS_IMAGE_EMBEDDING__AUTO_LOAD` | `false` | 进程启动时是否自动加载图像嵌入器 |
| `VS_MULTIMODAL_EMBEDDING__AUTO_LOAD` | `false` | 进程启动时是否自动加载跨模态嵌入器 |
| `VS_RERANKER__AUTO_LOAD` | `false` | 进程启动时是否自动加载 reranker |
| `VS_EMBEDDING_AUTO_DOWNLOAD` | `true` | 首次启动自动下载模型权重 |
| `VS_IMAGE_EMBEDDING__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 OpenCLIP 权重 |
| `VS_MULTIMODAL_EMBEDDING__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 Chinese-CLIP 权重 |
| `VS_RERANKER__AUTO_DOWNLOAD` | `true` | 首次启动自动下载 BGE-Reranker 权重 |

> **生产默认行为**：所有模型族 `AUTO_LOAD=false`，启动时不构造、不加载。详见 [model-lifecycle.md](model-lifecycle.md)。

## Milvus 连接

`VS_MILVUS_TOKEN` 优先级高于 `VS_MILVUS_USER` / `VS_MILVUS_PASSWORD`。`VS_MILVUS_TIMEOUT` 同时作用于连接与 RPC。完整的 Milvus 配置语义见 [vector-store.md](vector-store.md)。
