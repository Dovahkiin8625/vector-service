# 架构定位

## 拓扑

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

## 源码结构

```
vector-service/
├── pyproject.toml              # 项目元数据 + 依赖 + pytest 配置
├── README.md                   # 项目简介与入口
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
│   │   ├── middleware.py       # RequestID 中间件
│   │   └── model_lifecycle.py  # ModelSlot：每个模型族的常驻实例 + load/unload
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
├── docs/                       # 用户文档（本目录）
│   ├── README.md               # 文档目录索引
│   ├── quickstart.md
│   ├── architecture.md         # 本文件
│   ├── configuration.md
│   ├── api.md
│   ├── embedding-subsystems.md
│   ├── vector-store.md
│   ├── model-lifecycle.md
│   ├── errors.md
│   ├── testing.md
│   ├── extending.md
│   ├── dashboard-overview.png
│   └── dashboard-text-similarity.png
└── models/                     # 模型权重目录（运行时下载；已在 .gitignore）
```

## 扩展点

- **新增嵌入器**（文本 / 图像 / 跨模态）：在 `embeddings/` 下新建适配文件，并在对应的 `*_registry.py` 注册。详见 [extending.md](extending.md)。
- **新增 reranker**：在 `rerankers/` 下实现 `Reranker` ABC 并在 `RERANKER_REGISTRY` 注册。
- **新增向量库后端**：在 `stores/` 下实现 `VectorStore` 接口，并在 `stores/registry.py::build_store()` 加分支。
