# vector-service 文档

生产级 FastAPI 向量服务的详细文档。仓库根目录的 [README.md](../README.md) 是项目简介与快速入口。

## 目录

| 文档 | 内容 |
|------|------|
| [quickstart.md](quickstart.md) | 5 分钟跑起来：前置条件、配 `.env`、起 Milvus、装依赖、起服务 |
| [architecture.md](architecture.md) | 架构定位、四块职责、源码目录树 |
| [configuration.md](configuration.md) | `VS_*` 环境变量清单与子系统配置块（文本嵌入 / 图像嵌入 / 跨模态嵌入 / reranker / Milvus） |
| [api.md](api.md) | 全部 HTTP 端点表（Health / Model registry / 嵌入 / 重排 / 向量库管理 / Backend 诊断）+ 调用示例 |
| [embedding-subsystems.md](embedding-subsystems.md) | 4 个推理子系统的详细文档：BGE-M3 文本嵌入、OpenCLIP 图像嵌入、Chinese-CLIP 跨模态嵌入、BGE-Reranker 重排 |
| [vector-store.md](vector-store.md) | Milvus 数据库 CRUD：collection schema、upsert/search 规则、`/backend/raw` 诊断 |
| [model-lifecycle.md](model-lifecycle.md) | 模型热加载 / 热卸载：默认不加载策略、`load` / `unload` 端点、dashboard 面板、错误码、实现细节 |
| [errors.md](errors.md) | 统一错误信封结构 + 错误码清单 |
| [testing.md](testing.md) | 单元测试与 contract 测试、运行命令、目录组织 |
| [extending.md](extending.md) | 如何新增文本嵌入器 / 图像嵌入器 / 跨模态嵌入器 / reranker / 向量库后端 |

## 截图

| 文件 | 内容 |
|------|------|
| [dashboard-overview.png](dashboard-overview.png) | Dashboard 默认落地页（KPI 总览 + 模型注册表 + 向量库状态 + 机器指标 + GPU 卡） |
| [dashboard-text-similarity.png](dashboard-text-similarity.png) | 文本相似度调试面板（侧栏 模型 → 文本相似度） |
