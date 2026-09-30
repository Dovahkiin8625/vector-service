# TODO — vector-service 演进路线

本文件记录 **SQLite 事实来源 + Milvus 薄派生索引** 架构落地后的后续事项。

当前架构基线：

- **SQLite（`data/corpus/corpus.db`）是事实来源**：documents / chunks / ingest_jobs / chunk_indexes。
- **Milvus 是可重建的派生索引**：只存向量与最少标量（`id`、`doc_id`、`chunk_index`），不存切片正文、不做服务端分词。
- BM25 统计（`data/corpus/bm25/*.bm25.json`）同样是派生物，丢失即从 SQLite 重建。
- 检索流水线：rewrite → recall（dense / bm25 / summary 多通道）→ fuse → hydrate（SQLite 组装内容）→ mmr → rerank。
- 演进原则：**不做向后兼容**，破坏性改动直接落地，不写版本探测 / 回退 / 迁移分支。

---

## 1. 异步任务 API 与后台 worker

现状：同步 `/v1/ingest`（含 stream）已删除；任务提交/查询接口已落地（`POST /v1/jobs/ingest`、`GET /v1/jobs`、`GET /v1/jobs/{id}`），上传字节落 spool、任务行落 queued；进程内后台 worker 已接入并消费 queued 任务，支持阶段进度（parsing/chunking/embedding/upserting）、cancel（阶段边界协作取消）与重试（4xx 立即失败，5xx 指数退避重排，`not_before_ts` 落库）；`POST /v1/jobs/{id}/cancel` 可用。启动恢复（遗留 running 任务）尚未做。dashboard ingest 视图依赖旧路由，已知失效，待第 7 项重写。

- [x] 引入任务提交/查询接口：`POST /v1/jobs/ingest` 立即返回 `job_id`，`GET /v1/jobs/{id}` 查状态与进度。
- [x] 进程内后台 worker（单进程 `workers=1`，先不上独立进程/队列）：从 `ingest_jobs` 取 queued 任务执行，支持 cancel、重试次数、阶段进度（parsing/chunking/embedding/upserting）。
- [x] 启动时恢复：进程重启后把遗留的 running 任务置回 queued（spool 缺失→failed interrupted、遗留 cancel→cancelled、清理半成品后 attempts+1 重排）。
- [x] SSE 推送任务进度：`GET /v1/jobs/{id}/events`（进程内 JobEventBus nudge + 行签名去重 + 心跳 + 2s 兜底 resync，终态后关闭）。

## 2. 层级切片与 small-to-large / parent-document 检索

chunks 表已预留脚手架列：`parent_id`、`level`、`char_start`、`char_end`、`context`。

- [x] chunking 输出层级结构（如 small chunk → section → document），写入 parent_id / level / 字符区间。
- [x] 检索召回小切片后按 `parent_id` 向上扩展（expand）：返回父切片/整节/整篇，参数控制扩展层级与目标粒度。
- [x] 父子索引分离：小切片进 dense ANN，父级只存内容（或仅 summary_vector），避免父级文本污染召回。
- [x] 切片重叠 dedup：相邻小切片扩展到同一父级时合并输出。

## 3. 多索引与索引重建任务

`chunk_indexes` 已记录每条 chunk 的（index_kind、model、index_ref）；约束为每 (chunk, kind) 一条当前索引。

- [x] 重建任务：换 embedding / BM25 模型 = 新建 collection（或重建索引）→ 从 SQLite 重算全部向量 → 切换 registry 指向 → 删除旧索引。
- [x] 重建期间双索引并存（新旧 collection），检索侧灰度选择，完成后下线旧 collection。
- [x] BM25 统计失效自动化：删除/更新文档后按需 refit；当前只在 ingest 时重算，删除后统计会过期（已先靠 discard / 查询期重建兜底）。
- [x] 索引一致性校验任务：对比 SQLite chunk 数与 Milvus 行数，输出缺失/孤儿报告并支持一键修复。

## 4. 高级检索能力

- [x] 查询改写（rewrite）多路径：多查询生成、HyDE、step-back；目前通道框架已在，策略待丰富。
- [x] 融合策略扩展：RRF 之外增加加权分数归一化融合；通道权重可配置。
- [x] 上下文压缩：rerank 后按预算（token 数）裁剪，而非只按 top_k。
- [x] 引用回链：返回结果带 doc_id / chunk_index / page_number / char_start-char_end，支持前端高亮定位。
- [x] GraphRAG：新增 entities / edges / claims 表，抽取管线复用 chunking+ingest 任务框架；实体向量独立 collection，支持社区检测与社区摘要检索。
- [x] 多路混合索引：关键词、向量、图谱、结构化元数据按查询意图路由。

## 5. 反馈与评测

- [x] 检索结果反馈（点赞/点踩/点击/采纳）落库，关联 query、chunk_id、管道版本（模型/参数）。
- [x] 评测集管理：question / expected chunks / expected docs / 答案，批量跑检索与生成。
- [x] 指标：Recall@k、MRR、nDCG、rerank 前后对比、按通道归因（哪条通道贡献了命中）。
- [x] 数据集版本化，回归测试门禁化（模型/参数变更先过评测集再切换 registry）。

## 6. 存储与运行时演进

- [x] 线程池隔离：当前阻塞调用走默认 executor，需为 store RPC、模型推理、SQLite 配置独立线程池与限流，避免相互饿死。
- [x] SQLite 写竞争与体积：定期 VACUUM；附件/原文二进制不入库（文件存盘 + hash 引用）。
- [x] 多实例部署时迁移到 Postgres（WAL/连接池/行锁成熟），repository 接口保持不变，只替换实现；在此之前保持单实例约束。
- [x] 备份策略：corpus.db 定期备份（LiteStream 或定时拷贝 WAL 快照），Milvus 数据不备份（可重建）。

## 7. 可观测性与运维

- [x] 指标补齐：每通道召回量/命中率、融合后存活率、rerank 耗时与输入长度分布。
- [x] 分布式追踪（OpenTelemetry）：一次检索贯通 rewrite→各通道→hydrate→rerank 的 span。
- [x] dashboard：索引重建、任务队列、一致性校验、评测结果的可视化操作面。
