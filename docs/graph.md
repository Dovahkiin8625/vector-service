# GraphRAG：知识图谱与社区检索

知识图谱是继向量索引之后第二层**派生物**：从 SQLite 叶子切片经 LLM
抽取出实体 / 关系 / 声明，按规范名合并，做加权标签传播社区检测，
再用 LLM 生成社区摘要。图谱行写入 corpus，实体向量与社区向量进入
**两个独立的薄 Milvus collection**。图谱可随时从 corpus 整体重建，
与向量索引一样不做备份、不做向后兼容。

## 数据模型（corpus schema v6）

| 表 | 内容 |
|----|------|
| `entities` | `entity_id`、逻辑 collection 内唯一 `name`、`entity_type`、合并后的 `description` |
| `entity_mentions` | 实体 ↔ 叶子切片的出处（多对多） |
| `edges` | 有向边 `source_id → target_id`、`description`、`weight`（该方向在切片中被观察到的次数） |
| `edge_mentions` | 边 ↔ 叶子切片的出处 |
| `claims` | 关于某主体实体的事实陈述，可带客体实体、`claim_type`、`status`（true / false / suspect） |
| `communities` | `community_index`、LLM `summary` |
| `community_members` | 社区 ↔ 实体成员 |

身份与 id 全部确定，重建可对比、可整图覆盖：

- 实体身份 = 逻辑 collection 内的 **canonical name**（任意空白折叠为单空格）；
  没有额外的 LLM 实体消歧轮；
- 实体 id：`ge_` + `sha1("{database}/{collection}/{name}")[:16]`；
- 边 id：`ed_` + `sha1("{source_id}>{target_id}")[:16]`；
- 声明 id：`cl_` + `sha1("{chunk_id}/{subject_id}/{statement}")[:16]`；
- 社区 id：`cm_` + `sha1("{database}/{collection}/{成员有序拼接}")[:16]`。

## 构建管线

```
叶子切片（SQLite, level='chunk'）
  → extract    LLM 逐片抽取 entities / relationships / claims（有界并发）
  → merge      按 canonical name 合并：描述去重累加、出处合并、边权累加
  → communities 加权异步标签传播（边按无向处理）
  → summarize  LLM 为保留社区写 3–6 句摘要（有界并发）
  → index      实体向量 + 社区向量写入两个独立薄 collection
  → replace    单事务整图替换 7 张表
```

- 任务复用 ingest 任务框架：`job_type="graph_build"`，状态阶段新增
  `extracting` / `graphing`；`POST .../graph/build` 返回 `202` + `job_id`，
  进度、SSE、取消、重试语义与摄取/重建任务一致。
- **LLM 故障全链路降级**：单片抽取失败 → 该片视为空结果，不中断构建；
  社区摘要失败 → 摘要留空，嵌入时以成员名拼接兜底。
- 成功后校验两个向量 collection 的行数；不一致返回 500
  `graph_count_mismatch`（worker 可重试）。
- 删除联动：删除文档 / 切片时按出处级联清理——失去全部出处的边、
  实体被删除（外键级联声明与社区成员关系），失去最后一个成员的
  社区随之删除；`delete_for_collection` / `delete_for_database` 清空
  对应 scope 的全部图谱行。

### 实体 / 社区 collection 命名

逻辑名先做 Milvus 合法字符清洗（非字母数字下划线替换为 `_`），再：

- 实体：`{safe}__graph_entities`
- 社区：`{safe}__graph_communities`

两者均为薄 schema：主键 `id`（varchar 64）+ 单个 `vector`（cosine /
HNSW），不存任何正文。图谱 collection 不参与向量索引的 blue-green
binding，其命名只由逻辑 collection 决定。

### 构建请求参数

| 字段 | 默认 | 说明 |
|------|------|------|
| `entity_types` | `[]` | 追加到抽取 prompt 的类型提示 |
| `community_iterations` | 20 | 标签传播轮数上限（1–100） |
| `min_community_size` | 2 | 小于该规模的社区不摘要、不入索引（成员实体仍在实体索引中） |
| `include_claims` | true | 是否持久化抽取出的声明 |
| `batch_size` | 16 | 每次抽取批的叶子数（进度 tick / 取消门粒度） |

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/databases/{db}/collections/{coll}/graph/build` | 提交图谱构建任务，`202` 返回 `job_id` / `entity_collection` / `community_collection`；非 corpus collection → 422 `invalid_request` |
| `GET` | `/v1/databases/{db}/collections/{coll}/graph` | 图谱状态：四类计数 + 社区列表（含规模与摘要） |
| `DELETE` | `/v1/databases/{db}/collections/{coll}/graph` | 删除 corpus 图谱行并 drop 两个向量 collection（缺失视为成功） |

## 检索侧用法

`POST /v1/retrieval` 请求体新增 `graph` 段（默认关闭）：

```json
{
  "query": "Alice 和 Acme 是什么关系？",
  "graph": {
    "enabled": true,
    "entity_top_k": 10,
    "community_top_k": 3,
    "neighbor_depth": 1,
    "include_claims": true,
    "claims_limit": 20
  }
}
```

检索流程在 recall 与 fuse 之间新增 graph 阶段，详见
[retrieval.md § GraphRAG 图谱检索](retrieval.md#graphrag-图谱检索)：

1. 嵌入 query，对实体 collection 做 ANN 得到种子实体；
2. 按 `neighbor_depth` 经 `edges` 做 BFS 扩展（上限 500 节点，避免
   枢纽实体拖入整图），种子实体带 ANN 分数，扩展节点分数为 null；
3. `community_top_k > 0` 时对社区 collection 做 ANN，返回带成员列表
   的社区摘要；
4. `include_claims` 时返回与图中实体相关的声明。

`graph` 与 chunks 并列返回，不参与切片的融合 / MMR / 重排排序。
预检时若图谱行或向量 collection 缺失，返回 422 `graph_not_built`。
