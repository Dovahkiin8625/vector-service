# 评测集（Evaluation Sets）

离线评测回答"参数/模型调整后检索是变好了还是变差了"。评测数据是
**corpus 元数据**（schema v9，SQLite 事实来源），与向量索引无关，
不进 Milvus：

- **评测集（set）** 绑定一个 `(database, collection)`；
- **问题（question）** 携带期望切片 / 期望文档 / 期望答案（至少一项）；
- **版本（version）** 把某一时刻的问题与期望整体冻结为不可变快照；
- **运行（run）** 用同一份检索模板批量跑活问题或指定版本的冻结问题，
  落每问的有序切片 id、排名指标（Recall / MRR / nDCG）、rerank 前后
  对比与通道归因，并可选生成答案；
- **门禁（gate）** 把 scope 绑定到版本 + 阈值；**门禁检查（check）**
  对 parked canary 跑冻结问题并判定通过与否，canary 必须通过门禁
  才能 promote。

## 数据模型

| 表 | 说明 |
|----|------|
| `eval_sets` | `evs_` + uuid；scope + name/description |
| `eval_questions` | `evq_` + uuid；question、`expected_chunk_ids`、`expected_doc_ids`（JSON 数组）、`expected_answer`（可空） |
| `eval_set_versions` | `evv_` + uuid；`tag`（set 内唯一）、`question_count`、`snapshot_json` 冻结整组问题 |
| `eval_runs` | `evr_` + uuid；`params_json` 保存模板、`include_answer` 与可选 `version_id` |
| `eval_results` | 每问一行：有序 `chunk_ids`（JSON）、`metrics_json`、`answer` / `answer_error` |
| `regression_gates` | `gat_` + uuid；scope（`database`/`collection` 唯一）+ `set_id`/`version_id`、模板、阈值、可选 `baseline_run_id` |
| `gate_checks` | `evc_` + uuid；`run_id`、`candidate_ref`、`status`、`report_json` |

删除评测集级联删除问题、版本、运行、结果、门禁与检查。

## 使用流程

```bash
# 1. 建评测集
POST /v1/evaluation/sets
{"database": "default", "collection": "ingest", "name": "退款知识 v1"}

# 2. 批量加问题（至少带一种期望）
POST /v1/evaluation/sets/{set_id}/questions
{
  "questions": [
    {
      "question": "退款多久到账？",
      "expected_chunk_ids": ["doc_a_2"],
      "expected_doc_ids": ["doc_a"]
    },
    {
      "question": "退款流程是什么？",
      "expected_answer": "用户申请→客服审核→原路退回"
    }
  ]
}

# 3. 提交批量运行（202，后台 eval_run 任务）
POST /v1/evaluation/sets/{set_id}/runs
{
  "template": {
    "top_k": 10,
    "channels": {"dense": true, "bm25": true, "summary": false},
    "rerank": {"enabled": true, "candidate_pool": 25}
  },
  "include_answer": false
}
# → {"run_id": "evr_...", "job_id": "..."}

# 4. 查结果（任务 done 后）
GET /v1/evaluation/runs/{run_id}
```

## 数据集版本化

活问题会随维护增删改，回归对比需要一个稳定基线。版本把创建时刻
集合内全部问题（question、期望切片/文档/答案）原样冻结进
`snapshot_json`，之后不可修改；同一 set 的 tag 唯一。

```bash
# 冻结当前问题为一个版本（空集合 → 422 eval_set_empty）
POST /v1/evaluation/sets/{set_id}/versions
{"tag": "baseline-2026q3"}
# → {"version_id": "evv_...", "tag": "baseline-2026q3", "question_count": 12, ...}

# 列出全部版本（newest-first）
GET /v1/evaluation/sets/{set_id}/versions
```

- tag 重复 → **409 `version_tag_exists`**；
- 提交 run 时带 `version_id` 即对冻结问题运行，而不是活问题；版本
  不存在或不属于该 set → **404 `eval_version_not_found`**；
- 版本化 run 的问题措辞以快照为准：GET run 时即使活问题已删除/改写，
  结果中的 question 仍从快照恢复。

## 回归门禁

门禁把一个 scope 与"冻结版本 + 通过标准"绑定，是 canary → active
切换的强制关卡。

**阈值两类，至少配置一项：**

| 阈值 | 含义 |
|------|------|
| `min_recall` / `min_mrr` / `min_ndcg` | 绝对下限：check 的指标低于下限即违规 |
| `max_recall_drop` / `max_mrr_drop` / `max_ndcg_drop` | 相对跌幅：相对 `baseline_run_id` 的下降幅度不得超过该值；配置跌幅必须同时给 `baseline_run_id` |

```bash
# 1. 配置门禁（一个 scope 同时只有一个 gate）
POST /v1/evaluation/gates
{
  "database": "default",
  "collection": "ingest",
  "set_id": "evs_...",
  "version_id": "evv_...",
  "template": {"top_k": 10, "rerank": {"enabled": true, "candidate_pool": 25}},
  "min_recall": 0.8,
  "max_recall_drop": 0.05,
  "baseline_run_id": "evr_..."
}

# 2. （先提交 reindex，等新物理索引 parked 为 canary）

# 3. 提交门禁检查（202，后台 gate_check 任务）
POST /v1/evaluation/gates/{gate_id}/checks
# → {"check_id": "evc_...", "run_id": "evr_...", "job_id": "..."}

# 4. 查检查结果
GET /v1/evaluation/gates/{gate_id}/checks
```

门禁检查的语义：

- 模板的 `index_ref` 被**强制改写为 canary 物理索引**，从物理上保证
  评测对象就是待提升索引；无 parked canary → **409 `gate_no_canary`**；
- 对冻结版本逐问跑完整管线（不生成答案），汇总指标后评估阈值；
- 每个条件独立判定，违规码：`{metric}_below_min`、
  `{metric}_drop_exceeds`、`{metric}_unmeasured`（指标缺失）；
- `report_json` 记录完整报告：

```json
{
  "candidate_ref": "ingest__rebuild_ab12cd",
  "metrics": {"recall": 0.82, "mrr": 0.71, "ndcg": 0.76},
  "baseline": {"recall": 0.85, "mrr": 0.74, "ndcg": 0.78},
  "deltas": {"recall": -0.03},
  "violations": [],
  "passed": true
}
```

**Promote 强制：** `POST .../reindex/promote` 在切换前检查该 scope
的门禁——最近一次 check 必须是 `passed`，且其 `candidate_ref` 就是
当前 parked canary（为旧 canary 拿的通过不授权新索引），否则
**409 `gate_blocked`**。scope 未配置 gate 时不受限。

门禁管理：`GET /v1/evaluation/gates`（可按 database/collection 过滤）、
`GET` / `DELETE /v1/evaluation/gates/{gate_id}`（删除级联 checks）。

## 检索模板

模板（`EvalTemplate`）覆盖 `RetrievalRequest` 除 `database` /
`collection` / `query` 外的全部参数：`index_ref`、`top_k`、`channels`、
`fusion`、`rewrite`、`mmr`、`rerank`、`graph`、`routing`、
`context_budget`、`context_level`。scope 取评测集自身的 scope，query
逐问替换——同一份模板对每个问题各跑一次完整管线。

`index_ref` 可在蓝绿/灰度期间把整批评测精确钉到 active 或 canary
物理索引，实现"同一批问题、新旧索引对比"。

## 指标

每问按期望计算，落在该问的 `metrics` 中：

| 指标 | 含义 |
|------|------|
| `recall` | 命中期望切片数 / 期望切片总数（即 Recall@k，k = 最终返回长度） |
| `mrr` | 第一个期望切片的排名倒数，无命中为 0 |
| `ndcg` | nDCG：二值相关性，折损 `1/log2(rank+1)`，理想排序归一化 |
| `doc_hit` | 返回切片是否有任一篇属于期望文档 |

### rerank 前后对比

run 开启 rerank 时，管线记录送入 reranker 的候选 id 序（post-MMR、
pre-rerank），每问额外落 `metrics.rerank`：

```json
{"pre_recall": 0.5, "pre_mrr": 0.5, "pre_ndcg": 0.631}
```

前序指标在候选序上截断到 `top_k` 计算，与最终列表的 `recall` /
`mrr` / `ndcg` 直接可比。聚合摘要的 `summary.rerank` 给出两侧均值：

| 字段 | 含义 |
|------|------|
| `questions` | 实际经过 rerank 的问题数 |
| `mean_recall_pre` / `mean_recall_post` | rerank 前 / 最终 Recall 均值 |
| `mean_mrr_pre` / `mean_mrr_post` | rerank 前 / 最终 MRR 均值 |
| `mean_ndcg_pre` / `mean_ndcg_post` | rerank 前 / 最终 nDCG 均值 |

### 按通道归因

`metrics.channels` 回答"哪条通道贡献了命中"：

```json
{
  "hits":  {"dense": 2, "bm25": 1},
  "found": {"dense": 2, "bm25": 1, "summary": 0},
  "hits_total": 2,
  "wanted_total": 3
}
```

- `hits`：最终列表中命中的期望切片，按其 `matched_channels` 归因
  （一个切片可同时归给多条通道）；
- `found`：该通道在其原始 channel run 中出现过的期望切片数——即使
  融合 / rerank 后来丢掉了它；
- `hits_total` / `wanted_total`：本问命中数 / 期望总数，供汇总做
  微平均。

聚合摘要的 `summary.channel_attribution.channels` 按通道给出：

| 字段 | 含义 |
|------|------|
| `hit_share` | 该通道贡献的最终命中数 / 全部最终命中数 |
| `raw_recall` | 该通道在原始 run 中找到的期望切片 / 全部期望切片 |

### 聚合摘要

`summary` 汇总：`mean_recall` / `mean_mrr` / `mean_ndcg` /
`doc_hit_rate`（只统计带相应期望的问题），以及上面的 `rerank`
（无 rerank 问题时为 `null`）与 `channel_attribution`
（无切片期望时为 `null`）。

> `context_level` 为 `section` / `document` 时，返回 id 是上扩后的
> 父级 id，与叶子级期望切片不可比；切片级指标请保持 `chunk`，
> 文档级命中仍可经 `doc_hit` 比较。

## 答案生成

`include_answer=true` 时，每问检索完成后用 LLM 基于**返回切片正文**
生成答案（要求仅依据片段作答，不足时明确说明，不编造）。

- 提交时 LLM 未配置 → **503 `llm_unavailable`**；
- 单问答案生成失败不中断整批：该问 `answer=null`、`answer_error`
  记录原因，其余问题照常完成。

## 重跑语义

运行结果整体可从 corpus 派生：重试时 `eval_results` 对该 run
**整体替换**（不追加），进程重启中断的 `eval_run` / `gate_check`
任务在启动恢复时直接重排重跑（不清理 corpus；gate check 复用其 run，
取消标记 → job 与 check 均 `cancelled`，attempts 用尽
→ job `failed/interrupted`、check `failed`）。
