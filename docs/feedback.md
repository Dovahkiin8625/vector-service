# 检索反馈（Feedback）

反馈数据是评测与调参的原料。用户对检索结果的点赞 / 点踩 / 点击 /
采纳落库，关联 query、chunk 与**管道版本快照**，使后续可以回答
"哪个模型 / 参数组合下哪条路贡献了好结果"。

反馈表是 corpus 的一部分（schema v7，SQLite 事实来源），与向量索引
无关，不进 Milvus。

## 数据模型

`feedback` 表：

| 列 | 说明 |
|----|------|
| `feedback_id` | `fb_` + uuid，主键 |
| `database` / `collection` | 反馈针对的 scope |
| `query` | 原始检索 query（原样，不剥离谓词） |
| `chunk_id` | 可空：空 = 答案级（👍/👎）；非空 = 切片级（点击/采纳） |
| `kind` | `up` / `down` / `click` / `adopt` |
| `pipeline_json` | 管道版本快照：channels、fusion、rewrite、rerank、models、index_ref、routing |
| `comment` | 可选备注 |
| `created_ts` | 时间戳 |

管道快照由客户端在反馈时回传——它记录的是用户做出反应时那次检索
的上下文，服务端不做隐式关联（反馈与检索是独立请求）。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/feedback` | 提交一条反馈；返回 `feedback_id` |
| `GET` | `/v1/feedback` | 列出反馈（新→旧），支持 `database` / `collection` / `kind` 过滤与 `limit`（≤200）/ `offset` |

```json
POST /v1/feedback
{
  "database": "default",
  "collection": "ingest",
  "query": "author:张三 page>=10 退款流程怎么处理？",
  "chunk_id": null,
  "kind": "up",
  "comment": "解决了问题",
  "pipeline": {
    "channels": {"dense": true, "bm25": false, "summary": false},
    "fusion": {"method": "rrf"},
    "index_ref": "ingest",
    "routing": {"enabled": true, "mode": "auto"},
    "models": {"embedder": "bge-m3"}
  }
}
```

切片级反馈（用户点开 / 采纳某条切片）带 `chunk_id`，`kind` 通常为
`click` / `adopt`。

后续的评测集、Recall@k / MRR / nDCG 指标与通道归因以此表为输入
（见 TODO 第 5 节）。
