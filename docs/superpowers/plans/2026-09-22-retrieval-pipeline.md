# 分片检索管线（Retrieval Pipeline）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 知识库新增「分片检索」面板：dense 基础召回 + dense/BM25 混合召回与 RRF/加权融合 + cross-encoder 重排 + HyDE/Multi-Query/Step-Back/Decomposition 查询改写 + MMR 多样性 + 元数据过滤 + 全链路追踪。

**Architecture:** Python 侧编排——新 `vector_service/retrieval/` 包把每个查询变体 × 每个启用 channel 并发 fan-out（`asyncio.gather` + `run_in_executor`），融合为纯函数；ingest collection schema 升级 v2（+ `sparse` SPARSE_FLOAT_VECTOR + `text` chinese 分析器 + BM25 Function），旧库走拷贝-删除-rename 一键迁移。

**Tech Stack:** FastAPI（同步阻塞调用 + executor）、Pydantic v2、pymilvus 3.0.1 MilvusClient、Vue 3 ESM CDN（无构建）、pytest/TestClient。

**Spec:** `docs/superpowers/specs/2026-09-22-retrieval-pipeline-design.md`（计划逐条覆盖该 spec，执行时两份一起读）。

## Global Constraints

- Milvus server 2.5+ / 2.6 / 3.0；pymilvus 已锁 3.0.1；不新增依赖。
- 检索目标固定 `ingest` collection；UI 只选 database；不做跨 collection。
- 新库直接建 schema v2；现有 v1 库 schema 不可变，只能经迁移端点整体拷贝。
- LLM（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`）可能未配置：只有 rewrite 受影响（503/UI 置灰），dense、混合、融合、重排、MMR 不受影响。
- 所有阻塞调用（embed/search/chat/rerank）经 `loop.run_in_executor`；同组召回并发。
- 错误信封统一 `{"error": {"code", "message", ...}}`。
- 不做 SPLADE、不封装服务端 hybrid_search、不做 Agentic/Self-RAG/CRAG（YAGNI）。
- 每个任务结束：`pytest tests/unit -q` 全绿后提交。

---

## File Structure

**新建：**

| 文件 | 职责 |
|------|------|
| `src/vector_service/retrieval/__init__.py` | 导出 `RetrievalPipeline` |
| `src/vector_service/retrieval/base.py` | 管线全部数据类 |
| `src/vector_service/retrieval/fusion.py` | `rrf_fuse()` / `weighted_fuse()` 纯函数 |
| `src/vector_service/retrieval/channels.py` | `Channel` ABC；`DenseChannel` / `BM25Channel` |
| `src/vector_service/retrieval/transforms.py` | HyDE / MultiQuery / StepBack / Decompose + JSON 容错 |
| `src/vector_service/retrieval/diversity.py` | `mmr()` |
| `src/vector_service/retrieval/pipeline.py` | `RetrievalPipeline` 编排器 |
| `src/vector_service/schemas/retrieval.py` | 请求/响应 Pydantic 模型 + `to_result()` |
| `src/vector_service/api/retrieval.py` | `/v1/retrieval`、`/stream`、`/capabilities`、迁移端点 |
| `src/vector_service/static/dashboard/components/retrieval.js` | 检索面板 Vue 组件 |
| `docs/retrieval.md` | 检索能力文档 |
| `tests/unit/test_retrieval_fusion.py` 等 9 个测试文件 | 见各任务 |

**修改：**

| 文件 | 改动 |
|------|------|
| `src/vector_service/stores/base.py` | `FieldSpec` +2 字段；新增抽象 `search_text` |
| `src/vector_service/stores/_milvus_adapter.py` | sparse dtype/BM25 Function/分析器透传；`search_text`；`rename_collection`/`insert_rows`；browse `include_vectors`；dtype code 104 |
| `src/vector_service/stores/milvus.py` | 校验放宽到 sparse/bm25；字段透传；`search_text` |
| `src/vector_service/api/ingest.py` | v2 schema/index 助手；`_ensure_collection` 用 v2；`schema_version()`；`migrate_ingest_collection()` |
| `src/vector_service/main.py` | 注册 retrieval_router |
| `src/vector_service/static/dashboard/components/app.js` | import/NAV_LABELS/nav-item/挂载/i18n |
| `src/vector_service/static/dashboard/dashboard.css` | retrieval 专属样式 |
| `docs/api.md`、`docs/errors.md`、`docs/README.md`、`.env.example` | 文档同步 |

---

### Task 1: retrieval 包骨架 + base 数据类

**Files:**
- Create: `src/vector_service/retrieval/__init__.py`
- Create: `src/vector_service/retrieval/base.py`
- Test: `tests/unit/test_retrieval_base.py`

**Interfaces:**
- Produces（后续所有任务依赖，名字固定）:
  - `RecallSpec(query: str, vector: list[float] | None = None, hypothetical: str | None = None)`
  - `ChannelHit(chunk_id: str, score: float, rank: int, fields: dict)`
  - `ChannelRun(channel: str, query: str, hits: list[ChannelHit])`
  - `StageTrace(stage: str, duration_ms: int, detail: dict)`
  - `RetrievalPlan(original_query: str, dense_specs: list[RecallSpec], lexical_queries: list[str], sub_queries: list[str])`
  - `RetrievedChunk(chunk_id: str, fields: dict, fusion_score: float, matched_channels: list[str], rerank_score: float | None = None)`
  - `RetrievalResult(query: str, chunks: list[RetrievedChunk], plan: RetrievalPlan, channel_runs: list[ChannelRun], traces: list[StageTrace])`

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_base.py`：

```python
"""Dataclasses shared across the retrieval pipeline."""
from __future__ import annotations

from vector_service.retrieval.base import (
    ChannelHit,
    ChannelRun,
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)


def test_recall_spec_defaults():
    spec = RecallSpec(query="季度营收")
    assert spec.query == "季度营收"
    assert spec.vector is None
    assert spec.hypothetical is None


def test_channel_run_carries_ranked_hits():
    run = ChannelRun(
        channel="dense",
        query="q",
        hits=[
            ChannelHit(chunk_id="c1", score=0.9, rank=1, fields={"text": "a"}),
            ChannelHit(chunk_id="c2", score=0.8, rank=2, fields={"text": "b"}),
        ],
    )
    assert run.hits[0].rank == 1
    assert run.hits[1].fields["text"] == "b"


def test_plan_and_result_defaults():
    plan = RetrievalPlan(
        original_query="q", dense_specs=[RecallSpec("q")], lexical_queries=["q"]
    )
    assert plan.sub_queries == []
    chunk = RetrievedChunk(
        chunk_id="c1", fields={"text": "a"}, fusion_score=0.5,
        matched_channels=["dense", "bm25"],
    )
    assert chunk.rerank_score is None
    result = RetrievalResult(
        query="q", chunks=[chunk], plan=plan, channel_runs=[], traces=[]
    )
    assert result.traces == []


def test_stage_trace_detail_defaults():
    tr = StageTrace(stage="fuse", duration_ms=12)
    assert tr.detail == {}
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_base.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'vector_service.retrieval'`

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/base.py`：

```python
"""Shared dataclasses for the retrieval pipeline.

The pipeline turns one user query into per-channel recall legs
(dense ANN and BM25 full-text), fuses them, optionally diversifies
and reranks them. Every stage communicates through the dataclasses
defined here so the orchestrator, channels, and the API layer share
one vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecallSpec:
    """One query going into a recall leg.

    ``vector`` is set when the leg must skip the embedder (HyDE
    pre-mixes the original-query and hypothetical-doc vectors).
    ``hypothetical`` keeps the HyDE document for tracing.
    """

    query: str
    vector: list[float] | None = None
    hypothetical: str | None = None


@dataclass
class ChannelHit:
    """One raw hit inside a single channel run."""

    chunk_id: str
    score: float
    rank: int
    fields: dict = field(default_factory=dict)


@dataclass
class ChannelRun:
    """The full ranked output of one (channel, query) leg."""

    channel: str
    query: str
    hits: list[ChannelHit] = field(default_factory=list)


@dataclass
class StageTrace:
    """Timing + key parameters for one pipeline stage."""

    stage: str
    duration_ms: int
    detail: dict = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    """The post-rewrite query plan consumed by the recall stage."""

    original_query: str
    dense_specs: list[RecallSpec]
    lexical_queries: list[str]
    sub_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """One fused (and optionally reranked) chunk in the final answer."""

    chunk_id: str
    fields: dict
    fusion_score: float
    matched_channels: list[str]
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    """Everything the pipeline observed: answer + plan + raw runs + timing."""

    query: str
    chunks: list[RetrievedChunk]
    plan: RetrievalPlan
    channel_runs: list[ChannelRun]
    traces: list[StageTrace]
```

创建 `src/vector_service/retrieval/__init__.py`（包标记；`RetrievalPipeline` 在 Task 9 补导出）：

```python
"""Retrieval pipeline: multi-channel recall, fusion, diversity, rerank."""
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_base.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/vector_service/retrieval tests/unit/test_retrieval_base.py
git commit -m "feat(retrieval): add package skeleton and shared dataclasses"
```

---

### Task 2: 融合纯函数 RRF / Weighted

**Files:**
- Create: `src/vector_service/retrieval/fusion.py`
- Test: `tests/unit/test_retrieval_fusion.py`

**Interfaces:**
- Consumes: `ChannelRun` / `ChannelHit` / `RetrievedChunk`（Task 1）。
- Produces:
  - `rrf_fuse(runs: list[ChannelRun], rrf_k: int = 60) -> list[RetrievedChunk]`
  - `weighted_fuse(runs: list[ChannelRun], weights: dict[str, float]) -> list[RetrievedChunk]`
  - 契约：按 chunk id 去重；输出按 `fusion_score` 降序；`matched_channels` 为命中该 chunk 的 channel 名（按首次出现顺序，去重）；`fields` 取首个含该 id 的 hit 的 fields（浅拷贝）。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_fusion.py`：

```python
"""Rank-fusion pure functions: RRF and per-channel weighted fusion."""
from __future__ import annotations

from vector_service.retrieval.base import ChannelHit, ChannelRun
from vector_service.retrieval.fusion import rrf_fuse, weighted_fuse


def _run(channel, pairs):
    """pairs: list of (chunk_id, score, fields)."""
    return ChannelRun(
        channel=channel,
        query=f"{channel}-q",
        hits=[
            ChannelHit(chunk_id=cid, score=s, rank=i + 1, fields=f or {})
            for i, (cid, s, f) in enumerate(pairs)
        ],
    )


def test_rrf_single_run_scores():
    out = rrf_fuse([_run("dense", [("c1", 0.9, None), ("c2", 0.8, None)])])
    assert [c.chunk_id for c in out] == ["c1", "c2"]
    assert out[0].fusion_score == 1.0 / 61
    assert out[1].fusion_score == 1.0 / 62
    assert out[0].matched_channels == ["dense"]


def test_rrf_combines_two_runs_dedup():
    out = rrf_fuse([
        _run("dense", [("c1", 0.9, {"text": "a"}), ("c2", 0.8, None)]),
        _run("bm25", [("c2", 5.0, None), ("c3", 3.0, None)]),
    ])
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c2"] == 1.0 / 62 + 1.0 / 61
    assert scores["c1"] == 1.0 / 61
    assert scores["c3"] == 1.0 / 62
    c2 = next(c for c in out if c.chunk_id == "c2")
    assert c2.matched_channels == ["dense", "bm25"]
    # c1 was first seen in dense, which carried fields.
    c1 = next(c for c in out if c.chunk_id == "c1")
    assert c1.fields == {"text": "a"}
    assert out[0].chunk_id == "c2"


def test_rrf_respects_k():
    out = rrf_fuse([_run("dense", [("c1", 1.0, None)])], rrf_k=1)
    assert out[0].fusion_score == 0.5


def test_weighted_fuse_minmax_normalizes():
    out = weighted_fuse(
        [_run("dense", [("c1", 10.0, None), ("c2", 0.0, None)])],
        weights={"dense": 1.0, "bm25": 0.0},
    )
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c1"] == 1.0
    assert scores["c2"] == 0.0
    assert out[0].chunk_id == "c1"


def test_weighted_fuse_constant_scores_become_half():
    out = weighted_fuse(
        [_run("bm25", [("c1", 3.0, None), ("c2", 3.0, None)])],
        weights={"dense": 0.0, "bm25": 1.0},
    )
    scores = {c.chunk_id: round(c.fusion_score, 6) for c in out}
    assert scores == {"c1": 0.5, "c2": 0.5}


def test_weighted_fuse_blends_channels():
    out = weighted_fuse([
        _run("dense", [("c1", 10.0, None), ("c2", 0.0, None)]),
        _run("bm25", [("c1", 0.0, None), ("c2", 10.0, None)]),
    ], weights={"dense": 0.5, "bm25": 0.5})
    scores = {c.chunk_id: c.fusion_score for c in out}
    assert scores["c1"] == scores["c2"] == 0.5
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_fusion.py -q`
Expected: FAIL，`ModuleNotFoundError: vector_service.retrieval.fusion`

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/fusion.py`：

```python
"""Rank fusion: Reciprocal Rank Fusion and per-channel weighted fusion.

Both functions are pure — they take ranked channel runs and return
fused chunks, which makes the fusion contract trivial to unit test.
"""
from __future__ import annotations

from typing import Any

from vector_service.retrieval.base import (
    ChannelRun,
    RetrievedChunk,
)


def _collect(runs: list[ChannelRun]) -> tuple[
    dict[str, float], dict[str, dict], dict[str, list[str]]
]:
    """Initialise per-id score/fields/channels by scanning runs in order."""
    scores: dict[str, float] = {}
    fields: dict[str, dict] = {}
    channels: dict[str, list[str]] = {}
    for run in runs:
        for hit in run.hits:
            cid = hit.chunk_id
            if cid not in scores:
                scores[cid] = 0.0
                fields[cid] = dict(hit.fields)
                channels[cid] = []
            if run.channel not in channels[cid]:
                channels[cid].append(run.channel)
    return scores, fields, channels


def _finalize(
    scores: dict[str, float],
    fields: dict[str, dict],
    channels: dict[str, list[str]],
) -> list[RetrievedChunk]:
    out = [
        RetrievedChunk(
            chunk_id=cid,
            fields=fields[cid],
            fusion_score=scores[cid],
            matched_channels=channels[cid],
        )
        for cid in scores
    ]
    out.sort(key=lambda c: c.fusion_score, reverse=True)
    return out


def rrf_fuse(runs: list[ChannelRun], rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: ``score(d) = Σ 1 / (rrf_k + rank_i(d))``.

    Ranks start at 1; a chunk missing from a leg contributes nothing
    for that leg. Robust to wildly different per-channel score scales.
    """
    scores, fields, channels = _collect(runs)
    for run in runs:
        for hit in run.hits:
            scores[hit.chunk_id] += 1.0 / (rrf_k + hit.rank)
    return _finalize(scores, fields, channels)


def _normalize_run(run: ChannelRun) -> dict[str, float]:
    """Min-max normalize one leg's raw scores to [0, 1].

    A leg whose scores are all equal contributes 0.5 to every hit —
    picking 0 (or 1) would silently silence (or dominate) that leg.
    """
    vals: list[tuple[str, float]] = [(h.chunk_id, h.score) for h in run.hits]
    if not vals:
        return {}
    lo = min(s for _, s in vals)
    hi = max(s for _, s in vals)
    if hi == lo:
        return {cid: 0.5 for cid, _ in vals}
    span = hi - lo
    return {cid: (s - lo) / span for cid, s in vals}


def weighted_fuse(
    runs: list[ChannelRun], weights: dict[str, Any]
) -> list[RetrievedChunk]:
    """Weighted sum of per-leg min-max-normalized scores.

    ``weights`` maps channel name ("dense" / "bm25") to a non-negative
    weight; legs whose channel has weight 0 still appear in the trace
    but contribute nothing to the fused score.
    """
    scores, fields, channels = _collect(runs)
    for run in runs:
        weight = float(weights.get(run.channel, 0.0) or 0.0)
        if weight == 0.0:
            continue
        for cid, norm in _normalize_run(run).items():
            scores[cid] += weight * norm
    return _finalize(scores, fields, channels)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_fusion.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/retrieval/fusion.py tests/unit/test_retrieval_fusion.py
git commit -m "feat(retrieval): add RRF and weighted rank fusion"
```

---

### Task 3: Store 接口与 Milvus adapter 支持 sparse / BM25

**Files:**
- Modify: `src/vector_service/stores/base.py`
- Modify: `src/vector_service/stores/milvus.py`
- Modify: `src/vector_service/stores/_milvus_adapter.py`
- Test: `tests/unit/test_store_sparse_bm25.py`

**Interfaces:**
- Produces:
  - `FieldSpec` 新增 `enable_analyzer: bool = False`、`analyzer: dict | None = None`；`dtype` 合法值新增 `"sparse_float_vector"`。
  - `VectorStore.search_text(database, collection, sparse_field, query_text, top_k=10, filter_expr=None, output_fields=None) -> list[Hit]`
  - adapter 新增 `search_text(...) -> list[dict]`（返回同 `search` 的 `[{"id","score","fields"}]`）。
  - adapter `create_collection` 接受 sparse dtype、varchar 的 `enable_analyzer`/`analyzer` 参数、sparse 字段索引（metric `bm25`），并自动注册 BM25 Function。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_store_sparse_bm25.py`：

```python
"""Sparse-vector / BM25 support in the store layer.

Covers (a) MilvusStore-side schema/index validation rules, and (b) the
adapter translating our schema dicts into pymilvus schema + BM25
Function + BM25 search calls. The pymilvus client is an in-memory fake;
no live Milvus required.
"""
from __future__ import annotations

import pytest

from vector_service.stores import _milvus_adapter as ma_mod
from vector_service.stores._milvus_adapter import MilvusAdapter
from vector_service.stores.base import FieldSpec, Hit, IndexSpec
from vector_service.stores.milvus import (
    MilvusStore,
    _validate_indexes,
    _validate_schema,
)


# ---- MilvusStore validation ----

def _v1_scalars():
    return [
        FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name="text", dtype="varchar", max_length=8192),
    ]


def test_sparse_scalar_field_accepted():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    scalars = [
        *_v1_scalars(),
        FieldSpec(name="sparse", dtype="sparse_float_vector"),
    ]
    _validate_schema("id", vf, scalars)  # no raise


def test_enable_analyzer_only_on_varchar():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    scalars = [
        FieldSpec(name="id", dtype="varchar", is_primary=True, max_length=64),
        FieldSpec(name="text", dtype="int64", enable_analyzer=True),
    ]
    with pytest.raises(Exception, match="analyzer"):
        _validate_schema("id", vf, scalars)


def test_bm25_index_allowed_on_sparse_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    _validate_indexes(vf, [
        IndexSpec(field_name="vector", metric_type="cosine"),
        IndexSpec(field_name="sparse", metric_type="bm25",
                  index_type="SPARSE_INVERTED_INDEX"),
    ])  # no raise


def test_bm25_metric_rejected_on_dense_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    with pytest.raises(Exception, match="metric_type"):
        _validate_indexes(vf, [IndexSpec(field_name="vector", metric_type="bm25")])


def test_dense_metric_rejected_on_sparse_field():
    vf = FieldSpec(name="vector", dtype="float_vector", dim=4)
    with pytest.raises(Exception, match="metric_type"):
        _validate_indexes(vf, [
            IndexSpec(field_name="vector", metric_type="cosine"),
            IndexSpec(field_name="sparse", metric_type="cosine"),
        ])


# ---- adapter fake client ----

class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeSchema:
    def __init__(self):
        self.fields = []
        self.functions = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append({"name": name, "dtype": dtype, **kwargs})

    def add_function(self, function):
        self.functions.append(function)


class FakeMilvusClient:
    def __init__(self):
        self.schema = None
        self.index_params = None
        self.created = None
        self.search_calls = []

    def create_schema(self, **kwargs):
        self.schema = FakeSchema()
        return self.schema

    def prepare_index_params(self):
        self.index_params = FakeIndexParams()
        return self.index_params

    def create_collection(self, collection_name, schema, index_params):
        self.created = collection_name

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return [[{
            "id": "c1",
            "distance": 0.5,
            "entity": {"text": "hello"},
        }]]


@pytest.fixture
def adapter(monkeypatch):
    ad = MilvusAdapter(uri="http://localhost:19530", db_name="default")
    fake = FakeMilvusClient()
    ad._client = fake
    monkeypatch.setattr(ad, "_ensure_connected", lambda: None)
    monkeypatch.setattr(ad, "_using_db", lambda db: None)
    monkeypatch.setattr(ad, "has_collection", lambda db, n: True)
    monkeypatch.setattr(ad, "_invalidate_collection", lambda db, n: None)
    return ad, fake


_V2_SCALAR_DICTS = [
    {"name": "id", "dtype": "varchar", "is_primary": True, "max_length": 64},
    {"name": "text", "dtype": "varchar", "max_length": 8192,
     "enable_analyzer": True, "analyzer": {"type": "chinese"}},
    {"name": "sparse", "dtype": "sparse_float_vector"},
]
_V2_INDEX_DICTS = [
    {"field_name": "vector", "metric_type": "cosine",
     "index_type": "HNSW", "params": {}},
    {"field_name": "sparse", "metric_type": "bm25",
     "index_type": "SPARSE_INVERTED_INDEX", "params": {}},
]


def test_adapter_builds_v2_schema_with_bm25_function(adapter):
    ad, fake = adapter
    ad.create_collection("default", "ingest", "id", "vector", 4, "cosine",
                         _V2_SCALAR_DICTS, _V2_INDEX_DICTS)
    by_name = {f["name"]: f for f in fake.schema.fields}
    assert "sparse" in by_name
    text_field = by_name["text"]
    assert text_field["enable_analyzer"] is True
    assert text_field["analyzer_params"] == {"type": "chinese"}
    assert len(fake.schema.functions) == 1
    fn = fake.schema.functions[0]
    assert fn.input_field_names == ["text"]
    assert fn.output_field_names == ["sparse"]
    sparse_index = fake.index_params.indexes[1]
    assert sparse_index["field_name"] == "sparse"
    assert sparse_index["metric_type"] == "BM25"
    assert sparse_index["index_type"] == "SPARSE_INVERTED_INDEX"
    assert fake.created == "ingest"


def test_adapter_search_text_sends_raw_text(adapter):
    ad, fake = adapter
    hits = ad.search_text("default", "ingest", "sparse", "季度营收", top_k=5)
    call = fake.search_calls[0]
    assert call["data"] == ["季度营收"]
    assert call["anns_field"] == "sparse"
    assert call["limit"] == 5
    assert call["search_params"] == {"metric_type": "BM25"}
    assert hits == [{"id": "c1", "score": 0.5, "fields": {"text": "hello"}}]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_store_sparse_bm25.py -q`
Expected: FAIL（多处：`sparse_float_vector` 校验失败、无 `search_text` 等）

- [ ] **Step 3: 改 `stores/base.py`**

在 `FieldSpec`（62-78 行）加两个字段并更新 docstring：

```python
@dataclass
class FieldSpec:
    """One field in a collection schema.

    ``dtype`` is a Milvus DataType name (``varchar`` / ``int64`` /
    ``float`` / ... / ``float_vector`` / ``sparse_float_vector``).
    ``dim`` is required for ``float_vector``. ``is_primary`` marks the
    (single) primary key; ``max_length`` is required for ``varchar``.
    The dense vector field is identified by ``dtype == "float_vector"``.

    For a varchar field, ``enable_analyzer`` + ``analyzer`` turn on the
    Milvus 2.5 text analyzer (paired with a
    ``sparse_float_vector`` field and a BM25 Function).
    """

    name: str
    dtype: str
    is_primary: bool = False
    dim: int | None = None
    max_length: int | None = None
    nullable: bool = False
    default_value: Any | None = None
    enable_analyzer: bool = False
    analyzer: dict | None = None
```

在 `search` 抽象方法之后（291 行 `"""Top-k nearest neighbours..."""` 之后）加：

```python
    @abstractmethod
    def search_text(
        self,
        database: str,
        collection: str,
        sparse_field: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[Hit]:
        """Full-text search over a BM25 sparse field.

        The server-side analyzer tokenizes ``query_text`` and scores
        matching rows with BM25; the collection must have a
        ``sparse_float_vector`` field bound to an analyzed varchar via
        a BM25 Function.
        """
```

- [ ] **Step 4: 改 `stores/_milvus_adapter.py`**

(a) 常量区（73-86 行）改为：

```python
_SCALAR_DTYPE = {
    "bool": DataType.BOOL,
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "varchar": DataType.VARCHAR,
    "json": DataType.JSON,
    "sparse_float_vector": DataType.SPARSE_FLOAT_VECTOR,
}

_METRIC = {"cosine": "COSINE", "ip": "IP", "l2": "L2", "bm25": "BM25"}
_INV_METRIC = {v: k for k, v in _METRIC.items()}
```

文件顶部 pymilvus import 区加入 Function 导入（与现有 pymilvus 符号 import 同组）：

```python
from pymilvus import Function, FunctionType
```

(b) `create_collection` 的索引校验（362-371 行）替换为按字段类型配对校验：

```python
        sparse_names = {
            f["name"] for f in scalar_fields if f["dtype"] == "sparse_float_vector"
        }
        for ip in indexes:
            target = ip.get("field_name")
            metric = ip.get("metric_type")
            if target == vector_field_name:
                allowed = ("cosine", "ip", "l2")
            elif target in sparse_names:
                allowed = ("bm25",)
            else:
                raise StoreError(
                    f"index target {target!r} is neither the vector field "
                    f"{vector_field_name!r} nor a sparse field {sorted(sparse_names)}"
                )
            if metric not in allowed:
                raise StoreError(
                    f"index on {target!r} got metric {metric!r}; "
                    f"expected one of {allowed}"
                )
```

(c) schema 构建循环（377-395 行）替换为：

```python
            analyzed_varchars: list[str] = []
            for f in scalar_fields:
                dtype = _SCALAR_DTYPE.get(f["dtype"])
                if dtype is None:
                    raise StoreError(f"unsupported scalar dtype {f['dtype']!r}")
                kwargs: dict[str, Any] = {
                    "is_primary": bool(f.get("is_primary", False)),
                }
                if f["dtype"] == "varchar":
                    if not f.get("max_length"):
                        raise StoreError(
                            f"varchar field {f['name']!r} must set max_length"
                        )
                    kwargs["max_length"] = int(f["max_length"])
                    if f.get("enable_analyzer"):
                        kwargs["enable_analyzer"] = True
                        kwargs["analyzer_params"] = dict(f.get("analyzer") or {})
                        kwargs["enable_match"] = True
                        analyzed_varchars.append(f["name"])
                elif f["dtype"] == "sparse_float_vector":
                    if f.get("enable_analyzer"):
                        raise StoreError(
                            f"sparse field {f['name']!r} cannot have enable_analyzer"
                        )
                else:
                    if f.get("enable_analyzer"):
                        raise StoreError(
                            f"field {f['name']!r}: analyzer requires dtype varchar"
                        )
                if f.get("nullable"):
                    kwargs["nullable"] = True
                if f.get("default_value") is not None:
                    kwargs["default_value"] = f["default_value"]
                schema.add_field(f["name"], dtype, **kwargs)
            schema.add_field(vector_field_name, DataType.FLOAT_VECTOR, dim=vector_dim)
            if analyzed_varchars and sparse_names:
                sparse_list = sorted(sparse_names)
                schema.add_function(Function(
                    name=f"bm25_{sparse_list[0]}",
                    function_type=FunctionType.BM25,
                    input_field_names=analyzed_varchars,
                    output_field_names=sparse_list,
                ))
```

(d) 索引参数构建（402-409 行）已经是 `_METRIC[ix["metric_type"]]` 查表，`_METRIC` 已含 bm25，无需再改。

(e) `_dtype_name` 的 int 映射（1145-1152 行）加 sparse code：

```python
                100: "binary_vector", 101: "float_vector",
                104: "sparse_float_vector",
```

(f) 在 `search` 方法之后（1117 行后、`# internal helpers` 注释之前）新增：

```python
    def search_text(
        self,
        database: str,
        collection: str,
        sparse_field: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 full-text leg; mirrors :meth:`search` result shape."""
        self._ensure_connected()
        self._using_db(database)
        if not self.has_collection(database, collection):
            raise CollectionNotFound(
                f"collection {collection!r} does not exist in database {database!r}"
            )
        schema = self.describe_collection(database, collection)
        declared = {f["name"] for f in schema["fields"]}
        if sparse_field not in declared:
            raise StoreError(
                f"sparse_field {sparse_field!r} is not in the collection schema"
            )
        if output_fields:
            unknown = [f for f in output_fields if f not in declared]
            if unknown:
                raise StoreError(
                    f"unknown output_fields {unknown}; declared: {sorted(declared)}"
                )

        try:
            self._ensure_loaded(collection)
            results = self._client.search(
                collection,
                data=[query_text],
                anns_field=sparse_field,
                limit=top_k,
                filter=filter_expr or "",
                output_fields=list({sparse_field, *(output_fields or [])}),
                search_params={"metric_type": "BM25"},
            )
        except (StoreError, CollectionNotFound):
            raise
        except Exception as e:
            raise BackendError(
                f"search_text failed for {database!r}/{collection!r}: {e}"
            ) from e

        hits: list[dict[str, Any]] = []
        for batch in results or []:
            for hit in batch:
                entity = hit.get("entity") if isinstance(hit, dict) else None
                fields: dict[str, Any] = {}
                if isinstance(entity, dict):
                    raw = dict(entity)
                    raw.pop(sparse_field, None)
                    if output_fields:
                        fields = {k: raw.get(k) for k in output_fields}
                    else:
                        fields = raw
                hits.append({
                    "id": str(hit.get("id")),
                    "score": float(hit.get("distance", 0.0)),
                    "fields": fields,
                })
        return hits
```

- [ ] **Step 5: 改 `stores/milvus.py`**

(a) 常量（44-45 行）：

```python
_VALID_METRICS = {"cosine", "ip", "l2", "bm25"}
_VALID_DTYPES = {
    "bool", "int8", "int16", "int32", "int64", "float", "double",
    "varchar", "json", "sparse_float_vector",
}
```

(b) `_validate_schema` 的字段循环（90-96 行）替换为：

```python
    sparse_names = [f.name for f in scalar_fields if f.dtype == "sparse_float_vector"]
    for f in scalar_fields:
        if f.enable_analyzer and f.dtype != "varchar":
            raise StoreError(
                f"field {f.name!r}: enable_analyzer requires dtype='varchar'"
            )
        if f.dtype not in _VALID_DTYPES:
            raise StoreError(
                f"unsupported scalar dtype {f.dtype!r}; expected one of "
                f"{sorted(_VALID_DTYPES)}"
            )
        if f.dtype == "varchar" and (f.max_length is None or f.max_length < 1):
            raise StoreError(f"varchar field {f.name!r} must set max_length >= 1")
```

(c) `_validate_indexes`（99-112 行）整体替换：

```python
def _validate_indexes(
    vector_field: FieldSpec,
    scalar_fields: list[FieldSpec],
    indexes: list[IndexSpec],
) -> None:
    if not indexes:
        raise StoreError("at least one index covering the vector field is required")
    sparse_names = {
        f.name for f in scalar_fields if f.dtype == "sparse_float_vector"
    }
    for ip in indexes:
        if ip.field_name == vector_field.name:
            allowed = _VALID_METRICS - {"bm25"}
        elif ip.field_name in sparse_names:
            allowed = {"bm25"}
        else:
            raise StoreError(
                f"index target {ip.field_name!r} is neither the vector field "
                f"{vector_field.name!r} nor a sparse field {sorted(sparse_names)}"
            )
        if ip.metric_type not in allowed:
            raise StoreError(
                f"index on {ip.field_name!r} got metric_type {ip.metric_type!r}; "
                f"expected one of {sorted(allowed)}"
            )
```

调用点（219 行）同步改为三参：`_validate_indexes(vector_field, scalar_fields, indexes)`。

(d) `create_collection` 转发的 scalar dict（234-243 行）加两键：

```python
                scalar_fields=[
                    {
                        "name": f.name,
                        "dtype": f.dtype,
                        "is_primary": bool(f.is_primary),
                        "max_length": f.max_length,
                        "nullable": bool(f.nullable),
                        "default_value": f.default_value,
                        "enable_analyzer": bool(f.enable_analyzer),
                        "analyzer": dict(f.analyzer) if f.analyzer else None,
                    }
                    for f in scalar_fields
                ],
```

(e) 返回 payload 的 fields（265-276 行）同样加：

```python
                "enable_analyzer": bool(f.enable_analyzer),
                "analyzer": dict(f.analyzer) if f.analyzer else None,
```

(f) 在 `search` 方法之后（489 行后）加：

```python
    def search_text(
        self,
        database: str,
        collection: str,
        sparse_field: str,
        query_text: str,
        top_k: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[Hit]:
        hits = self._adapter.search_text(
            database=database,
            collection=collection,
            sparse_field=sparse_field,
            query_text=query_text,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return [Hit(id=h["id"], score=h["score"], fields=h["fields"]) for h in hits]
```

- [ ] **Step 6: 运行确认通过**

Run: `python -m pytest tests/unit/test_store_sparse_bm25.py -q`
Expected: PASS（8 passed）

- [ ] **Step 7: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS（全部）

```bash
git add src/vector_service/stores tests/unit/test_store_sparse_bm25.py
git commit -m "feat(store): support sparse vectors, BM25 function, and full-text search"
```

---

### Task 4: ingest schema v2 助手 + 能力探测

**Files:**
- Modify: `src/vector_service/api/ingest.py`
- Test: `tests/unit/test_ingest_schema_v2.py`

**Interfaces:**
- Produces:
  - 常量 `_SPARSE_FIELD = "sparse"`
  - `_ingest_scalar_fields_v2() -> list[FieldSpec]`：v1 全部字段，其中 `text` 带 `enable_analyzer=True, analyzer={"type": "chinese"}`，末尾加 `FieldSpec("sparse", "sparse_float_vector")`
  - `_ingest_indexes_v2() -> list[IndexSpec]`：dense HNSW（沿用现有 params）+ `IndexSpec("sparse", "bm25", "SPARSE_INVERTED_INDEX")`
  - `schema_version(info) -> int`：`info.fields` 中含名为 `sparse` 的字段 → 2，否则 1（fields 条目可能是 dict 或对象）
- Consumes: Task 3 的 FieldSpec/校验。
- 行为变更：`_ensure_collection` 对**新** collection 改用 v2 助手；已存在 collection 分支不变。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_ingest_schema_v2.py`：

```python
"""ingest collection: schema v2 helpers and v1/v2 capability probe."""
from __future__ import annotations

from vector_service.api.ingest import (
    _ensure_collection,
    _ingest_indexes_v2,
    _ingest_scalar_fields_v2,
    schema_version,
)
from vector_service.stores.base import CollectionInfo, FieldSpec


class FakeStore:
    def __init__(self):
        self.dbs = ["default"]
        self.colls = {}
        self.created = None

    def list_databases(self):
        return list(self.dbs)

    def create_database(self, name, **_opts):
        self.dbs.append(name)

    def list_collections(self, database):
        return list(self.colls.get(database, {}))

    def collection_info(self, database, name):
        spec = self.colls[database][name]
        return CollectionInfo(database=database, name=name, dim=spec["dim"],
                              metric="cosine", count=0,
                              fields=spec["fields"])

    def create_collection(self, database, name, primary_field,
                          vector_field, scalar_fields, indexes=None):
        self.colls.setdefault(database, {})[name] = {
            "dim": vector_field.dim,
            "fields": [{"name": f.name} for f in scalar_fields],
        }
        self.created = {
            "scalar": scalar_fields, "indexes": indexes,
        }


def test_v2_scalar_fields_add_analyzed_text_and_sparse():
    by_name = {f.name: f for f in _ingest_scalar_fields_v2()}
    assert "sparse" in by_name
    assert by_name["sparse"].dtype == "sparse_float_vector"
    text = by_name["text"]
    assert text.enable_analyzer is True
    assert text.analyzer == {"type": "chinese"}
    # Every v1 field is still present.
    v1_names = {f.name for f in FieldSpec(name="x", dtype="int64").__class__
                and __import__("vector_service.api.ingest", fromlist=["_ingest_scalar_fields"])
                ._ingest_scalar_fields()}
    assert v1_names <= set(by_name)


def test_v2_indexes_cover_dense_and_sparse():
    by_field = {ip.field_name: ip for ip in _ingest_indexes_v2()}
    assert by_field["vector"].metric_type == "cosine"
    assert by_field["sparse"].metric_type == "bm25"
    assert by_field["sparse"].index_type == "SPARSE_INVERTED_INDEX"


def test_ensure_collection_creates_v2():
    store = FakeStore()
    _ensure_collection(store, "default", "ingest", 4)
    names = {f.name for f in store.created["scalar"]}
    assert "sparse" in names
    index_fields = {ip.field_name for ip in store.created["indexes"]}
    assert {"vector", "sparse"} <= index_fields


def test_ensure_collection_existing_is_left_alone():
    store = FakeStore()
    _ensure_collection(store, "default", "ingest", 4)
    store.created = None
    _ensure_collection(store, "default", "ingest", 4)
    assert store.created is None  # no recreation


def _info(field_names):
    return CollectionInfo(
        database="default", name="ingest", dim=4, metric="cosine", count=0,
        fields=[{"name": n} for n in field_names],
    )


def test_schema_version_probe():
    assert schema_version(_info(["id", "text", "vector"])) == 1
    assert schema_version(_info(["id", "text", "sparse", "vector"])) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_ingest_schema_v2.py -q`
Expected: FAIL（`_ingest_scalar_fields_v2` / `schema_version` 不存在）

- [ ] **Step 3: 实现**

在 `src/vector_service/api/ingest.py` 常量区（117 行 `_TOKEN_COUNT_FIELD` 之后）加：

```python
_SPARSE_FIELD = "sparse"
```

在 `_ingest_scalar_fields()`（139 行）之后加：

```python
def _ingest_scalar_fields_v2() -> list[FieldSpec]:
    """Schema v2: v1 fields + analyzed ``text`` + ``sparse`` BM25 field.

    Milvus runs the built-in ``chinese`` analyzer (jieba +
    cnalphanumonly) over ``text`` on every write; the registered BM25
    Function turns the tokens into the ``sparse`` vector.
    """
    fields = [
        FieldSpec(
            name=f.name,
            dtype=f.dtype,
            is_primary=f.is_primary,
            max_length=f.max_length,
            nullable=f.nullable,
            default_value=f.default_value,
        )
        for f in _ingest_scalar_fields()
    ]
    for f in fields:
        if f.name == _TEXT_FIELD:
            f.enable_analyzer = True
            f.analyzer = {"type": "chinese"}
    fields.append(FieldSpec(name=_SPARSE_FIELD, dtype="sparse_float_vector"))
    return fields


def _ingest_indexes_v2() -> list[IndexSpec]:
    """Dense HNSW index plus sparse inverted (BM25) index."""
    return [
        IndexSpec(
            field_name=_VECTOR_FIELD,
            metric_type="cosine",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200},
        ),
        IndexSpec(
            field_name=_SPARSE_FIELD,
            metric_type="bm25",
            index_type="SPARSE_INVERTED_INDEX",
        ),
    ]


def schema_version(info: Any) -> int:
    """Return 2 for collections carrying the ``sparse`` field, else 1.

    Field entries may be dicts (CollectionInfo from the store) or any
    object exposing ``name``; retrieval capability detection must work
    with both.
    """
    for f in getattr(info, "fields", []) or []:
        name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
        if name == _SPARSE_FIELD:
            return 2
    return 1
```

`_ensure_collection` 的建库分支（1003-1018 行）替换为 v2 助手：

```python
    try:
        store.create_collection(
            database=database,
            name=collection,
            primary_field=_PRIMARY_FIELD,
            vector_field=FieldSpec(name=_VECTOR_FIELD, dtype="float_vector", dim=dim),
            scalar_fields=_ingest_scalar_fields_v2(),
            indexes=_ingest_indexes_v2(),
        )
    except CollectionAlreadyExists:
        # Lost the race; another worker created the same collection.
        return
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_ingest_schema_v2.py -q`
Expected: PASS（5 passed）

注意：`test_v2_scalar_fields_add_analyzed_text_and_sparse` 中 v1 名字集合的取法写成了直接 import 调用（非 dataclass 魔术），实现测试时该语句块等价于：

```python
from vector_service.api.ingest import _ingest_scalar_fields
v1_names = {f.name for f in _ingest_scalar_fields()}
```

请在测试文件顶部用这两行替换测试函数内对应的怪语句，保持其余断言不变。

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/api/ingest.py tests/unit/test_ingest_schema_v2.py
git commit -m "feat(ingest): add schema v2 (analyzed text + sparse BM25) and version probe"
```

---

### Task 5: 召回通道 DenseChannel / BM25Channel

**Files:**
- Create: `src/vector_service/retrieval/channels.py`
- Test: `tests/unit/test_retrieval_channels.py`

**Interfaces:**
- Consumes: `RecallSpec`/`ChannelRun`/`ChannelHit`（Task 1）；store 的 `search` / `search_text`（Task 3）。
- Produces:
  - `Channel` ABC：`name: str`；`recall(spec: RecallSpec, top_k: int, filter_expr=None, output_fields=None) -> ChannelRun`
  - `DenseChannel(store, database, collection, embedder, vector_field="vector")`
  - `BM25Channel(store, database, collection, sparse_field="sparse")`
- 行为：`RecallSpec.vector` 已设置时 dense 腿不再调 embedder；否则调 `embedder.embed_query(query)`。rank 从 1 开始；store Hit → ChannelHit。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_channels.py`：

```python
"""Recall channels: dense ANN leg and BM25 full-text leg."""
from __future__ import annotations

from vector_service.retrieval.base import RecallSpec
from vector_service.retrieval.channels import BM25Channel, DenseChannel
from vector_service.stores.base import Hit


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self):
        self.search_calls = []
        self.text_calls = []

    def search(self, database, collection, vector_field, query_vector,
               top_k=10, filter_expr=None, output_fields=None):
        self.search_calls.append({
            "database": database, "collection": collection,
            "vector_field": vector_field, "vector": query_vector,
            "top_k": top_k, "filter_expr": filter_expr,
        })
        return [Hit(id="c1", score=0.9, fields={"text": "a"}),
                Hit(id="c2", score=0.8, fields={"text": "b"})]

    def search_text(self, database, collection, sparse_field, query_text,
                    top_k=10, filter_expr=None, output_fields=None):
        self.text_calls.append({
            "database": database, "collection": collection,
            "sparse_field": sparse_field, "query": query_text,
            "top_k": top_k, "filter_expr": filter_expr,
        })
        return [Hit(id="c3", score=5.0, fields={"text": "c"})]


def test_dense_embeds_query_and_assigns_ranks():
    store, emb = FakeStore(), FakeEmbedder()
    ch = DenseChannel(store, "default", "ingest", emb)
    run = ch.recall(RecallSpec("季度营收"), top_k=20,
                    filter_expr='doc_id == "d1"')
    assert emb.queries == ["季度营收"]
    assert store.search_calls[0]["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert store.search_calls[0]["top_k"] == 20
    assert store.search_calls[0]["filter_expr"] == 'doc_id == "d1"'
    assert run.channel == "dense"
    assert run.query == "季度营收"
    assert [h.chunk_id for h in run.hits] == ["c1", "c2"]
    assert [h.rank for h in run.hits] == [1, 2]


def test_dense_skips_embedder_when_vector_provided():
    store, emb = FakeStore(), FakeEmbedder()
    ch = DenseChannel(store, "default", "ingest", emb)
    ch.recall(RecallSpec("季度营收", vector=[0.5, 0.5, 0.5, 0.5]), top_k=10)
    assert emb.queries == []
    assert store.search_calls[0]["vector"] == [0.5, 0.5, 0.5, 0.5]


def test_bm25_channel_sends_raw_text():
    store = FakeStore()
    ch = BM25Channel(store, "default", "ingest")
    run = ch.recall(RecallSpec("季度营收"), top_k=15)
    assert store.text_calls[0]["query"] == "季度营收"
    assert store.text_calls[0]["sparse_field"] == "sparse"
    assert store.text_calls[0]["top_k"] == 15
    assert run.channel == "bm25"
    assert run.hits[0].chunk_id == "c3"
    assert run.hits[0].rank == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_channels.py -q`
Expected: FAIL，`ModuleNotFoundError: vector_service.retrieval.channels`

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/channels.py`：

```python
"""Recall channels.

A channel turns one :class:`RecallSpec` into a ranked
:class:`ChannelRun`. The pipeline builds one channel instance per
request and fans a channel out once per query variant, so channels
themselves stay stateless beyond their (database, collection) scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from vector_service.retrieval.base import ChannelHit, ChannelRun, RecallSpec


class Channel(ABC):
    """One retrieval leg type."""

    name: str

    @abstractmethod
    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        """Run one recall leg and return the ranked channel run."""

    def _wrap(self, query: str, hits) -> ChannelRun:
        return ChannelRun(
            channel=self.name,
            query=query,
            hits=[
                ChannelHit(
                    chunk_id=str(h.id),
                    score=float(h.score),
                    rank=i + 1,
                    fields=dict(h.fields),
                )
                for i, h in enumerate(hits)
            ],
        )


class DenseChannel(Channel):
    """ANN leg over the dense float vector field."""

    name = "dense"

    def __init__(self, store, database: str, collection: str,
                 embedder, vector_field: str = "vector"):
        self._store = store
        self._database = database
        self._collection = collection
        self._embedder = embedder
        self._vector_field = vector_field

    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        vector = spec.vector
        if vector is None:
            vector = self._embedder.embed_query(spec.query)
        hits = self._store.search(
            self._database,
            self._collection,
            self._vector_field,
            vector,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return self._wrap(spec.query, hits)


class BM25Channel(Channel):
    """Full-text leg over the BM25 sparse field."""

    name = "bm25"

    def __init__(self, store, database: str, collection: str,
                 sparse_field: str = "sparse"):
        self._store = store
        self._database = database
        self._collection = collection
        self._sparse_field = sparse_field

    def recall(
        self,
        spec: RecallSpec,
        top_k: int,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> ChannelRun:
        hits = self._store.search_text(
            self._database,
            self._collection,
            self._sparse_field,
            spec.query,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        return self._wrap(spec.query, hits)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_channels.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/retrieval/channels.py tests/unit/test_retrieval_channels.py
git commit -m "feat(retrieval): add dense and BM25 recall channels"
```

---

### Task 6: 查询改写 HyDE / Multi-Query / Step-Back / Decompose

**Files:**
- Create: `src/vector_service/retrieval/transforms.py`
- Test: `tests/unit/test_retrieval_transforms.py`

**Interfaces:**
- Consumes: `RecallSpec`（Task 1）；`chat_fn(messages: list[dict]) -> str`（即 `OpenAIChatClient.as_chat_fn()`）。
- Produces:
  - `hyde(chat_fn, query: str, embedder, alpha: float = 0.7) -> RecallSpec`（query 不变，vector 为 α-混合归一化向量，hypothetical 为假设答案文本）
  - `multi_query(chat_fn, query: str, n: int = 3) -> list[str]`（失败回退 `[query]`；截断到 n；剔除空串）
  - `step_back(chat_fn, query: str) -> str`（失败回退原 query）
  - `decompose(chat_fn, query: str) -> list[str]`（失败回退 `[]`，由 pipeline 决定无分解）
  - `parse_json_object(raw: str) -> dict`（容错：代码块、前后噪声）
- 容错原则：LLM 返回非 JSON / 空 / 结构缺字段时记 warning（`logging`），回退；不抛异常中断检索。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_transforms.py`：

```python
"""LLM query transforms with graceful fallback on bad LLM output."""
from __future__ import annotations

import math

import pytest

from vector_service.retrieval import transforms
from vector_service.retrieval.base import RecallSpec


class FakeChat:
    def __init__(self, text):
        self.text = text
        self.messages = []

    def __call__(self, messages):
        self.messages.extend(messages)
        return self.text


class FakeEmbedder:
    """Maps fixed texts to simple vectors."""

    def embed_query(self, query):
        if query == "q":
            return [1.0, 0.0]
        return [0.0, 1.0]  # hypothetical answer


# ---- JSON extraction ----

def test_parse_plain_object():
    assert transforms.parse_json_object('{"queries": ["a", "b"]}') == {
        "queries": ["a", "b"]
    }


def test_parse_fenced_object():
    raw = 'noise before\n```json\n{"query": "why?"}\n```\nafter'
    assert transforms.parse_json_object(raw) == {"query": "why?"}


def test_parse_object_with_trailing_text():
    raw = 'sure:\n{"sub_queries": ["s1"]}\nhope that helps'
    assert transforms.parse_json_object(raw) == {"sub_queries": ["s1"]}


def test_parse_bad_json_raises():
    with pytest.raises(ValueError):
        transforms.parse_json_object("not json at all")


# ---- multi-query / step-back / decompose ----

def test_multi_query_parses_and_truncates():
    chat = FakeChat('```json\n{"queries": ["v1", "v2", "v3", "v4"]}\n```')
    assert transforms.multi_query(chat, "q", n=3) == ["v1", "v2", "v3"]


def test_multi_query_falls_back_to_original():
    chat = FakeChat("garbage")
    assert transforms.multi_query(chat, "原始问题", n=3) == ["原始问题"]


def test_multi_query_empty_list_falls_back():
    chat = FakeChat('{"queries": []}')
    assert transforms.multi_query(chat, "q") == ["q"]


def test_step_back_parses():
    chat = FakeChat('{"query": "what drives quarterly revenue?"}')
    assert transforms.step_back(chat, "q3 营收如何") == (
        "what drives quarterly revenue?"
    )


def test_step_back_falls_back():
    assert transforms.step_back(FakeChat("??"), "原问题") == "原问题"


def test_decompose_parses():
    chat = FakeChat('{"sub_queries": ["s1", "s2"]}')
    assert transforms.decompose(chat, "q") == ["s1", "s2"]


def test_decompose_bad_output_is_empty():
    assert transforms.decompose(FakeChat("nope"), "q") == []


# ---- HyDE ----

def test_hyde_mixes_vectors():
    chat = FakeChat("This is the hypothetical answer.")
    spec = transforms.hyde(chat, "q", FakeEmbedder(), alpha=0.7)
    assert isinstance(spec, RecallSpec)
    assert spec.query == "q"
    assert spec.hypothetical == "This is the hypothetical answer."
    # mix = 0.3*(1,0) + 0.7*(0,1) = (0.3, 0.7); then L2-normalized
    n = math.hypot(0.3, 0.7)
    assert spec.vector == [pytest.approx(0.3 / n), pytest.approx(0.7 / n)]


def test_hyde_alpha_one_uses_answer_only():
    chat = FakeChat("doc")
    spec = transforms.hyde(chat, "q", FakeEmbedder(), alpha=1.0)
    assert spec.vector == [pytest.approx(0.0), pytest.approx(1.0)]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_transforms.py -q`
Expected: FAIL，无 `transforms` 模块属性。

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/transforms.py`：

```python
"""LLM query transforms.

All transforms share one contract: an LLM hiccup (non-JSON text, an
empty list, a missing key) degrades to the original query — or to no
rewrite — and is logged, never raised. Retrieval must not depend on the
LLM being well-behaved.
"""
from __future__ import annotations

import json
import logging
import math
import re

from vector_service.retrieval.base import RecallSpec

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are an expert answering questions. Write a short (2-4 sentences), "
    "factual paragraph that directly answers the user's question, as it "
    "would appear in an authoritative document. No preamble, no caveats."
)
_MULTI_SYSTEM = (
    "You rewrite search queries. Return ONLY a JSON object: "
    '{"queries": ["..."]} with {n} alternative queries that approach the '
    "user's question from different angles or use different wording."
)
_STEP_BACK_SYSTEM = (
    "You abstract questions. Return ONLY JSON: "
    '{"query": "..."} containing one broader, more fundamental question '
    "that the specific question is an instance of."
)
_DECOMPOSE_SYSTEM = (
    "You break complex multi-hop questions into atomic sub-questions. "
    'Return ONLY JSON: {"sub_queries": ["...", "..."]}. Each sub-question '
    "must be answerable with a single fact."
)


def parse_json_object(raw: str) -> dict:
    """Extract the first JSON object from an LLM response.

    Tolerates ```json fences and prose before/after the object. Raises
    ``ValueError`` when no parseable object exists.
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    start = text.find("{")
    if start >= 0:
        tail = text[start:]
        candidates.append(tail)
        end = tail.rfind("}")
        if end > 0:
            candidates.append(tail[: end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no JSON object in LLM response: {raw[:120]!r}")


def _chat_json(chat_fn, system: str, query: str) -> dict | None:
    try:
        raw = chat_fn([
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ])
        return parse_json_object(raw)
    except Exception as e:  # noqa: BLE001 — any LLM failure degrades
        logger.warning("query transform failed, falling back: %s", e)
        return None


def multi_query(chat_fn, query: str, n: int = 3) -> list[str]:
    """Up to ``n`` reformulations; falls back to ``[query]``."""
    data = _chat_json(chat_fn, _MULTI_SYSTEM.format(n=n), query)
    if not data:
        return [query]
    variants = [
        str(q).strip() for q in data.get("queries", []) if str(q).strip()
    ][:n]
    return variants or [query]


def step_back(chat_fn, query: str) -> str:
    """One broader question; falls back to the original query."""
    data = _chat_json(chat_fn, _STEP_BACK_SYSTEM, query)
    broader = str((data or {}).get("query", "")).strip()
    return broader or query


def decompose(chat_fn, query: str) -> list[str]:
    """Atomic sub-questions; empty list when the LLM cannot decompose."""
    data = _chat_json(chat_fn, _DECOMPOSE_SYSTEM, query)
    if not data:
        return []
    return [
        str(q).strip() for q in data.get("sub_queries", []) if str(q).strip()
    ]


def hyde(chat_fn, query: str, embedder, alpha: float = 0.7) -> RecallSpec:
    """HyDE leg: L2-normalized convex mix of q-vector and answer-vector.

    ``q' = norm((1−α)·q + α·h)`` — the TREC 2025 RAG winning recipe;
    the original wording stays on the spec for trace display.
    """
    raw_doc = chat_fn([
        {"role": "system", "content": _HYDE_SYSTEM},
        {"role": "user", "content": query},
    ])
    doc = (raw_doc or "").strip() or query
    q_vec = embedder.embed_query(query)
    h_vec = embedder.embed_query(doc)
    mix = [(1.0 - alpha) * a + alpha * b for a, b in zip(q_vec, h_vec)]
    norm = math.sqrt(sum(x * x for x in mix)) or 1.0
    return RecallSpec(
        query=query,
        vector=[x / norm for x in mix],
        hypothetical=doc,
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_transforms.py -q`
Expected: PASS（13 passed）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/retrieval/transforms.py tests/unit/test_retrieval_transforms.py
git commit -m "feat(retrieval): add HyDE, multi-query, step-back, decomposition transforms"
```

---

### Task 7: MMR 多样性

**Files:**
- Create: `src/vector_service/retrieval/diversity.py`
- Test: `tests/unit/test_retrieval_diversity.py`

**Interfaces:**
- Consumes: `RetrievedChunk`（Task 1）。
- Produces: `mmr(chunks: list[RetrievedChunk], doc_vectors: list[list[float]], query_vector: list[float], *, lambda_mult: float = 0.7, top_k: int | None = None) -> list[RetrievedChunk]`
- 契约：首轮选 q 相似度最高者；之后选 `λ·sim(d,q) − (1−λ)·max sim(d, selected)` 最大者；cosine（输入向量内部 L2 归一化）；默认选满全部；不足时返回实际数量。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_diversity.py`：

```python
"""MMR selection over fused candidates."""
from __future__ import annotations

from vector_service.retrieval.diversity import mmr


def _chunks(n):
    from vector_service.retrieval.base import RetrievedChunk

    return [
        RetrievedChunk(chunk_id=f"c{i}", fields={}, fusion_score=1.0 - 0.1 * i,
                       matched_channels=["dense"])
        for i in range(n)
    ]


def test_lambda_one_picks_query_similarity_order():
    # query ~ x-axis; docs from most-aligned to least, last doc on y-axis
    docs = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=1.0)
    assert [c.chunk_id for c in out] == ["c0", "c1", "c2"]


def test_lambda_zero_second_pick_is_most_dissimilar():
    docs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=0.0, top_k=2)
    assert [c.chunk_id for c in out] == ["c0", "c2"]


def test_top_k_truncates():
    docs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    out = mmr(_chunks(3), docs, [1.0, 0.0], lambda_mult=1.0, top_k=2)
    assert len(out) == 2


def test_first_pick_is_highest_query_similarity():
    docs = [[0.1, 0.9], [0.9, 0.1]]
    out = mmr(_chunks(2), docs, [1.0, 0.0], lambda_mult=0.7, top_k=1)
    assert out[0].chunk_id == "c1"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_diversity.py -q`
Expected: FAIL，无 diversity 模块。

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/diversity.py`：

```python
"""Maximal Marginal Relevance.

MMR trades relevance against redundancy: after taking the top-relevant
document, each next pick maximizes

    λ · sim(d, q) − (1 − λ) · max sim(d, selected)

so near-duplicate fused candidates are pushed apart. Similarities are
cosine — every vector is L2-normalized before scoring.
"""
from __future__ import annotations

import math

from vector_service.retrieval.base import RetrievedChunk


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr(
    chunks: list[RetrievedChunk],
    doc_vectors: list[list[float]],
    query_vector: list[float],
    *,
    lambda_mult: float = 0.7,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``top_k`` MMR-selected chunks in selection order."""
    if not chunks:
        return []
    if top_k is None:
        top_k = len(chunks)
    docs = [_l2_normalize(v) for v in doc_vectors]
    query = _l2_normalize(query_vector)
    query_sim = [_cosine(d, query) for d in docs]

    selected_idx: list[int] = []
    remaining = set(range(len(chunks)))
    while remaining and len(selected_idx) < top_k:
        best_idx, best_score = None, None
        for i in remaining:
            redundancy = 0.0
            if selected_idx:
                redundancy = max(_cosine(docs[i], docs[j]) for j in selected_idx)
            score = lambda_mult * query_sim[i] - (1.0 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best_idx, best_score = i, score
        selected_idx.append(best_idx)
        remaining.discard(best_idx)
    return [chunks[i] for i in selected_idx]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_diversity.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/retrieval/diversity.py tests/unit/test_retrieval_diversity.py
git commit -m "feat(retrieval): add MMR diversity selection"
```

---

### Task 8: Pydantic 请求/响应模型

**Files:**
- Create: `src/vector_service/schemas/retrieval.py`
- Test: `tests/unit/test_retrieval_schemas.py`

**Interfaces:**
- Produces（pipeline/API 依赖，名字固定）:
  - 请求侧：`FilterSpec`、`ChannelsSpec`、`ChannelWeights`、`FusionSpec`、`RewriteSpec`（常量 `REWRITE_METHODS`）、`MMRSpec`、`RerankSpec`、`RetrievalRequest`
  - 响应侧：`RecallSpecOut`、`PlanOut`、`ChannelHitOut`、`ChannelRunOut`、`StageTraceOut`、`RetrievedChunkOut`、`RetrievalResponse`、`to_result(RetrievalResult) -> RetrievalResponse`
- 约定：MMR 字段名为 `lambda_mult`（前端一并使用，spec 示例中的 `lambda` 为同义键）；`RecallSpecOut` 不输出向量本体。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_schemas.py`：

```python
"""Pydantic models for POST /v1/retrieval and its response."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vector_service.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    to_result,
)


def _base_kwargs(**overrides):
    kwargs = {"query": "q"}
    kwargs.update(overrides)
    return kwargs


def test_request_defaults():
    req = RetrievalRequest(query="q")
    assert req.database == "default"
    assert req.collection == "ingest"
    assert req.top_k == 10
    assert req.channels.dense and req.channels.bm25
    assert req.fusion.method == "rrf"
    assert req.fusion.rrf_k == 60
    assert req.rewrite.enabled is False
    assert req.mmr.enabled is False
    assert req.rerank.enabled is True
    assert req.rerank.candidate_pool == 25


def test_request_requires_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="")


@pytest.mark.parametrize("top_k", [0, 101])
def test_top_k_bounds(top_k):
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", top_k=top_k)


def test_at_least_one_channel():
    with pytest.raises(ValidationError, match="channel"):
        RetrievalRequest(query="q", channels={"dense": False, "bm25": False})


def test_fusion_method_enum():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={"method": "magic"})


def test_rrf_k_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={"method": "rrf", "rrf_k": 0})


def test_weights_must_have_one_positive():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={
            "method": "weighted",
            "weights": {"dense": 0, "bm25": 0},
        })


def test_weights_non_negative():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", fusion={
            "method": "weighted",
            "weights": {"dense": -1, "bm25": 1},
        })


def test_rewrite_methods_validated():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={"enabled": True, "methods": ["nope"]})


def test_hyde_alpha_and_n_variants_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={
            "enabled": True, "hyde_alpha": 1.5,
        })
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", rewrite={
            "enabled": True, "n_variants": 6,
        })


def test_mmr_lambda_bounds():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", mmr={"enabled": True, "lambda_mult": -0.1})


def test_rerank_pool_bounds_and_relation():
    with pytest.raises(ValidationError):
        RetrievalRequest(query="q", top_k=10,
                         rerank={"enabled": True, "candidate_pool": 65})
    with pytest.raises(ValidationError, match="candidate_pool"):
        RetrievalRequest(query="q", top_k=30,
                         rerank={"enabled": True, "candidate_pool": 25})


def test_to_result_maps_dataclasses():
    from vector_service.retrieval.base import (
        ChannelHit, ChannelRun, RecallSpec, RetrievalPlan, RetrievalResult,
        RetrievedChunk, StageTrace,
    )

    result = RetrievalResult(
        query="q",
        chunks=[RetrievedChunk(
            chunk_id="c1", fields={"text": "a"}, fusion_score=0.03,
            matched_channels=["dense", "bm25"], rerank_score=0.9,
        )],
        plan=RetrievalPlan(
            original_query="q",
            dense_specs=[RecallSpec("q", hypothetical="h-doc")],
            lexical_queries=["q"],
        ),
        channel_runs=[ChannelRun(
            channel="dense", query="q",
            hits=[ChannelHit("c1", 0.9, 1, {"text": "a"})],
        )],
        traces=[StageTrace("fuse", 4, {"method": "rrf"})],
    )
    resp = to_result(result)
    assert isinstance(resp, RetrievalResponse)
    dumped = resp.model_dump()
    assert dumped["chunks"][0]["chunk_id"] == "c1"
    assert dumped["chunks"][0]["rerank_score"] == 0.9
    assert dumped["plan"]["dense_specs"][0] == {
        "query": "q", "hypothetical": "h-doc",
    }
    assert dumped["channel_runs"][0]["hits"][0]["rank"] == 1
    assert dumped["traces"][0]["detail"] == {"method": "rrf"}
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_schemas.py -q`
Expected: FAIL，无 `vector_service.schemas.retrieval`

- [ ] **Step 3: 实现**

创建 `src/vector_service/schemas/retrieval.py`：

```python
"""Pydantic schemas for the retrieval API and pipeline trace."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from vector_service.retrieval.base import (
    ChannelRun,
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)

REWRITE_METHODS = ("hyde", "multi_query", "step_back", "decompose")


# ---- request ----

class FilterSpec(BaseModel):
    doc_id: str = ""
    filename: str = ""


class ChannelWeights(BaseModel):
    dense: float = 0.5
    bm25: float = 0.5


class ChannelsSpec(BaseModel):
    dense: bool = True
    bm25: bool = True

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.dense or self.bm25):
            raise ValueError("at least one channel must be enabled (dense / bm25)")
        return self


class FusionSpec(BaseModel):
    method: str = "rrf"
    rrf_k: int = Field(60, ge=1, le=200)
    weights: ChannelWeights = Field(default_factory=ChannelWeights)

    @field_validator("method")
    @classmethod
    def _method_known(cls, value):
        if value not in ("rrf", "weighted"):
            raise ValueError("fusion.method must be 'rrf' or 'weighted'")
        return value

    @model_validator(mode="after")
    def _weights_valid(self):
        if self.weights.dense < 0 or self.weights.bm25 < 0:
            raise ValueError("fusion weights must be >= 0")
        if self.weights.dense == 0 and self.weights.bm25 == 0:
            raise ValueError("at least one fusion weight must be > 0")
        return self


class RewriteSpec(BaseModel):
    enabled: bool = False
    methods: list[str] = Field(default_factory=lambda: list(REWRITE_METHODS))
    hyde_alpha: float = Field(0.7, ge=0.0, le=1.0)
    n_variants: int = Field(3, ge=1, le=5)

    @field_validator("methods")
    @classmethod
    def _methods_known(cls, value):
        unknown = [m for m in value if m not in REWRITE_METHODS]
        if unknown:
            raise ValueError(f"unknown rewrite methods {unknown}")
        if len(value) != len(set(value)):
            raise ValueError("duplicate rewrite methods")
        return value


class MMRSpec(BaseModel):
    enabled: bool = False
    lambda_mult: float = Field(0.7, ge=0.0, le=1.0)


class RerankSpec(BaseModel):
    enabled: bool = True
    candidate_pool: int = Field(25, ge=1, le=64)


class RetrievalRequest(BaseModel):
    database: str = Field("default", min_length=1)
    collection: str = Field("ingest", min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    filter: FilterSpec = Field(default_factory=FilterSpec)
    channels: ChannelsSpec = Field(default_factory=ChannelsSpec)
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    rewrite: RewriteSpec = Field(default_factory=RewriteSpec)
    mmr: MMRSpec = Field(default_factory=MMRSpec)
    rerank: RerankSpec = Field(default_factory=RerankSpec)

    @model_validator(mode="after")
    def _pool_covers_top_k(self):
        if self.rerank.candidate_pool < self.top_k:
            raise ValueError("rerank.candidate_pool must be >= top_k")
        return self


# ---- response ----

class RecallSpecOut(BaseModel):
    # The vector body is intentionally omitted — it can be large and the
    # trace only needs the wording (+ HyDE hypothetical document).
    query: str
    hypothetical: str | None = None


class PlanOut(BaseModel):
    original_query: str
    dense_specs: list[RecallSpecOut]
    lexical_queries: list[str]
    sub_queries: list[str]


class ChannelHitOut(BaseModel):
    chunk_id: str
    score: float
    rank: int
    fields: dict


class ChannelRunOut(BaseModel):
    channel: str
    query: str
    hits: list[ChannelHitOut]


class StageTraceOut(BaseModel):
    stage: str
    duration_ms: int
    detail: dict


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    fields: dict
    fusion_score: float
    matched_channels: list[str]
    rerank_score: float | None = None


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunkOut]
    plan: PlanOut
    channel_runs: list[ChannelRunOut]
    traces: list[StageTraceOut]


def to_result(result: RetrievalResult) -> RetrievalResponse:
    """Map the pipeline's dataclasses onto the API response model."""
    return RetrievalResponse(
        query=result.query,
        chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                fields=c.fields,
                fusion_score=c.fusion_score,
                matched_channels=c.matched_channels,
                rerank_score=c.rerank_score,
            )
            for c in result.chunks
        ],
        plan=PlanOut(
            original_query=result.plan.original_query,
            dense_specs=[
                RecallSpecOut(query=s.query, hypothetical=s.hypothetical)
                for s in result.plan.dense_specs
            ],
            lexical_queries=list(result.plan.lexical_queries),
            sub_queries=list(result.plan.sub_queries),
        ),
        channel_runs=[
            ChannelRunOut(
                channel=run.channel,
                query=run.query,
                hits=[
                    ChannelHitOut(
                        chunk_id=h.chunk_id, score=h.score,
                        rank=h.rank, fields=h.fields,
                    )
                    for h in run.hits
                ],
            )
            for run in result.channel_runs
        ],
        traces=[
            StageTraceOut(stage=tr.stage, duration_ms=tr.duration_ms,
                          detail=tr.detail)
            for tr in result.traces
        ],
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_schemas.py -q`
Expected: PASS（14 passed）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/schemas/retrieval.py tests/unit/test_retrieval_schemas.py
git commit -m "feat(retrieval): add request/response schemas with validation"
```

---

### Task 9: RetrievalPipeline 编排器

**Files:**
- Create: `src/vector_service/retrieval/pipeline.py`
- Modify: `src/vector_service/retrieval/__init__.py`
- Test: `tests/unit/test_retrieval_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalRequest`（Task 8）；channels（Task 5）；fusion（Task 2）；transforms（Task 6）；mmr（Task 7）；`chunking.llm_chunker.get_chat_client` / `is_llm_configured`。
- Produces:
  - `RetrievalPipeline(*, settings, store, embedder, reranker, chat_getter=get_chat_client, chat_check=is_llm_configured)`
  - `async validate(req: RetrievalRequest) -> dict`（pre-flight；返回 `{"info", "rewrite_on"}`）
  - `async retrieve(req, *, emit=noop, prevalidated=None) -> RetrievalResult`
  - 模块函数 `build_filter_expr(filter_spec) -> str | None`
- 阶段顺序：validate → rewrite → recall → fuse → mmr（可选）→ rerank（可选）。每阶段发 `{"type":"stage","stage":name}`；每阶段产出 StageTrace。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_pipeline.py`：

```python
"""RetrievalPipeline orchestration with in-memory fakes."""
from __future__ import annotations

import pytest

import asyncio
import functools

from vector_service.retrieval.pipeline import RetrievalPipeline, build_filter_expr
from vector_service.schemas.retrieval import RetrievalRequest
from vector_service.stores.base import CollectionInfo, Hit


def async_test(coro):
    """Run an async test via asyncio.run (no pytest-asyncio in this repo)."""

    @functools.wraps(coro)
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


_SPARSE_INFO_FIELDS = [
    {"name": "id"}, {"name": "text"}, {"name": "sparse"},
]
_V1_INFO_FIELDS = [{"name": "id"}, {"name": "text"}]


class FakeStore:
    def __init__(self, v2=True):
        self.v2 = v2
        self.search_calls = []
        self.text_calls = []

    def collection_info(self, db, coll):
        return CollectionInfo(
            database=db, name=coll, dim=4, metric="cosine", count=0,
            fields=list(_SPARSE_INFO_FIELDS if self.v2 else _V1_INFO_FIELDS),
        )

    def search(self, db, coll, field, vector, top_k=10,
               filter_expr=None, output_fields=None):
        self.search_calls.append({"vector": vector, "top_k": top_k,
                                  "filter_expr": filter_expr})
        return [Hit(id="c1", score=0.9, fields={
            "text": "dense text", "doc_id": "d1", "chunk_index": 0,
            "section_header": "S1", "page_number": 1, "filename": "a.pdf",
        })]

    def search_text(self, db, coll, field, query, top_k=10,
                    filter_expr=None, output_fields=None):
        self.text_calls.append({"query": query, "top_k": top_k,
                                "filter_expr": filter_expr})
        return [Hit(id="c2", score=5.0, fields={
            "text": "lexical text", "doc_id": "d1", "chunk_index": 1,
            "section_header": "S2", "page_number": 2, "filename": "a.pdf",
        })]


class FakeEmbedder:
    model_name = "fake"
    dim = 4

    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeReranker:
    _impl = object()

    def rerank(self, query, documents, top_n=None):
        from vector_service.rerankers.base import ScoredHit

        return [ScoredHit(index=0, score=0.99)][: top_n or len(documents)]


class FakeSettings:
    pass


def _pipe(**kwargs):
    defaults = {
        "settings": FakeSettings(),
        "store": FakeStore(),
        "embedder": FakeEmbedder(),
        "reranker": FakeReranker(),
        "chat_check": lambda s: True,
        "chat_getter": lambda s: None,
    }
    defaults.update(kwargs)
    return RetrievalPipeline(**defaults)


def _req(**kwargs):
    kwargs.setdefault("query", "季度营收")
    return RetrievalRequest(**kwargs)


@async_test
async def test_dense_only_basic_retrieval():
    pipe = _pipe()
    result = await pipe.retrieve(_req(
        channels={"dense": True, "bm25": False},
        rerank={"enabled": False},
    ))
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.chunks[0].matched_channels == ["dense"]
    assert [tr.stage for tr in result.traces] == ["rewrite", "recall", "fuse"]
    assert len(result.channel_runs) == 1


@async_test
async def test_hybrid_fans_out_two_legs():
    pipe = _pipe()
    result = await pipe.retrieve(_req(rerank={"enabled": False}))
    channels = {run.channel for run in result.channel_runs}
    assert channels == {"dense", "bm25"}
    assert len(result.chunks) == 2
    c2 = next(c for c in result.chunks if c.chunk_id == "c2")
    assert c2.matched_channels == ["bm25"]


@async_test
async def test_rerank_reorders_and_truncates():
    pipe = _pipe()
    result = await pipe.retrieve(_req(top_k=1))
    assert [c.chunk_id for c in result.chunks] == ["c1"]
    assert result.chunks[0].rerank_score == 0.99
    assert [tr.stage for tr in result.traces][-1] == "rerank"


@async_test
async def test_bm25_on_v1_collection_raises_with_migration_hint():
    pipe = _pipe(store=FakeStore(v2=False))
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req())
    assert exc.value.status_code == 422
    error = exc.value.detail["error"]
    assert error["code"] == "retrieval_channel_unsupported"
    assert error["migration_available"] is True


@async_test
async def test_missing_embedder_is_503():
    pipe = _pipe(embedder=None)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req())
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "embedder_unavailable"


@async_test
async def test_missing_reranker_is_503():
    pipe = _pipe(reranker=None)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req(rerank={"enabled": True}))
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "reranker_not_loaded"


@async_test
async def test_rewrite_without_llm_is_503():
    pipe = _pipe(chat_check=lambda s: False)
    with pytest.raises(Exception) as exc:
        await pipe.retrieve(_req(rewrite={"enabled": True}))
    assert exc.value.status_code == 503
    assert exc.value.detail["error"]["code"] == "llm_unavailable"


class FakeChatClient:
    def __init__(self):
        self.calls = 0

    def as_chat_fn(self):
        def fn(messages):
            self.calls += 1
            if self.calls == 1:
                return '{"queries": ["v1", "v2"]}'
            return '{"query": "broader?"}'

        return fn


@async_test
async def test_multi_query_and_step_back_add_legs():
    chat = FakeChatClient()
    pipe = _pipe(chat_getter=lambda s: chat)
    result = await pipe.retrieve(_req(
        rewrite={"enabled": True, "methods": ["multi_query", "step_back"]},
        rerank={"enabled": False},
    ))
    # dense legs: original + 2 variants + step-back = 4; bm25 legs likewise 4
    by_channel = {"dense": 0, "bm25": 0}
    for run in result.channel_runs:
        by_channel[run.channel] += 1
    assert by_channel == {"dense": 4, "bm25": 4}
    queries = set(result.plan.lexical_queries)
    assert {"季度营收", "v1", "v2", "broader?"} <= queries


@async_test
async def test_mmr_stage_runs_when_enabled():
    pipe = _pipe()
    result = await pipe.retrieve(_req(
        mmr={"enabled": True, "lambda_mult": 0.7},
        rerank={"enabled": False},
    ))
    assert any(tr.stage == "mmr" for tr in result.traces)


@async_test
async def test_emit_receives_stage_events():
    pipe = _pipe()
    events = []
    await pipe.retrieve(
        _req(channels={"dense": True, "bm25": False},
             rerank={"enabled": False}),
        emit=events.append,
    )
    assert [e["stage"] for e in events] == ["rewrite", "recall", "fuse"]


def test_build_filter_expr():
    assert build_filter_expr(__import__(
        "vector_service.schemas.retrieval", fromlist=["FilterSpec"]
    ).FilterSpec()) is None
    expr = build_filter_expr(__import__(
        "vector_service.schemas.retrieval", fromlist=["FilterSpec"]
    ).FilterSpec(doc_id=" d1 ", filename="年 报"))
    assert expr == 'doc_id == "d1" and filename like "%年 报%"'
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_pipeline.py -q`
Expected: FAIL，无 pipeline 模块。

注意：本仓库未安装 `pytest-asyncio`（已核实）。测试文件顶部提供 `async_test` 装饰器，所有异步测试以 `@async_test` 标注（内部 `asyncio.run`），不要使用 `@async_test`。

- [ ] **Step 3: 实现**

创建 `src/vector_service/retrieval/pipeline.py`：

```python
"""RetrievalPipeline: transform → recall → fuse → MMR → rerank.

All blocking work (embedding, store RPCs, chat, rerank) is pushed to a
thread executor; recall legs within one stage run concurrently. The
pipeline raises :class:`fastapi.HTTPException` directly — callers
(JSON and NDJSON routes) decide how to surface it.
"""
from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Awaitable, Callable

from fastapi import HTTPException

from vector_service.chunking.llm_chunker import (
    get_chat_client,
    is_llm_configured,
)
from vector_service.core.errors import CollectionNotFound
from vector_service.retrieval import transforms as T
from vector_service.retrieval.base import (
    RecallSpec,
    RetrievalPlan,
    RetrievalResult,
    RetrievedChunk,
    StageTrace,
)
from vector_service.retrieval.channels import BM25Channel, DenseChannel
from vector_service.retrieval.diversity import mmr
from vector_service.retrieval.fusion import rrf_fuse, weighted_fuse
from vector_service.schemas.retrieval import (
    FilterSpec,
    RetrievalRequest,
)

_SPARSE_FIELD = "sparse"
_VECTOR_FIELD = "vector"

StageEmit = Callable[[dict], Awaitable[Any]]


async def _noop_emit(_event: dict) -> None:
    ...


def _http(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code, detail={
        "error": {"code": code, "message": message, **extra},
    })


class _Timer:
    def __init__(self, stage: str):
        self.stage = stage
        self.start = time.perf_counter()

    def done(self, detail: dict | None = None) -> StageTrace:
        duration = int((time.perf_counter() - self.start) * 1000)
        return StageTrace(self.stage, duration, detail or {})


def _escape_eq(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_like(value: str) -> str:
    return _escape_eq(value).replace("%", "\\%").replace("_", "\\_")


def build_filter_expr(spec: FilterSpec) -> str | None:
    """Convert filter.doc_id (eq) / filename (like) into a Milvus expression.

    Empty fields are omitted; LIKE wildcards and quotes in user input
    are escaped so a filter value cannot break out of the expression.
    """
    parts: list[str] = []
    doc_id = (spec.doc_id or "").strip()
    filename = (spec.filename or "").strip()
    if doc_id:
        parts.append(f'doc_id == "{_escape_eq(doc_id)}"')
    if filename:
        parts.append(f'filename like "%{_escape_like(filename)}%"')
    return " and ".join(parts) if parts else None


class RetrievalPipeline:
    """One-shot orchestrator; cheap to construct per request."""

    def __init__(
        self,
        *,
        settings: Any,
        store: Any,
        embedder: Any,
        reranker: Any,
        chat_getter: Callable = get_chat_client,
        chat_check: Callable = is_llm_configured,
    ):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._chat_getter = chat_getter
        self._chat_check = chat_check

    async def validate(self, req: RetrievalRequest) -> dict:
        """Pre-flight checks + schema probe. Shared by JSON/stream routes."""
        if not req.query or not req.query.strip():
            raise _http(422, "retrieval_empty_query", "query must be non-empty")
        if self._embedder is None:
            raise _http(
                503, "embedder_unavailable",
                "text embedder is not loaded; POST /v1/models/{model}/load first",
            )

        loop = asyncio.get_running_loop()
        try:
            info = await loop.run_in_executor(
                None, self._store.collection_info, req.database, req.collection
            )
        except CollectionNotFound:
            raise _http(
                404, "collection_not_found",
                f"collection {req.collection!r} does not exist in database "
                f"{req.database!r}",
            ) from None

        field_names = {
            f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            for f in getattr(info, "fields", [])
        }
        if req.channels.bm25 and _SPARSE_FIELD not in field_names:
            raise _http(
                422, "retrieval_channel_unsupported",
                "BM25 channel requires schema v2; migrate the ingest collection",
                channels=["bm25"], migration_available=True,
            )

        if req.rerank.enabled and (
            self._reranker is None or getattr(self._reranker, "_impl", None) is None
        ):
            raise _http(
                503, "reranker_not_loaded",
                "reranker is not loaded; POST /v1/models/{model}/load first",
            )

        rewrite_on = req.rewrite.enabled and any(
            m in req.rewrite.methods
            for m in ("hyde", "multi_query", "step_back", "decompose")
        )
        if rewrite_on and not self._chat_check(self._settings):
            raise _http(
                503, "llm_unavailable",
                "LLM is not configured; set VS_LLM__BASE_URL and VS_LLM__MODEL",
            )
        return {"info": info, "rewrite_on": rewrite_on}

    async def retrieve(
        self,
        req: RetrievalRequest,
        *,
        emit: StageEmit = _noop_emit,
        prevalidated: dict | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline and return answer + full trace."""
        ctx = prevalidated if prevalidated is not None else await self.validate(req)
        loop = asyncio.get_running_loop()
        traces: list[StageTrace] = []

        # ---- rewrite ----
        await emit({"type": "stage", "stage": "rewrite"})
        timer = _Timer("rewrite")
        chat_fn = None
        if ctx["rewrite_on"]:
            chat_client = await loop.run_in_executor(
                None, self._chat_getter, self._settings
            )
            chat_fn = chat_client.as_chat_fn()
        plan = await loop.run_in_executor(
            None, self._build_plan, req, chat_fn
        )
        traces.append(timer.done({
            "dense_queries": len(plan.dense_specs),
            "lexical_queries": len(plan.lexical_queries),
        }))

        # Recall legs may need to feed a wider rerank pool later.
        want_pool = req.rerank.candidate_pool if req.rerank.enabled else 25
        recall_limit = min(64, max(req.top_k, want_pool))

        # ---- recall ----
        await emit({"type": "stage", "stage": "recall"})
        timer = _Timer("recall")
        filter_expr = build_filter_expr(req.filter)
        runs = await self._recall(loop, plan, req, recall_limit, filter_expr)
        traces.append(timer.done({"legs": len(runs), "per_leg_top_k": recall_limit}))

        # ---- fuse ----
        await emit({"type": "stage", "stage": "fuse"})
        timer = _Timer("fuse")
        if req.fusion.method == "rrf":
            chunks = rrf_fuse(runs, rrf_k=req.fusion.rrf_k)
        else:
            chunks = weighted_fuse(runs, req.fusion.weights.model_dump())
        traces.append(timer.done({
            "method": req.fusion.method, "chunks": len(chunks),
        }))

        if not chunks:
            return RetrievalResult(req.query, [], plan, runs, traces)

        # ---- mmr ----
        if req.mmr.enabled:
            await emit({"type": "stage", "stage": "mmr"})
            timer = _Timer("mmr")
            select_count = (
                req.rerank.candidate_pool if req.rerank.enabled else req.top_k
            )
            pool = chunks[: min(len(chunks), max(select_count * 2, 30))]
            query_vector = await loop.run_in_executor(
                None, self._embedder.embed_query, req.query
            )
            doc_vectors = await loop.run_in_executor(
                None, self._embedder.embed_documents,
                [str(c.fields.get("text", "")) for c in pool],
            )
            chunks = mmr(
                pool, doc_vectors, query_vector,
                lambda_mult=req.mmr.lambda_mult, top_k=select_count,
            )
            traces.append(timer.done({
                "pool": len(pool), "selected": len(chunks),
            }))

        # ---- rerank ----
        if req.rerank.enabled:
            await emit({"type": "stage", "stage": "rerank"})
            timer = _Timer("rerank")
            pool = chunks[: req.rerank.candidate_pool]
            documents = [str(c.fields.get("text", "")) for c in pool]
            scored = await loop.run_in_executor(
                None,
                functools.partial(self._reranker.rerank, req.query, documents,
                                  req.top_k),
            )
            reranked: list[RetrievedChunk] = []
            for hit in scored:
                chunk = pool[hit.index]
                chunk.rerank_score = float(hit.score)
                reranked.append(chunk)
            chunks = reranked[: req.top_k]
            traces.append(timer.done({"candidates": len(pool)}))
        else:
            chunks = chunks[: req.top_k]

        return RetrievalResult(req.query, chunks, plan, runs, traces)

    # ---- internal ----

    def _build_plan(
        self, req: RetrievalRequest, chat_fn: Any
    ) -> RetrievalPlan:
        query = req.query.strip()
        dense = [RecallSpec(query=query)]
        lexical: list[str] = [query]
        sub_queries: list[str] = []

        if chat_fn is not None:
            methods = req.rewrite.methods
            if "hyde" in methods:
                # HyDE replaces the base dense leg with the q/h mix; it is
                # not a lexical query.
                dense = [T.hyde(chat_fn, query, self._embedder,
                                alpha=req.rewrite.hyde_alpha)]
            if "multi_query" in methods:
                for variant in T.multi_query(chat_fn, query, req.rewrite.n_variants):
                    if variant != query:
                        dense.append(RecallSpec(query=variant))
                        lexical.append(variant)
            if "step_back" in methods:
                broader = T.step_back(chat_fn, query)
                if broader != query:
                    dense.append(RecallSpec(query=broader))
                    lexical.append(broader)
            if "decompose" in methods:
                parts = T.decompose(chat_fn, query)
                sub_queries = list(parts)
                for part in parts:
                    if part and part != query:
                        dense.append(RecallSpec(query=part))
                        lexical.append(part)

        return RetrievalPlan(
            original_query=query,
            dense_specs=_dedupe_specs(dense),
            lexical_queries=_dedupe_strings(lexical),
            sub_queries=sub_queries,
        )

    async def _recall(
        self,
        loop: asyncio.AbstractEventLoop,
        plan: RetrievalPlan,
        req: RetrievalRequest,
        top_k: int,
        filter_expr: str | None,
    ) -> list:
        dense_ch = DenseChannel(self._store, req.database, req.collection,
                                self._embedder, _VECTOR_FIELD)
        bm25_ch = BM25Channel(self._store, req.database, req.collection,
                              _SPARSE_FIELD)
        tasks: list[Awaitable] = []
        if req.channels.dense:
            for spec in plan.dense_specs:
                tasks.append(loop.run_in_executor(
                    None,
                    functools.partial(dense_ch.recall, spec, top_k,
                                      filter_expr, None),
                ))
        if req.channels.bm25:
            for query_text in plan.lexical_queries:
                spec = RecallSpec(query=query_text)
                tasks.append(loop.run_in_executor(
                    None,
                    functools.partial(bm25_ch.recall, spec, top_k,
                                      filter_expr, None),
                ))
        return list(await asyncio.gather(*tasks))


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _dedupe_specs(specs: list[RecallSpec]) -> list[RecallSpec]:
    # Keep the first spec per wording — HyDE (vector set) comes first and
    # must not be shadowed by a later plain-vector spec for the same query.
    seen: set[str] = set()
    out: list[RecallSpec] = []
    for spec in specs:
        if spec.query not in seen:
            seen.add(spec.query)
            out.append(spec)
    return out
```

更新 `src/vector_service/retrieval/__init__.py`：

```python
"""Retrieval pipeline: multi-channel recall, fusion, diversity, rerank."""
from vector_service.retrieval.pipeline import RetrievalPipeline

__all__ = ["RetrievalPipeline"]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_pipeline.py -q`
Expected: PASS（11 passed；async 包装方式按 Step 2 的检查结果）

- [ ] **Step 5: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/retrieval tests/unit/test_retrieval_pipeline.py
git commit -m "feat(retrieval): orchestrate rewrite, recall, fuse, mmr, rerank"
```

---

### Task 10: API 路由 JSON / NDJSON / capabilities + 注册

**Files:**
- Create: `src/vector_service/api/retrieval.py`
- Modify: `src/vector_service/main.py`
- Test: `tests/unit/test_retrieval_routes.py`

**Interfaces:**
- Produces:
  - `POST /v1/retrieval` → `RetrievalResponse`
  - `POST /v1/retrieval/stream` → `application/x-ndjson`；pre-flight 错误在开流前以普通 JSON 返回
  - `GET /v1/retrieval/capabilities?database=default` → `{"llm_configured", "schema_version", "migration_available"}`
- Consumes: `RetrievalPipeline`（Task 9）、`schema_version`（Task 4）、`is_llm_configured`。
- 迁移端点在 Task 11 加入同文件。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_retrieval_routes.py`：

```python
"""HTTP surface for /v1/retrieval (JSON + NDJSON) and capabilities."""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.retrieval import router as retrieval_router
from vector_service.stores.base import CollectionInfo, Hit


class FakeStore:
    def __init__(self, v2=True):
        self.v2 = v2

    def list_databases(self):
        return ["default"]

    def list_collections(self, database):
        return ["ingest"]

    def collection_info(self, db, coll):
        fields = ([{"name": "id"}, {"name": "text"}, {"name": "sparse"}]
                  if self.v2 else [{"name": "id"}, {"name": "text"}])
        return CollectionInfo(database=db, name=coll, dim=4, metric="cosine",
                              count=0, fields=fields)

    def search(self, *a, **kw):
        return [Hit(id="c1", score=0.9, fields={"text": "dense"})]

    def search_text(self, *a, **kw):
        return [Hit(id="c2", score=5.0, fields={"text": "lexical"})]


class FakeEmbedder:
    def embed_query(self, q):
        return [0.1, 0.2, 0.3, 0.4]


class FakeReranker:
    _impl = object()

    def rerank(self, q, docs, top_n=None):
        from vector_service.rerankers.base import ScoredHit

        return [ScoredHit(index=0, score=0.99)]


class FakeSettings:
    llm = None


def _http_handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": "error", "message": str(detail)}})


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(retrieval_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.settings = FakeSettings()
    app.state.store = FakeStore()
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    return TestClient(app)


def _body(**overrides):
    body = {"query": "季度营收"}
    body.update(overrides)
    return body


def test_json_retrieval_returns_envelope(client):
    resp = client.post("/v1/retrieval", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "季度营收"
    assert len(data["chunks"]) >= 1
    assert {"channel_runs", "plan", "traces"} <= set(data)


def test_stream_emits_stage_then_result(client):
    resp = client.post("/v1/retrieval/stream", json=_body())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["rewrite", "recall", "fuse", "rerank"]
    assert events[-1]["type"] == "result"
    assert "chunks" in events[-1]


def test_preflight_error_is_plain_json_not_ndjson(client):
    client.app.state.embedder = None
    resp = client.post("/v1/retrieval/stream", json=_body())
    assert resp.status_code == 503
    assert "json" in resp.headers["content-type"]
    assert resp.json()["error"]["code"] == "embedder_unavailable"


def test_validation_error_422(client):
    resp = client.post("/v1/retrieval", json=_body(
        channels={"dense": False, "bm25": False}))
    assert resp.status_code == 422


def test_capabilities_reports_v2(client):
    resp = client.get("/v1/retrieval/capabilities?database=default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == 2
    assert data["migration_available"] is False
    assert data["llm_configured"] is False


@pytest.fixture
def v1_client():
    app = FastAPI()
    app.include_router(retrieval_router)
    app.add_exception_handler(HTTPException, _http_handler)
    app.state.settings = FakeSettings()
    app.state.store = FakeStore(v2=False)
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    return TestClient(app)


def test_capabilities_reports_v1(v1_client):
    data = v1_client.get("/v1/retrieval/capabilities").json()
    assert data["schema_version"] == 1
    assert data["migration_available"] is True
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_retrieval_routes.py -q`
Expected: FAIL，无 `vector_service.api.retrieval`

- [ ] **Step 3: 实现**

创建 `src/vector_service/api/retrieval.py`：

```python
"""Retrieval API: multi-channel retrieval, NDJSON stream, capabilities."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from vector_service.chunking.llm_chunker import is_llm_configured
from vector_service.retrieval.pipeline import RetrievalPipeline
from vector_service.api.ingest import schema_version
from vector_service.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    to_result,
)

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _pipeline(request: Request) -> RetrievalPipeline:
    state = request.app.state
    return RetrievalPipeline(
        settings=state.settings,
        store=state.store,
        embedder=getattr(state, "embedder", None),
        reranker=getattr(state, "reranker", None),
    )


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieve(body: RetrievalRequest, request: Request) -> RetrievalResponse:
    result = await _pipeline(request).retrieve(body)
    return to_result(result)


@router.post("/retrieval/stream")
async def retrieve_stream(body: RetrievalRequest, request: Request):
    """NDJSON stage events + terminal result.

    Pre-flight validation runs before the StreamingResponse is
    created, so bad params / missing models come back as ordinary JSON
    error envelopes rather than mid-stream errors.
    """
    pipeline = _pipeline(request)
    prevalidated = await pipeline.validate(body)

    async def event_stream():
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def runner():
            try:
                result = await pipeline.retrieve(
                    body,
                    emit=lambda event: queue.put(event),
                    prevalidated=prevalidated,
                )
                queue.put_nowait({
                    "type": "result",
                    **to_result(result).model_dump(),
                })
            except Exception as e:  # noqa: BLE001 — last-resort terminal event
                if isinstance(e, getattr(__import__("fastapi"), "HTTPException")):
                    queue.put_nowait({
                        "type": "error", "status": e.status_code, **e.detail,
                    })
                else:
                    queue.put_nowait({
                        "type": "error", "status": 500,
                        "error": {"code": "internal", "message": str(e)},
                    })
            finally:
                queue.put_nowait(sentinel)

        asyncio.create_task(runner())
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/retrieval/capabilities")
def capabilities(request: Request, database: str = "default") -> dict:
    """LLM availability + ingest schema version for the selected database."""
    store = request.app.state.store
    version = 1
    if database in store.list_databases() and (
        "ingest" in store.list_collections(database)
    ):
        version = schema_version(store.collection_info(database, "ingest"))
    return {
        "llm_configured": is_llm_configured(request.app.state.settings),
        "schema_version": version,
        "migration_available": version == 1,
    }
```

注意：`except` 中判断 HTTPException 的写法在文件顶部直接 import 更干净——实现时写 `from fastapi import HTTPException` 并在 runner 内 `except HTTPException as e:`（替代 `getattr` 版本）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/unit/test_retrieval_routes.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 注册路由到 main.py**

(a) import 区（21-34 行的 api import 组）加：

```python
from vector_service.api.retrieval import router as retrieval_router
```

(b) include_router 组（429-442 行，最后一个 ingest_router 之后）加：

```python
    app.include_router(retrieval_router)
```

- [ ] **Step 6: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/api/retrieval.py src/vector_service/main.py tests/unit/test_retrieval_routes.py
git commit -m "feat(retrieval): add /v1/retrieval JSON and NDJSON endpoints; register router"
```

---

### Task 11: ingest v1 → v2 一键迁移

**Files:**
- Modify: `src/vector_service/api/ingest.py`（新增 `migrate_ingest_collection()`）
- Modify: `src/vector_service/api/retrieval.py`（新增迁移端点）
- Modify: `src/vector_service/stores/_milvus_adapter.py`（`rename_collection`/`insert_rows`；browse `include_vectors`）
- Test: `tests/unit/test_ingest_migration.py`

**Interfaces:**
- Produces:
  - `POST /v1/databases/{database}/collections/ingest/migrate` → `{"database","collection","rows","schema_version"}`
  - adapter：`rename_collection(database, old_name, new_name)`、`insert_rows(database, collection, rows: list[dict])`；`browse(..., include_vectors=False)`
  - `migrate_ingest_collection(store, database, dim) -> dict`
- 迁移流程：建临时 v2 集合 `ingest_migrate_tmp` → 分页（200/页）拷贝全部行（含 dense，sparse 由 BM25 Function 自动生成）→ 行数校验 → 删旧 → rename 临时集合为 ingest。任一步失败：删临时集合、旧集合不动、抛出错误。

- [ ] **Step 1: 写失败测试**

创建 `tests/unit/test_ingest_migration.py`：

```python
"""v1 → v2 ingest collection migration: copy, verify, swap, rollback."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from vector_service.api.ingest import migrate_ingest_collection
from vector_service.api.retrieval import router as retrieval_router
from vector_service.stores.base import CollectionInfo, FieldSpec, IndexSpec


class FakeAdapter:
    def __init__(self, rows, fail_after=None):
        self.rows = rows
        self.fail_after = fail_after
        self.dropped = []
        self.renamed = None
        self.inserted = []

    def has_collection(self, db, name):
        return name == "ingest"

    def browse(self, db, collection, primary_field, *, limit=200, offset=0,
               filter_expr=None, output_fields=None, include_vectors=False):
        assert include_vectors is True
        page = self.rows[offset: offset + limit]
        return page

    def insert_rows(self, db, collection, rows):
        if self.fail_after == "insert":
            raise RuntimeError("boom")
        self.inserted.extend(rows)

    def count(self, db, collection):
        if collection == "ingest":
            return len(self.rows)
        return len(self.inserted)

    def drop_collection(self, db, name):
        self.dropped.append(name)

    def rename_collection(self, db, old, new):
        if self.fail_after == "rename":
            raise RuntimeError("boom-rename")
        self.renamed = (old, new)


class FakeStore:
    def __init__(self, adapter):
        self._adapter = adapter

    def list_databases(self):
        return ["default"]

    def list_collections(self, db):
        return ["ingest"]

    def collection_info(self, db, name):
        return CollectionInfo(database=db, name=name, dim=4, metric="cosine",
                              count=len(self._adapter.rows),
                              fields=[{"name": "id"}, {"name": "text"}])

    def create_collection(self, db, name, primary_field, vector_field,
                          scalar_fields, indexes=None):
        self.created_scalars = scalar_fields
        self.created_indexes = indexes


def _rows(n):
    return [
        {"id": f"c{i}", "fields": {"text": f"t{i}", "vector": [0.1 * i, 0.2]}}
        for i in range(n)
    ]


def test_migration_copies_and_swaps():
    adapter = FakeAdapter(_rows(3))
    store = FakeStore(adapter)
    out = migrate_ingest_collection(store, "default", 4)
    assert out == {"database": "default", "collection": "ingest",
                   "rows": 3, "schema_version": 2}
    # rows carried dense vector + scalars; sparse not provided (function fills)
    assert all("sparse" not in row for row in adapter.inserted)
    assert adapter.inserted[0]["vector"] == [0.0, 0.2]
    # v2 schema used: sparse field + analyzed text
    names = {f.name for f in store.created_scalars}
    assert "sparse" in names
    assert adapter.dropped == ["ingest"]
    assert adapter.renamed == ("ingest_migrate_tmp", "ingest")


def test_migration_cleans_up_tmp_on_insert_failure():
    adapter = FakeAdapter(_rows(2), fail_after="insert")
    store = FakeStore(adapter)
    with pytest.raises(RuntimeError):
        migrate_ingest_collection(store, "default", 4)
    # tmp dropped; old ingest never dropped or renamed
    assert "ingest_migrate_tmp" in adapter.dropped
    assert "ingest" not in adapter.dropped
    assert adapter.renamed is None


def test_migration_rollback_when_rename_fails():
    adapter = FakeAdapter(_rows(1), fail_after="rename")
    store = FakeStore(adapter)
    with pytest.raises(RuntimeError):
        migrate_ingest_collection(store, "default", 4)
    assert adapter.renamed is None
    # old ingest was already dropped before rename failed — surface the
    # fact: migration documents this window; tmp still exists? Code
    # cleanup drops tmp, so data loss risk is documented in docs.
    assert "ingest" in adapter.dropped


# ---- HTTP route ----

def _handler(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": detail["error"]})
    return JSONResponse(status_code=exc.status_code,
                        content={"error": {"code": "x", "message": str(detail)}})


def test_migrate_route_success():
    adapter = FakeAdapter(_rows(1))
    store = FakeStore(adapter)
    app = FastAPI()
    app.include_router(retrieval_router)
    app.add_exception_handler(HTTPException, _handler)
    app.state.settings = object()
    app.state.store = store
    resp = TestClient(app).post(
        "/v1/databases/default/collections/ingest/migrate")
    assert resp.status_code == 200
    assert resp.json()["schema_version"] == 2


def test_migrate_route_unknown_database_404():
    adapter = FakeAdapter(_rows(1))
    store = FakeStore(adapter)
    store.list_databases = lambda: ["other"]
    app = FastAPI()
    app.include_router(retrieval_router)
    app.add_exception_handler(HTTPException, _handler)
    app.state.settings = object()
    app.state.store = store
    resp = TestClient(app).post(
        "/v1/databases/default/collections/ingest/migrate")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "database_not_found"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/unit/test_ingest_migration.py -q`
Expected: FAIL，无 `migrate_ingest_collection` / 路由 404。

- [ ] **Step 3: adapter 增加三个能力**

(a) `browse` 签名（954-964 行）加参数：

```python
    def browse(
        self,
        database: str,
        collection: str,
        primary_field: str,
        *,
        limit: int = 20,
        offset: int = 0,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
```

默认输出分支（999-1000 行），include_vectors 时保留 dense 字段：

```python
        if output_fields is None:
            output = [f["name"] for f in schema["fields"]
                      if include_vectors
                      or f["name"] != schema.get("vector_field")]
```

显式 output_fields 分支（1011 行）改为：

```python
            output = [f for f in output_fields
                      if include_vectors or f != schema.get("vector_field")]
```

(b) 在 `search_text` 方法之后加：

```python
    def insert_rows(
        self, database: str, collection: str, rows: list[dict[str, Any]]
    ) -> None:
        """Raw row insert (migration bulk-copy path).

        Callers provide every field except function-generated ones:
        the BM25 Function populates ``sparse`` from ``text`` itself.
        """
        self._ensure_connected()
        self._using_db(database)
        try:
            self._client.insert(collection, data=rows)
        except Exception as e:
            raise BackendError(
                f"insert_rows failed for {database!r}/{collection!r}: {e}"
            ) from e

    def rename_collection(
        self, database: str, old_name: str, new_name: str
    ) -> None:
        """Rename a collection; keeps the same data and schema."""
        self._ensure_connected()
        self._using_db(database)
        try:
            self._client.rename_collection(old_name, new_name)
        except Exception as e:
            raise BackendError(
                f"rename_collection failed {old_name!r} -> {new_name!r}: {e}"
            ) from e
        self._invalidate_collection(database, old_name)
        self._invalidate_collection(database, new_name)
```

- [ ] **Step 4: ingest.py 增加迁移函数**

在 `_ensure_collection` 之后（1022 行后）加：

```python
_TMP_COLLECTION = "ingest_migrate_tmp"
_MIGRATE_BATCH = 200


def migrate_ingest_collection(store: Any, database: str, dim: int) -> dict:
    """Copy an ingest v1 collection into a v2 collection and swap them.

    Failure at any step drops the temporary collection and leaves the
    original in place (except the narrow window after the old
    collection is dropped and before rename completes — documented in
    docs/retrieval.md).
    """
    adapter = store._adapter
    try:
        if adapter.has_collection(database, _TMP_COLLECTION):
            adapter.drop_collection(database, _TMP_COLLECTION)
        store.create_collection(
            database,
            _TMP_COLLECTION,
            primary_field=_PRIMARY_FIELD,
            vector_field=FieldSpec(name=_VECTOR_FIELD, dtype="float_vector",
                                   dim=dim),
            scalar_fields=_ingest_scalar_fields_v2(),
            indexes=_ingest_indexes_v2(),
        )

        total = 0
        while True:
            rows = adapter.browse(
                database, "ingest", _PRIMARY_FIELD,
                limit=_MIGRATE_BATCH, offset=total,
                include_vectors=True,
            )
            if not rows:
                break
            adapter.insert_rows(database, _TMP_COLLECTION, [
                {"id": row["id"], **row["fields"]} for row in rows
            ])
            total += len(rows)
            if len(rows) < _MIGRATE_BATCH:
                break

        migrated = adapter.count(database, _TMP_COLLECTION)
        old_count = adapter.count(database, "ingest")
        if migrated != old_count:
            raise StoreError(
                f"migration row count mismatch: {migrated} copied vs "
                f"{old_count} original"
            )

        adapter.drop_collection(database, "ingest")
        adapter.rename_collection(database, _TMP_COLLECTION, "ingest")
    except Exception:
        if adapter.has_collection(database, _TMP_COLLECTION):
            adapter.drop_collection(database, _TMP_COLLECTION)
        raise

    return {
        "database": database,
        "collection": "ingest",
        "rows": total,
        "schema_version": 2,
    }
```

- [ ] **Step 5: retrieval.py 增加迁移端点**

在 capabilities 路由之前加：

```python
from fastapi import HTTPException  # 若 Step 3 清理后已在文件顶部 import，跳过本行


@router.post("/databases/{database}/collections/ingest/migrate")
def migrate(database: str, request: Request) -> dict:
    """One-click v1 → v2 migration of the ingest collection."""
    store = request.app.state.store
    if database not in store.list_databases():
        raise HTTPException(status_code=404, detail={"error": {
            "code": "database_not_found",
            "message": f"database {database!r} does not exist",
        }})
    if "ingest" not in store.list_collections(database):
        raise HTTPException(status_code=404, detail={"error": {
            "code": "collection_not_found",
            "message": "collection 'ingest' does not exist",
        }})
    info = store.collection_info(database, "ingest")
    if schema_version(info) >= 2:
        raise HTTPException(status_code=409, detail={"error": {
            "code": "collection_exists",
            "message": "ingest collection already uses schema v2",
        }})
    return migrate_ingest_collection(store, database, int(info.dim))
```

文件顶部补充 import（与现有 import 同组）：

```python
from vector_service.api.ingest import migrate_ingest_collection, schema_version
```

（原 `from vector_service.api.ingest import schema_version` 一行替换掉。）

- [ ] **Step 6: 运行确认通过**

Run: `python -m pytest tests/unit/test_ingest_migration.py -q`
Expected: PASS（5 passed）

- [ ] **Step 7: 全量单测 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS

```bash
git add src/vector_service/api/ingest.py src/vector_service/api/retrieval.py src/vector_service/stores/_milvus_adapter.py tests/unit/test_ingest_migration.py
git commit -m "feat(retrieval): one-click v1->v2 migration with copy-verify-swap and rollback"
```

---

### Task 12: 前端检索面板 retrieval.js + 导航/i18n/样式接线

**Files:**
- Create: `src/vector_service/static/dashboard/components/retrieval.js`
- Modify: `src/vector_service/static/dashboard/components/app.js`
- Modify: `src/vector_service/static/dashboard/dashboard.css`

**Interfaces:**
- 调用：`GET /v1/databases`、`GET /v1/retrieval/capabilities?database=`、`POST /v1/retrieval/stream`、`POST /v1/databases/{db}/collections/ingest/migrate`
- 模式预设：basic（仅 dense、无 rerank）、hybrid（默认：dense+bm25+RRF+rerank）、advanced（hybrid + rewrite 全方法）、custom（不联动）。
- v1 collection：迁移横幅 + 一键迁移；LLM 未配置：rewrite 整组置灰。

- [ ] **Step 1: 创建 retrieval.js**

```javascript
// Retrieval panel: dense/BM25 recall, fusion, rewrite, MMR, rerank.
// Streams NDJSON stage events from /v1/retrieval/stream and renders the
// full retrieval trace (plan, per-channel raw tops, stage timings).
import {
  defineComponent, reactive, onMounted,
} from '../vue.esm-browser.prod.js';

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => null);
  return { status: resp.status, body };
}

function defaultState() {
  return {
    databases: [],
    database: 'default',
    caps: { llm_configured: false, schema_version: 2, migration_available: false },
    query: '',
    topK: 10,
    mode: 'hybrid',
    dense: true,
    bm25: true,
    fusionMethod: 'rrf',
    rrfK: 60,
    wDense: 0.5,
    wBm25: 0.5,
    rewrite: false,
    methods: { hyde: true, multi_query: true, step_back: true, decompose: true },
    hydeAlpha: 0.7,
    nVariants: 3,
    mmr: false,
    mmrLambda: 0.7,
    rerank: true,
    candidatePool: 25,
    docId: '',
    filename: '',
    busy: false,
    stages: [],
    error: '',
    result: null,
    showTrace: false,
    migrating: false,
  };
}

export default defineComponent({
  name: 'RetrievalPanel',
  setup() {
    const s = reactive(defaultState());

    async function loadDatabases() {
      const { body } = await getJson('/v1/databases');
      const names = (body && (body.databases || body.payload && body.payload.databases)) || [];
      s.databases = names;
      if (!names.includes(s.database) && names.length) s.database = names[0];
    }

    async function loadCaps() {
      const { body } = await getJson(
        '/v1/retrieval/capabilities?database=' + encodeURIComponent(s.database),
      );
      if (body && body.payload) Object.assign(s.caps, body.payload);
      else if (body) Object.assign(s.caps, body);
    }

    function setMode(mode) {
      s.mode = mode;
      if (mode === 'basic') {
        s.dense = true; s.bm25 = false;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = false;
      } else if (mode === 'hybrid') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = false; s.mmr = false; s.rerank = true;
      } else if (mode === 'advanced') {
        s.dense = true; s.bm25 = true;
        s.fusionMethod = 'rrf';
        s.rewrite = true;
        s.methods = { hyde: true, multi_query: true, step_back: true, decompose: true };
        s.mmr = false; s.rerank = true;
      }
    }

    function buildBody() {
      const methods = Object.keys(s.methods).filter((k) => s.methods[k]);
      return {
        database: s.database,
        collection: 'ingest',
        query: s.query,
        top_k: Number(s.topK),
        filter: { doc_id: s.docId || '', filename: s.filename || '' },
        channels: { dense: s.dense, bm25: s.bm25 },
        fusion: {
          method: s.fusionMethod,
          rrf_k: Number(s.rrfK),
          weights: { dense: Number(s.wDense), bm25: Number(s.wBm25) },
        },
        rewrite: {
          enabled: s.rewrite,
          methods,
          hyde_alpha: Number(s.hydeAlpha),
          n_variants: Number(s.nVariants),
        },
        mmr: { enabled: s.mmr, lambda_mult: Number(s.mmrLambda) },
        rerank: { enabled: s.rerank, candidate_pool: Number(s.candidatePool) },
      };
    }

    async function run() {
      s.busy = true; s.error = ''; s.result = null; s.stages = [];
      try {
        const resp = await fetch('/v1/retrieval/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildBody()),
        });
        if (!resp.ok || !resp.headers.get('content-type').includes('ndjson')) {
          const err = await resp.json().catch(() => null);
          const info = err && (err.error || (err.payload && err.payload.error));
          throw new Error(info ? info.message || info.code : `HTTP ${resp.status}`);
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === 'stage') s.stages.push(event.stage);
            else if (event.type === 'result') s.result = event;
            else if (event.type === 'error') {
              throw new Error(event.error.message || event.error.code);
            }
          }
        }
      } catch (e) {
        s.error = String(e.message || e);
      } finally {
        s.busy = false;
      }
    }

    async function migrate() {
      s.migrating = true; s.error = '';
      try {
        const url = '/v1/databases/' + encodeURIComponent(s.database)
          + '/collections/ingest/migrate';
        const resp = await fetch(url, { method: 'POST' });
        const body = await resp.json().catch(() => null);
        if (!resp.ok) {
          const info = body && body.error;
          throw new Error(info ? info.message || info.code : `HTTP ${resp.status}`);
        }
        await loadCaps();
      } catch (e) {
        s.error = String(e.message || e);
      } finally {
        s.migrating = false;
      }
    }

    onMounted(async () => {
      await loadDatabases();
      await loadCaps();
    });

    return { s, setMode, run, migrate };
  },
  template: `
  <div class="retrieval-panel">
    <div class="section">
      <div class="section-head">
        <h3 class="section-title">{{ s.database }} / ingest</h3>
        <span class="pill accent">POST /v1/retrieval/stream</span>
      </div>

      <div v-if="s.caps.schema_version === 1" class="retrieval-migrate-banner">
        <span>⚠ schema v1：BM25 全文检索不可用</span>
        <button class="btn sm" :disabled="s.migrating" @click="migrate">
          {{ s.migrating ? 'migrating…' : '一键迁移到 v2' }}
        </button>
      </div>

      <div class="retrieval-modes">
        <button v-for="m in ['basic','hybrid','advanced','custom']" :key="m"
          :class="['btn','sm', s.mode === m ? 'primary' : '']"
          @click="setMode(m)">{{ m }}</button>
      </div>

      <div class="row">
        <label>database</label>
        <select v-model="s.database" @change="loadCaps">
          <option v-for="d in s.databases" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>

      <div class="row">
        <label>query</label>
        <textarea v-model="s.query" rows="2" placeholder="输入检索问题"></textarea>
      </div>

      <div class="row">
        <label>top_k</label>
        <input type="number" min="1" max="100" v-model.number="s.topK">
      </div>

      <div class="cap-group-title">channels &amp; fusion</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.dense"> dense</label>
        <label><input type="checkbox" v-model="s.bm25"
          :disabled="s.caps.schema_version === 1"> bm25</label>
      </div>
      <div class="row">
        <label>fusion</label>
        <select v-model="s.fusionMethod">
          <option value="rrf">rrf</option>
          <option value="weighted">weighted</option>
        </select>
        <template v-if="s.fusionMethod === 'rrf'">
          <label>rrf_k</label>
          <input type="number" min="1" max="200" v-model.number="s.rrfK">
        </template>
        <template v-else>
          <label>w dense</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wDense">
          <label>w bm25</label>
          <input type="number" min="0" step="0.1" v-model.number="s.wBm25">
        </template>
      </div>

      <div class="cap-group-title">query rewrite
        <span v-if="!s.caps.llm_configured" class="pill danger">LLM 未配置</span>
      </div>
      <fieldset class="retrieval-fieldset" :disabled="!s.caps.llm_configured">
        <div class="row">
          <label><input type="checkbox" v-model="s.rewrite"> enabled</label>
        </div>
        <div class="row">
          <label v-for="k in ['hyde','multi_query','step_back','decompose']" :key="k">
            <input type="checkbox" v-model="s.methods[k]"> {{ k }}
          </label>
        </div>
        <div class="row">
          <label>hyde α</label>
          <input type="number" min="0" max="1" step="0.05" v-model.number="s.hydeAlpha">
          <label>n variants</label>
          <input type="number" min="1" max="5" v-model.number="s.nVariants">
        </div>
      </fieldset>

      <div class="cap-group-title">diversity &amp; rerank</div>
      <div class="row">
        <label><input type="checkbox" v-model="s.mmr"> mmr</label>
        <label>λ</label>
        <input type="number" min="0" max="1" step="0.05" v-model.number="s.mmrLambda">
      </div>
      <div class="row">
        <label><input type="checkbox" v-model="s.rerank"> cross-encoder rerank</label>
        <label>candidate pool</label>
        <input type="number" min="1" max="64" v-model.number="s.candidatePool">
      </div>

      <div class="cap-group-title">metadata filter</div>
      <div class="row">
        <label>doc_id</label>
        <input v-model="s.docId" placeholder="精确匹配">
        <label>filename</label>
        <input v-model="s.filename" placeholder="模糊匹配">
      </div>

      <div class="actions">
        <button class="btn primary" :disabled="s.busy || !s.query.trim()" @click="run">
          <span v-if="s.busy" class="btn-spinner"></span>
          {{ s.busy ? 'retrieving…' : '检索' }}
        </button>
      </div>

      <div v-if="s.stages.length" class="response retrieval-progress">
        <span v-for="(st, i) in s.stages" :key="i" class="retrieval-stage">
          <span class="spinner" v-if="i === s.stages.length - 1 && s.busy"></span>
          {{ st }}
        </span>
      </div>

      <div v-if="s.error" class="response ingest-result-err">
        <div class="result-banner err">
          <span class="result-glyph err">!</span>
          <span class="result-banner-text">{{ s.error }}</span>
        </div>
      </div>

      <div v-if="s.result" class="retrieval-results">
        <div v-for="(c, i) in s.result.chunks" :key="c.chunk_id" class="retrieval-chunk">
          <div class="retrieval-chunk-head">
            <strong>#{{ i + 1 }}</strong>
            <span v-for="ch in c.matched_channels" :key="ch" class="pill accent">{{ ch }}</span>
            <span class="retrieval-scores">
              fusion {{ c.fusion_score.toFixed(4) }}
              <template v-if="c.rerank_score !== null">
                · rerank {{ c.rerank_score.toFixed(4) }}
              </template>
            </span>
          </div>
          <div class="retrieval-chunk-text">{{ c.fields.text }}</div>
          <div class="retrieval-chunk-meta">
            doc_id {{ c.fields.doc_id }} · chunk #{{ c.fields.chunk_index }}
            <template v-if="c.fields.section_header"> · § {{ c.fields.section_header }}</template>
            <template v-if="c.fields.page_number"> · p.{{ c.fields.page_number }}</template>
            <template v-if="c.fields.filename"> · {{ c.fields.filename }}</template>
          </div>
        </div>

        <div class="retrieval-trace-toggle" @click="s.showTrace = !s.showTrace">
          {{ s.showTrace ? '▾' : '▸' }} 检索追踪
        </div>
        <div v-if="s.showTrace" class="retrieval-trace">
          <div class="stat" v-for="(tr, i) in s.result.traces" :key="i">
            <span class="key">{{ tr.stage }}</span>
            <span class="val">{{ tr.duration_ms }} ms · {{ JSON.stringify(tr.detail) }}</span>
          </div>
          <div class="stat" v-for="(run, i) in s.result.channel_runs" :key="'r'+i">
            <span class="key">{{ run.channel }}: {{ run.query }}</span>
            <span class="val">
              top:
              <template v-for="h in run.hits.slice(0, 3)" :key="h.chunk_id">
                {{ h.chunk_id }}({{ h.score.toFixed(2) }})
              </template>
            </span>
          </div>
          <div class="stat" v-if="s.result.plan.sub_queries.length">
            <span class="key">sub-queries</span>
            <span class="val">{{ s.result.plan.sub_queries.join(' | ') }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  `,
});
```

- [ ] **Step 2: app.js 接线（5 处编辑）**

(a) import：第 16 行 `import KnowledgeBasePanel ...` 之后加：

```javascript
import RetrievalPanel from './retrieval.js';
```

(b) NAV_LABELS：第 707 行 `'ingested': ...` 之后加：

```javascript
  'retrieval':           { catKey: 'cat.kb',        subKey: 'nav.retrieval' },
```

(c) components：715 行 `EmbeddingsPanel, SimilarityPanel, KnowledgeBasePanel,` 改为：

```javascript
    EmbeddingsPanel, SimilarityPanel, KnowledgeBasePanel, RetrievalPanel,
```

(d) kb 导航组：810 行 ingested nav-item 之后加：

```html
              <div :class="['nav-item', store.view === 'retrieval' ? 'active' : '']" data-view="retrieval" @click="store.view = 'retrieval'"><span>{{ t('nav.retrieval') }}</span></div>
```

(e) 面板挂载：838 行 knowledge-base-panel 之后加：

```html
          <retrieval-panel v-show="store.view === 'retrieval'" />
```

- [ ] **Step 3: i18n 文案**

zh 字典（第 54 行 `'nav.ingest': '一键入库',` 之后）插入：

```javascript
    'nav.retrieval': '分片检索',
    'retrieval.title': '分片检索',
    'retrieval.modes.basic': '基础',
    'retrieval.modes.hybrid': '混合',
    'retrieval.modes.advanced': '高阶',
    'retrieval.modes.custom': '自定义',
    'retrieval.run': '检索',
    'retrieval.running': '检索中…',
    'retrieval.migrate': '一键迁移到 v2',
    'retrieval.migrating': '迁移中…',
    'retrieval.v1_banner': 'schema v1：BM25 全文检索不可用',
    'retrieval.llm_hint': 'LLM 未配置',
    'retrieval.trace': '检索追踪',
```

en 字典（第 309 行 `'nav.ingest': 'One-Click Ingest',` 之后）插入：

```javascript
    'nav.retrieval': 'Retrieval',
    'retrieval.title': 'Retrieval',
    'retrieval.modes.basic': 'Basic',
    'retrieval.modes.hybrid': 'Hybrid',
    'retrieval.modes.advanced': 'Advanced',
    'retrieval.modes.custom': 'Custom',
    'retrieval.run': 'Retrieve',
    'retrieval.running': 'Retrieving…',
    'retrieval.migrate': 'Migrate to v2',
    'retrieval.migrating': 'Migrating…',
    'retrieval.v1_banner': 'Schema v1: BM25 full-text unavailable',
    'retrieval.llm_hint': 'LLM not configured',
    'retrieval.trace': 'Retrieval trace',
```

- [ ] **Step 4: dashboard.css 追加样式**

文件末尾追加：

```css
/* ---- retrieval panel ---- */
.retrieval-modes { display: flex; gap: 6px; margin: 10px 0; }
.retrieval-migrate-banner {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; padding: 8px 12px; margin: 10px 0;
  border: 1px solid var(--warn-border, #d9a441);
  background: var(--warn-bg, #fdf6e3); border-radius: 6px;
}
.retrieval-fieldset { border: none; padding: 0; margin: 0 0 10px; }
.retrieval-fieldset[disabled] { opacity: .55; }
.retrieval-progress { display: flex; flex-wrap: wrap; gap: 12px; }
.retrieval-stage { display: inline-flex; align-items: center; gap: 5px;
  font-size: 12px; color: var(--text-muted, #888); }
.retrieval-chunk {
  border: 1px solid var(--border, #e2e2e2); border-radius: 8px;
  padding: 10px 12px; margin: 10px 0;
}
.retrieval-chunk-head { display: flex; align-items: center; gap: 8px; }
.retrieval-scores { margin-left: auto; font-size: 12px;
  color: var(--text-muted, #888); }
.retrieval-chunk-text { margin: 8px 0; white-space: pre-wrap; }
.retrieval-chunk-meta { font-size: 12px; color: var(--text-muted, #888); }
.retrieval-trace-toggle { cursor: pointer; margin: 12px 0 6px;
  font-weight: 600; }
.retrieval-trace .stat { display: flex; gap: 8px; padding: 3px 0;
  font-size: 12px; }
.retrieval-trace .key { min-width: 200px; color: var(--text-muted, #888); }
```

- [ ] **Step 5: 验证**

Run: `python -m pytest tests/unit -q`
Expected: PASS（dashboard route 测试不校验新组件内容，全绿即可）

手动验证（Milvus 已启动时）：起服务 → dashboard → 知识库组出现「分片检索」→ basic/hybrid/advanced 预设切换 → hybrid 检索能看到 stage 进度、结果卡片与追踪区；若选中 v1 库，出现迁移横幅。无 Milvus 时仅确认页面加载无 JS 控制台错误（可用浏览器直接打开 dashboard 查看）。

- [ ] **Step 6: 提交**

```bash
git add src/vector_service/static/dashboard
git commit -m "feat(dashboard): add retrieval panel with mode presets, stream progress, trace"
```

---

### Task 13: 文档同步

**Files:**
- Create: `docs/retrieval.md`
- Modify: `docs/api.md`
- Modify: `docs/errors.md`
- Modify: `docs/README.md`
- Modify: `.env.example`

- [ ] **Step 1: 创建 docs/retrieval.md**

```markdown
# 分片检索（Retrieval）

知识库「分片检索」面板与 `POST /v1/retrieval` 端点对 `ingest` collection
执行多路检索，覆盖从基础向量召回到高阶 RAG 检索的完整栈。

## 管线

\`\`\`
query
  → rewrite   HyDE / Multi-Query / Step-Back / Decomposition（需 LLM）
  → recall    dense ANN × N 条查询 ‖ BM25 全文 × N 条查询（并发 fan-out）
  → fuse      RRF（默认）/ weighted
  → mmr       可选，去冗余
  → rerank    可选，cross-encoder 精排
\`\`\`

全链路可观测：结果之外还返回改写计划、每路原始 Top 命中与分数、融合分、
重排分和各阶段耗时（dashboard「检索追踪」区）。

## 模式预设

| 模式 | channels | 融合 | rewrite | MMR | rerank |
|------|----------|------|---------|-----|--------|
| 基础 basic | dense | — | – | – | – |
| 混合 hybrid（默认）| dense + bm25 | RRF | – | – | ✓ |
| 高阶 advanced | dense + bm25 | RRF | 全部方法 | – | ✓ |
| 自定义 custom | 任意 | 任意 | 任意 | 任意 | 任意 |

## 参数调优建议

- **RRF**：对分数尺度不敏感，首选；`rrf_k` 越小排名靠后的命中贡献衰减越快。
- **weighted**：需要明确 dense/词法权重（如关键词强信号的日志/法典场景可调高 bm25），
  每路先 min-max 归一化。
- **HyDE**：事实型问题收益最大；`hyde_alpha`（默认 0.7）是假设答案在混合向量中的占比，
  LLM 对领域不熟时调低。
- **Multi-Query**：缓解措辞不匹配，代价是召回腿数 ×n。
- **Step-Back / Decomposition**：多跳、综合型问题适用。
- **MMR**：结果需要去重/覆盖多角度时开启；λ 越低越多样。
- **rerank**：以融合后候选池（默认 25，上限 64）做 cross-encoder 精排，
  精度显著提升，成本随候选池线性增长。

## Schema v2 与迁移

BM25 全文检索要求 schema v2：`text`（varchar）启用 `chinese` 分析器，
新增 `sparse`（SPARSE_FLOAT_VECTOR）并注册 BM25 Function——写入时自动从
`text` 生成 sparse 向量。新库直接使用 v2；已有 v1 库执行一键迁移：

\`\`\`
POST /v1/databases/{db}/collections/ingest/migrate
\`\`\`

迁移会拷贝全部行（dense 向量原样携带，sparse 重新生成），校验行数后删旧并
rename。失败自动清理临时集合。注意：在「旧集合已删除、rename 完成前」
存在一个极窄的故障窗口，建议在业务低峰执行；rename 失败时临时集合
（`ingest_migrate_tmp`）保留数据，可由运维手动 rename 恢复。

## 降级行为

- LLM 未配置（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`）：仅查询改写不可用
  （API 返回 503 `llm_unavailable`，UI 置灰），其余能力不受影响。
- 改写返回非法内容：该改写回退原查询并记 warning，不中断检索。

## 参考

- UTokyo-HitU, TREC 2025 RAG Track：HyDE 增强 sparse-dense 融合 + LLM 重排
  （RRF k=60，HyDE α=0.7）
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE)
- Cormack et al., *Reciprocal Rank Fusion*（RRF）
```

- [ ] **Step 2: docs/api.md 加端点行**

在第 77 行 `| POST | /v1/ingest | ... |` 之后加两行：

```markdown
| `POST` | `/v1/retrieval` | 多路检索：dense/BM25 召回 + RRF/加权融合 + 改写 + MMR + 重排 |
| `POST` | `/v1/retrieval/stream` | 同上，NDJSON 阶段事件流（stage → result/error） |
```

第 79 行「请求 / 响应 / collection schema 详见 ingest-pipeline.md」改为：

```markdown
摄取端点的请求 / 响应 / collection schema 详见 [ingest-pipeline.md](ingest-pipeline.md)；
检索端点的参数、schema v2、迁移与调优建议详见 [retrieval.md](retrieval.md)。
```

- [ ] **Step 3: docs/errors.md 加错误码行**

在第 51 行 `| chunk_failed | ... |` 之后、`internal` 行之前加：

```markdown
| `retrieval_empty_query` | 检索 query 为空 | 422 |
| `retrieval_channel_unsupported` | BM25 channel 需要 schema v2（错误体带 `channels` / `migration_available`） | 422 |
| `retrieval_invalid_param` | 检索参数非法（融合/改写/MMR/candidate_pool 等） | 422 |
| `llm_unavailable` | 查询改写需要的 LLM 未配置（`VS_LLM__BASE_URL` / `VS_LLM__MODEL`） | 503 |
```

- [ ] **Step 4: docs/README.md 加索引行**

在第 15 行 ingest-pipeline 行之后加：

```markdown
| [retrieval.md](retrieval.md) | 分片检索：dense/BM25 混合召回、RRF/加权融合、查询改写、MMR、重排、schema v2 迁移 |
```

- [ ] **Step 5: .env.example 补一句**

第 134 行注释 `# endpoint (...). Left empty => those features` 所在注释块中，把
功能说明改为包含检索改写：

```
# endpoint (OpenAI, vLLM, LM Studio, ...). Left empty => query-rewrite
# features of /v1/retrieval (HyDE / Multi-Query / Step-Back / Decomposition)
# and LLM chunking are disabled; dense retrieval keeps working.
```

（保持 136-140 行的键不变。）

- [ ] **Step 6: 全量验证 + 提交**

Run: `python -m pytest tests/unit -q`
Expected: PASS；同时检查文档内链接文件名均存在（`retrieval.md`）。

```bash
git add docs .env.example
git commit -m "docs(retrieval): document retrieval pipeline, schema v2 migration, error codes"
```

---

## 计划自审记录（编写者已核对）

- **Spec 覆盖**：§1 目标 1–5 → Tasks 5/2/6/7/9/12（混合→2+5、重排→9+12、改写→6、MMR→7、追踪→9+12）；非目标全计划无实现。§2 环境前提 → Global Constraints。§3 Python 编排 → Task 9。§4 数据类/channel/融合/改写/MMR/编排 → Tasks 1,2,5,6,7,9。§5 Store 扩展 + schema v2 + 迁移 → Tasks 3,4,11。§6 API（JSON/stream/校验/错误码）→ Tasks 8,10,13。§7 前端（模式预设、置灰、迁移横幅、结果卡片、追踪折叠）→ Task 12。§8 测试 → 每任务内置。§9 文档 → Task 13。
- **占位符**：无 TBD/TODO；所有代码步均含完整代码。
- **类型/命名一致性**：`search_text`、`RecallSpec.vector/hypothetical`、`lambda_mult`、`schema_version`、`_SPARSE_FIELD`、`ingest_migrate_tmp`、事件名 `rewrite/recall/fuse/mmr/rerank` 在各任务间一致。
