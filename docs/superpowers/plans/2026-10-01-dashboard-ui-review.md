# Dashboard UI 系统性审视与改进计划

> 状态：**评审通过，实施中**。计划完成于 2026-10-01。
> 范围：`src/vector_service/static/dashboard/` 全部 21 个视图 + `templates/dashboard.html`。
> 本文档为诊断与规划基线；已实施部分见各节末尾的「实施记录」。
>
> | 项 | 状态 |
> |---|---|
> | S1 i18n 体系 | **已实施**（2026-10-01，见 §1 S1 实施记录） |
> | S2 操作反馈 | **已实施**（2026-10-01，见 §1 S2 实施记录） |
> | S3 功能 Bug 清单（B1–B14） | **已实施**（2026-10-01，见 §1 S3 实施记录） |
> | S4 版本号单一来源 | **已实施**（2026-10-01，见 §1 S4 实施记录） |
> | S5–S7、阶段一剩余项、阶段三 | 未开始 |

---

## 0. 已确认的决策前提

| 决策点 | 结论 |
|---|---|
| 改进幅度 | **分层推进**：先修系统性基础问题（i18n、反馈失效、bug、排版规范），再做信息架构与视觉升级 |
| 暗色模式 | **本次不引入**；令牌（CSS custom properties）结构为将来留口 |
| 响应式 | **维持桌面端（≥1024px）**；只修复 1024 宽度下的挤压/截断，不做手机适配 |
| 等宽字体 | **收窄使用范围**：mono 仅用于代码 / 端点 / ID / 指标数值等技术元素；中文标题与表单标签改用无衬线 |

其他默认约束：

- **稳定 DOM 锚点保持不变**：`#app`、`#led-healthz`、`#led-readyz`、`#dim-dots`、`#crumb-cat`、`#crumb-sub`（`templates/dashboard.html` 注释声明被测试断言）。视觉重构可改样式与内部 DOM，但这些 id 必须保留。
- 无构建步骤的约束不变：Vue 3 in-browser ESM，组件为原生 ESM `.js`，不引入打包器/TS/SFC。
- 后端接口形态不变；UI 改进优先消化已有接口。

审查方式：1440×900 / 1024×768 两档分辨率逐页走查截图（走查期证据，未入库），并对全部组件源码逐行审查（硬编码字符串、交互、可访问性、bug、死代码）。

---

## 1. 系统性问题（跨页面，按严重度排序）

### S1. 【P0】i18n 体系大面积失效：同一 Dashboard 三种文本来源

切换中/英文时，页面文本行为分三类，用户无法通过语言开关得到一致界面：

| 类型 | 文件 | 表现 |
|---|---|---|
| 已接入 `$t()` | `databases.js`、`overview.js`、`ops-queue/reindex/consistency/eval.js`、`models.js`（大部分） | 跟随语言切换 |
| **中文硬编码** | `collections.js`、`records.js` | 英文模式下仍全中文（数据库、集合、主键字段名、待嵌入文本…） |
| **英文硬编码** | `embeddings.js`、`rerank.js`、`similarity.js`、`search.js`、`browse.js`、`knowledge-base.js`、`retrieval.js` | 中文模式下仍全英文（MODEL、INPUT MODE、run、rerank、chunk、search、refresh、分页…） |

- `embeddings.js` / `rerank.js` / `similarity.js` 全文件 **0 处 `t()` 调用**；`retrieval.js` 甚至未 import `t`——而字典中 `retrieval.modes.*`、`retrieval.run/running`、`retrieval.llm_hint`、`retrieval.trace` 等 key **早已存在却未被使用**。
- 字典缺 key 导致裸 key 外泄：`modals.js` 引用的 `modals.preset_hnsw_cosine` 等预设按钮文案无任何字典条目，界面直接显示 key 字符串。
- 中英混杂细节：`similarity.js` 默认值同时存在英文 `'wireless mouse'` 与中文 `'一只猫'`；`records.js` 中文表单配英文状态值 `OK / FAIL: ...` 与英文按钮后缀 `(PUT)/(POST)`。
- `retrieval.js` 英文模式下仍显中文：占位符「输入检索问题」「按 token 预算裁剪」「精确匹配」「模糊匹配（SQLite 解析）」、徽标「LLM 未配置」、按钮「检索」、开关「检索追踪」。
- 单位/状态后缀散落硬编码：`s`（秒）、`q`（题数）、`OK`、`page ... of ...`、`B/KB/MB/GB` 等。

**改进措施**：

1. 在 `app.js` 的 `I18N.zh/en` 中补齐全部面板所需 key（按面板分组命名：`embeddings.*`、`rerank.*`、`similarity.*`、`records.*`、`search.*`、`browse.*`、`kb.*`、`retrieval.*`、`collections.*`、`modals.*`）。
2. 所有组件消灭自然语言硬编码：协议字面值（`cosine`、`HNSW`、`true`、端点路径）保留原文，其余一律 `t()`。
3. `t()` 增加开发期兜底：缺 key 时控制台 `console.warn` 一次（仍渲染 key 本身），便于后续查漏；在 CI/测试中加一个"两语言 key 集合一致"的断言脚本。
4. 状态机值（`idle/loading/ok/error`）只作内部值，展示时映射为本地化文案 + 语义样式。

**实施记录（2026-10-01）**：四条措施全部落地。

- 字典：`app.js` 的 `I18N.zh/en` 各 **664 个 key**，两侧 key 集合、`{placeholder}` 名完全一致；新增 `embeddings.* / rerank.* / similarity.* / records.* / search.* / browse.* / kb.* / retrieval.* / collections.* / modals.*` 分组，并清掉 8 个失效 key（`retrieval.title`、`common.status_label`、`common.new_collection`、`common.params_label_pre`、`common.running`、`rerank.query`、`common.status.idle`、`common.status.loading`）与两个拼接式 key（`databases.confirm_drop_pre` + `databases.and_collections` → 参数化 `databases.confirm_drop`）。
- 组件：21 个组件中的全部面板与容器改为全程走 `t()` / `$t()`；`markdown.js`（Markdown 渲染器）与 `ops-common.js`（状态→样式映射）不含任何文案，无需改动。协议字面值保留原文：端点路径、`cosine`/`ip`/`l2`、`HNSW`/`IVF_FLAT`/`DISKANN`/`FLAT`、`rrf`/`weighted`、`top_k`/`rrf_k`/`doc_id`/`filename`/`filter_expr`、`w dense/bm25/summary/graph`、MIME、`image/png`、字段名 `max_length`/`nullable`/`default_value`/`params`。顺带修掉 `databases.js` 的硬编码中文确认框（原文未列入 S1 清单但属同一缺陷）。
- 校验：`t()` 缺 key 时 `console.warn` 一次并渲染 key 本身；新增 `tests/unit/test_dashboard_i18n.py`（11 例）断言 key 集合一致、placeholder 一致、非空值、面板分组存在、检索阶段词覆盖 `retrieval/pipeline.py`、`t()` 语义（node 实跑）、面板无残留硬编码。
- 状态值：`records.js`、`knowledge-base.js` 的 `status` 改为内部值 + 本地化渲染（`common.status.ok` / `common.status.error`）；面板的状态→样式映射统一走 `feedback.js` 的 `EmptyState` / `StatusBanner`，不再有各自的映射表。
- 冒烟：1440×900 下 21 视图分别在中/英两种语言逐页走查，`#app` 内除语言切换按钮的「中」外无第二种语言残留；Browse / 入库浏览中的中文为语料内容（用户数据），符合预期；控制台 **0 条** `[i18n] missing key` 告警。
- 未纳入本次（按 §3 阶段划分留给后续）：`app.js` 页脚版本号 `0.2.0` 与 overview 的 `0.1.0` 不一致（S4）；语言选择未持久化（刷新回落中文）。

### S2. 【P0】操作反馈系统性失效：用户做了操作却看不到结果/错误

这是当前最伤产品体验的问题，且呈模式化出现：

- **`records.js` doFetch（按主键获取）**：接口返回数据被完全丢弃，无任何 ref 承接，UI 只显示 `OK`——用户永远看不到获取到的记录。
- **`rerank.js`**：模板完全不渲染 `status`；loading 无指示、错误信息无处显示；首次运行失败时 `result` 为 null，页面零反馈。
- **`embeddings.js` / `similarity.js`**：status 渲染在 `v-if="result"` 内部，**首次请求的 loading 与 error 不可见**。
- **`browse.js`**：查询 `status`（loading/ok/error）在模板中完全没被引用——查询无 spinner、查询错误用户看不到；空态在初始/加载中/失败/真空四种情况下文案相同（`collection is empty or filter matches nothing.`）。
- **`collections.js`**：详情加载失败不写缓存，模板永久显示「正在加载集合详情...」转圈，失败态与加载态不分。
- **空 catch 吞错普遍存在**：`embeddings.js:32`、`rerank.js:25`、`similarity.js:48`、`databases.js:22,30`、`collections.js:20,34`、`search.js:28,36,44`、`overview.js:46`、`knowledge-base.js:293,303`。刷新/加载失败一律静默，空态无法区分"真空"与"请求失败"。
- 校验/错误反馈大量使用原生 `alert()` / `confirm()`（至少 20+ 处），无样式、无上下文、不可持久；`knowledge-base.js` 的错误只出现在面板最底部一行常驻的 `status:` 调试行。

**改进措施**：

1. 建立统一反馈组件约定（复用现有 token，不引库）：
   - **行内状态条**：表单/结果区内 `role="status"` / `role="alert"` 的 banner（error / success / info 三态），替代 alert；
   - **按钮忙碌态**：所有提交按钮 busy 时 disabled + 文案/ spinner（如 `检索 → 检索中…`），杜绝重复提交（当前 run/rerank/chunk/search 等按钮 loading 期间均可重复点击）；
   - **空态四态区分**：加载中（spinner）/ 空（引导文案）/ 错误（原因 + 重试按钮）/ 未操作。
2. 所有空 catch 改为至少落入对应面板的 error 状态；`api()` 失败信息（`payload.error.message`）须可达可见。
3. 原生 `confirm()` 替换为模态确认（删除类危险操作显示影响面：库/集合名、将删除的数量），复用现有 modal 基础样式。

**实施记录（2026-10-01）**：三条措施全部落地。

- 新增 `components/feedback.js`（不引库、不 import `app.js` 以免成环），导出四个原语：
  - `StatusBanner`（`kind` = error/success/info/warn，error 用 `role="alert"`、其余 `role="status"`；可选 `retry` + `retryLabel`，仅在"重跑同一动作有意义"时由调用方传入，表单校验类提示不带重试按钮）；
  - `BusyButton`（busy 时 `disabled` + `aria-busy` + spinner + 文案切换；`emits: ['click']` 是必需的——否则 Vue 会把父级 `@click` 同时透传到根元素导致处理器触发两次）；
  - `EmptyState`（`state` = idle/loading/empty/error 四态；loading 出 spinner、error 出原因 + 重试、文案缺省回落 `common.state.*`）；
  - `NoticeBar` + `notify()/dismissNotice()`（无面板归属的全局提示条，挂在 `.main` 内，避免破坏 `.app` 的三行 grid 命名区）；`ConfirmHost` + `askConfirm()`（模态确认，列出影响面）。
- **反馈组件约定落地到全部面板**：`records`（B2：`doFetch` 返回值接入 `fetched` 并以表格展示，缺失主键单独告警）、`rerank`、`embeddings`、`similarity`（status 从 `v-if="result"` 内提到外面，首次请求的 loading/error 可见）、`browse`（查询四态空态 + 翻页失败 banner，二者按 `items.length` 互斥显示，失败不清空既有行）、`search`、`databases`、`collections`（详情失败写入 `detailErr`，不再永久转圈）、`overview`、`knowledge-base`（含 Docling 引擎状态）、`models`、`parser-cards`、`ops-queue`（含取消作业确认）、`ops-consistency`（修复前确认）、`ops-reindex`（提交重建 / 提升前确认）。
- **空 catch 全部落 error 态**：面板级失败进各自 `loadErr`/`dbsErr`/`modelsErr`/`parserErr`/`gateErr`；无面板归属的后台轮询（`app.js` 的模型自动刷新）按"进入失败"边沿上报一次到 NoticeBar、恢复时撤回自身提示，避免每 5s 重新弹出一个刚被关闭的提示。保留的空 catch 仅 3 处且均有注释说明：SSE 畸形帧、`JSON.parse` 探测、剪贴板 `execCommand` 回退——都不是网络请求。
- **原生 `alert()` / `confirm()` 清零**：`browse`、`collections`、`databases`、`records`、`ops-queue`、`ops-consistency`、`ops-reindex` 共 7 个破坏性面板改走 `askConfirm()`，模态列出影响面（库/集合/行数/filter 表达式/门禁 ID/候选 ref）。`records` 的删除体在弹确认框**之前**就构建并校验完毕，因此模态展示的影响面与实际发出的请求体完全一致。`ConfirmHost` 打开时聚焦"取消"按钮（误按回车不会确认删除），Esc 关闭。
- 新增 i18n key（两侧同步）：`records.fetched_missing/upsert_ok/delete_ok/vector_dims/no_vector`、`collections.dropped/index_dropped/index_created/index_field`、`databases.dropped`、`chunks.running`、`ops.reindex.submitting/promoting/refreshing`、`ops.consistency.scanning/repairing`、`kb.parser_failed`、`models.poll_failed`、`models.refreshing`。
- 校验：新增 `tests/unit/test_dashboard_feedback.py`（5 例）——剥掉整行注释后断言组件内**不存在** `alert(`/`confirm(` 调用点（`askConfirm(`/`settleConfirm(` 首字母大写故不匹配）、`feedback.js` 导出齐全且不 import `app.js`、7 个破坏性面板均走 `askConfirm()`、`app.js` 已挂载两个宿主。`test_dashboard_i18n.py` 增加"组件引用的每个字面 key 都已在两语言中定义"（拼接式 key 以前缀形式匹配、按尾点跳过）。
- 测试：`tests/` 全量通过（2 例按既有条件跳过）；`test_dashboard_route.py` 的 `browse.count("confirm(")` 断言改为 `askConfirm(`；`similarity` 面板的 `'kind:' not in sim` 改为精确匹配 prop 声明形态（`kind: { type:`）与 `<similarity-panel :kind`——原来的子串断言会把 i18n 占位符 `{ kind: ... }` 误判为 prop。
- 冒烟：21 个视图逐一点击走查，控制台 **0 error / 0 warning**（无 `[i18n] missing key`）；实测一致性面板的"修复"确认模态：`role="alertdialog"`、影响面表列出 `数据库=default`、`集合=ingest`、焦点默认落在"取消"，按 Esc 关闭且**未发出任何请求**。
- 未纳入本次（用户明确只选 S2，留给阶段一）：B1/B3/B4/B5/B7/B8/B12/B13/B14 等 §2 bug 清单其余条目仍待修；本节的措施 1–3 只覆盖"操作反馈"，不改这些 bug 的行为。

### S3. 【P0】确认的功能 Bug 清单

| # | 位置 | Bug |
|---|---|---|
| B1 | `modals.js` NewDbModal | `setup()` 返回 `{ name, close, submit }`，**未返回 `store`**，模板中 `store.modalErr.newDb` 渲染即 TypeError——「新建数据库」模态框永远打不开（已实测：控制台报错、弹层空白）。NewCollModal 正常 |
| B2 | `records.js` doFetch | 返回数据被丢弃（见 S2），功能不可用 |
| B3 | `browse.js` 分页跳转 | 跳页输入框绑定只读 computed `currentPage`（无 setter），赋值无效且告警，**跳页功能不可用** |
| B4 | `browse.js` 跨页/跨库选择 | 切库、改 filter 查询均不清 `selected`，已选不可见 PK 仍会被 delete selected 提交——**跨库误删风险**；全选框无 indeterminate 半选态；pageSize 改变不重置 offset |
| B5 | `search.js` | 切换集合后已选 output_fields 不重置，可把上一集合字段名提交给新集合；`score*100` 当百分比显示，对 l2 距离/非 0~1 分数语义错误且无 NaN 防护 |
| B6 | `retrieval.js` | `c.fusion_score.toFixed(4)`、`h.score.toFixed(2)` 无空值保护，`rerank_score === undefined` 时渲染崩溃；`content-type` 为 null 时 `.includes()` 抛错；`s.stages` 只 push 不去重 |
| B7 | `databases.js:83-84` | 行内 meta 斜杠两侧是**同一个值**（集合数）渲染两次，`X 集合数 / X`，第二个数字无数据来源（疑似本想显示 metadata 条目数） |
| B8 | `ops-reindex.js` | promote 按钮位于 `v-else`（gates 非空）分支内，**无门禁分支没有提升按钮**，与「可直接提升金丝雀」文案矛盾；promote 的 409 写入 gateErr 错误槽；只认 `gates[0]`/`checks[0]` |
| B9 | `retrieval.js` custom 模式 | `setMode` 无 custom 分支，点击只改名无任何效果——死模式 |
| B10 | `embeddings.js:52-53` | list 模式 JSON 解析失败静默回退 `[textInput]`；隐藏的 textInput 默认值 `'hello world'` 仍会随请求发出 |
| B11 | `collections.js:220,227` | 新索引表单用静态 `selected` 固定 cosine/HNSW，不随集合真实 metric 联动；params 只校验 JSON 不校验是对象 |
| B12 | `knowledge-base.js` | 循环内固定 `id="chunk-context-text"` 产生重复 id；SSE/XHR 流无超时、无 abort/取消按钮 |
| B13 | `ops-eval.js` | runDetail 打开后切换 tab 被静默吞掉（内容仍显示 run 详情）；`runBack`、`parseParams` 为死状态/死代码 |
| B14 | 多处 | window 事件监听（`refresh-dbs` 等）无 `onUnmounted` 移除，组件重挂载重复绑定 |

**实施记录（2026-10-01）**：B1–B14 全部处理完毕（B2 已随 S2 落地，本次无改动）。

| # | 结论 | 落地方式 |
|---|---|---|
| B1 | 已修 | `modals.js` NewDbModal 的 `setup()` 返回对象补回 `store`，模板读 `store.modalErr.newDb` 不再抛错，模态可正常打开 |
| B2 | 已修（S2） | 见 §1 S2 实施记录，本次未改动 |
| B3 | 已修 | 跳页框改为独立可写 `jumpPage`，`currentPage` 变更时用 watch 同步回去；`jumpTo()` 夹取到 `[1, totalPages]` 后换算 offset |
| B4 | 已修 | 新增 `resetSelectionState()`，在切库 / 切集合 / 改主键字段时清空 `selected`；`runQuery` 记录 `appliedFilter`，filter 表达式变化即清选择；`pageSize` 变更重置 offset；全选框补 `indeterminate`（`somePageSelected`） |
| B5 | 已修 | `watch(coll)` 清空 `outputFields` 并重取 schema；`metric` 从集合详情读取，结果区标题显示「metric · 越大/越小越接近」；分数改为 `formatScore()` 原始值 4 位小数，非有限值渲染 `—` |
| B6 | 已修 | 新增 `fmtScore()` 覆盖 fusion/rerank/trace 三处；`content-type` 取不到时按空串处理，不再 `.includes()` 抛错；`stages` 改为去重 push |
| B7 | 已修 | `databases.js` 行内 meta 右侧改为 metadata 条目数（`Object.keys(...).length`），不再是集合数渲染两次 |
| B8 | 已修 | 提升按钮从 `v-else-if="gates.length"` 分支内移出，无门禁分支同样可见；`promote` 失败改写入新 `promoteErr` 槽（不再污染 gateErr 的「重试=重跑门禁列表」语义）；按钮 `:disabled="!canPromote"` 并给出禁用原因 |
| B9 | 已修 | `setMode('custom')` 落地为真正的原始 JSON 请求体编辑器：进入时按当前控件生成 body，运行前 `JSON.parse` 并校验为对象，失败写 `retrieval.err.bad_json` |
| B10 | 已修 | list 模式下解析失败不再回退隐藏的 `textInput`（其 `'hello world'` 默认值曾被静默发出），改为报 `embeddings.err.bad_list` |
| B11 | 已修 | 新索引表单改为按集合键控的响应式状态（`newIndex`），在 `loadDetail` 时以 `payload.metric` 播种；params 除 JSON 可解析外还校验为**对象**（排除 `[]`/`5`/`null`） |
| B12 | 已修 | 删掉 `v-for` 内重复的 `id="chunk-context-text"`；`streamPost` 增加 120s 空闲看门狗与 `onabort` 处理，解析/入库进度条各加「取消」按钮 |
| B13 | 已修 | `switchTab` 清空 `runDetail`（此前被 `v-if` 静默吞掉）；`closeRun` 读取 `runBack` 回到来源 tab；删除死代码 `parseParams` |
| B14 | 已修 | `databases.js`、`collections.js` 补 `onUnmounted` 注销 `refresh-dbs` / `refresh-colls` 监听 |

**偏离计划的两处**（均为核实后主动收窄/扩展）：

- **B8「多 gate/check 时加选择器」未采纳**。核实后端契约后确认该建议不成立：`corpus/schema.py` 的 `regression_gates` 有 `UNIQUE(database, collection)`，一个 scope 至多一条门禁；`api/rebuild.py` 的 `promote_reindex_job` 自身通过 `get_gate_for_collection` + `get_latest_gate_check` 解析门禁，不接受客户端传入 gate/check。即客户端的 `gates[0]`/`checks[0]` 与服务端将读取的行**必然一致**，加选择器会暗示客户端能对另一条门禁提升。改为用 `scopeGate`/`latestCheck`/`canPromote`/`promoteBlockReason` 把这层语义写显式，并在界面注明「门禁按数据库+集合唯一」。
- **顺带修掉两处计划外缺陷**：① `collections.js` 的 `refreshColls` 用 `db::coll` 全键去比对只含集合名的 `live` 集合，导致每次刷新清空全部详情缓存——改为比对 `::` 之后的集合名（否则 B11 的表单状态也会被一并清掉）；② `ops-eval.js` 的 `runDetail.summary.*` 在 summary 缺失/不完整时渲染即崩，抽出 `summary` computed 兜底为 `{}`（`fixed()`/`pct()` 已把缺失值渲染为 `—`）。第 ② 条对应 §2 第 21 条笔记中的「KPI summary 缺失会崩」，其余同条笔记项（gates 分页、子表空态/局部 loading、无 run_id 行的可点击样式）仍留待阶段三。

- 新增 i18n key（两侧同步、placeholder 一致）：`search.score`、`search.metric_higher`、`search.metric_lower`、`retrieval.custom_json`、`retrieval.custom_hint`、`retrieval.custom_regen`、`retrieval.err.bad_json`、`embeddings.err.bad_list`、`ops.reindex.block_no_check`、`ops.reindex.block_check_failed`（`{status}`）、`ops.reindex.gate_scope_note`、`kb.cancelled`、`kb.err.timeout`（`{seconds}`）。「取消」按钮复用既有 `common.cancel`。
- 测试：`tests/` 全量 **958 passed / 2 skipped / 0 failed**（2 例按既有条件跳过）。`test_dashboard_route.py` / `test_dashboard_i18n.py` / `test_dashboard_feedback.py` 无改动即通过；`browse.js` 内一处注释因含 `test_no_hardcoded_panel_prose_survives` 的字面量黑名单词而改写。
- 未纳入本次：阶段一第 3 项（版本号单一来源，S4）、第 4 项（records 预填示例 ID）仍未开始；§2 各视图的「改进」类条目按计划留待阶段三/四。（第 3 项已于同日随 S4 落地，见本节 S4 实施记录；第 4 项仍待办。）

### S4. 【P1】版本号三处不一致

- 总览 KPI 显示 **0.1.0**：`GET /v1/system/status` → `api/system.py` → `vector_service/__init__.py: __version__ = "0.1.0"`；
- 底部状态栏硬编码 **0.2.0**：`app.js:1180`；
- Prometheus `VS_INFO{version=...}` 也硬编码 **0.2.0**：`core/lifespan.py:94-95`。

**措施**：确定单一真实版本（建议以 `__init__.py.__version__` 为准，先定版到正确号），状态栏改为从 `/v1/system/status` 读取（store 中已有 health 轮询，可合并），`VS_INFO` 引用 `__version__`，消灭全部硬编码版本字面量。

**实施记录（2026-10-01）**：四处版本号收敛为单一来源。

- **定版到 `0.2.0`**（而非把界面降回 0.1.0）。核实 git 历史后确认 0.2.0 才是真实进度：`685c5eb`（观测性：OTel tracing + Prometheus metrics + ops 面板）把 0.2.0 写进 `lifespan.py` 的 `VS_INFO`，`f782db7`（dashboard 拆分为模板 + 静态 Vue app）把 0.2.0 写进状态栏；而 0.1.0 停留在 `1165955`（脚手架）之后再未跟进，`pyproject.toml` 与 `__init__.py` 才是陈旧副本。
- **`__init__.py.__version__` 成为唯一字面量**：
  - `pyproject.toml` 改为 `dynamic = ["version"]` + `[tool.setuptools.dynamic] version = {attr = "vector_service.__version__"}`，打包元数据不再自存一份（setuptools 走 AST 静态读取，不触发导入）；
  - `core/lifespan.py` 的 `VS_INFO.labels(version=__version__, ...)` 改为引用该常量；
  - `uv.lock` 中 editable self 条目的 `version = "0.1.0"` 随之消失（`uv run` 自动重写）。
- **状态栏改从接口取值**：`app.js` 新增 `store.service = { version, embeddingBackend, storeBackend }`，由既有的 5s health 轮询（`pollHealth`，与 `#led-healthz` / `#led-readyz` 同一 tick）多拉一次 `GET /v1/system/status` 填充；状态栏三段由 `0.2.0` / `bge-m3` / `milvus` 三个字面量改为 `store.service.*` 绑定，未取到时渲染 `—`。顺带修掉计划外同类缺陷：`EMB` / `STORE` 两段此前也硬编码（恰好等于 `config.py` 的默认值，配置一改即说谎）。
- **失败语义**：该请求失败时保留上次成功的值、不弹提示——它与同 tick 的两个探针打的是同一个进程，服务不可达时两个 LED 已经变红，再叠一条需要手动关闭的提示只是同一信号的第二份副本（`catch` 处有注释说明）。
- **校验**：新增 `tests/unit/test_version_single_source.py`（6 例）——① 全树扫描（`src/**/*.py` + `components/*.js` + `templates/*.html`，排除 `__init__.py` 与 vendored 的 Vue 包）断言三段式版本字面量**有且仅有** `__init__.py` 一处，正则用 `(?<![\d.])\d+\.\d+\.\d+(?![\d.])` 以免把 `config.py` 的 `host = "0.0.0.0"` 误判；② `pyproject.toml` 不得有静态 `version`，`dynamic` 必须列出它且 `attr` 指向包属性；③ `__version__` 形如版本号（状态栏原样渲染）；④ 路由 `service.version == __version__`；⑤ 实跑 lifespan 后读 Prometheus registry，`vs_info` 的 version 标签集合恰为 `{__version__}`；⑥ 状态栏绑定 `store.service.*` 且 `app.js` 确实拉取该端点。
- **测试**：`tests/` 全量 **966 例 / 0 failed / 0 error / 2 skipped**（2 例按既有条件跳过）。
- **冒烟**：用**改前启动**的 dev server（仍在跑 0.1.0 的旧代码）配上热更新的静态 `app.js`——状态栏 VER 显示 **0.1.0**（旧字面量 0.2.0 已消失），EMB/STORE 显示 `bge-m3` / `milvus`，即状态栏确实取自接口而非字面量；控制台 0 error / 0 warning。另 `uv run python -c` 复核 `vector_service.__version__` 与 `importlib.metadata.version("vector-service")` 均为 `0.2.0`，attr 与打包元数据同源。
- **未纳入本次**：状态栏 `SVC` 段的 `vector-service` 仍是字面量（进程名常量，非配置项）；`app.js` 的 health 轮询与 overview 面板各自拉一次 `/v1/system/status`（每 5s 两次），合并为共享轮询属 §S6「模型数据收敛」的同类工作，留给阶段二。

### S5. 【P1】排版与字体系统：层级含混、截断频发

- **等宽字体被用在几乎所有中文标签与标题上**（导航、表单 label、卡片标题、KPI 标签），中英文排印混排时灰度不均、阅读节奏差。按决策收窄 mono：仅代码、端点、ID、字段名、数值保留 mono；中文叙述性文字走无衬线，并明确字重/字号层级（当前 fz-xs 11px … fz-xl 16px 梯度偏小，标题与正文区分弱）。
- **截断问题**（1024 宽尤其明显）：
  - 总览能力卡标题截断：「标准版（版面+…」「原生提取（快速…」「视觉大模型（V…」；
  - 圆形预热徽标内「已预热/未预热」三字竖排成一列；模型卡右侧状态 pill 在窄卡内「未加载」竖排；
  - 运维面板 label 列过窄，「数据库/集合」逐字换行。
- **卡片头信息挤压**：入库浏览每个分片卡头部一行塞 index 徽标 + chars/tokens + section_header + 页码 + 文件名；databases/collections 行的 `– / – ▸` / `– · – · – ▸` 含义不明（实为未加载详情的占位 + 独立展开箭头，视觉粘连）。
- 嵌套滚动：`ingested` 每个分片卡 `<pre>` `max-height:380px` 独立滚动条，外层页面再一层；ingested 全页高约 4533px。

**措施**：

1. 调整 token：明确「显示字（无衬线）/ 技术字（mono）」两套用途；梳理 h1–h4、label、value、caption 的字号字重对照表，写进 `dashboard.css` 顶部注释。
2. 卡片标题允许两行或卡内 flex 收缩给 min-width；徽标统一为横向 pill，禁止内部竖排；运维表单 label 列固定足够宽度（或改为 label 在上、控件在下的纵向布局）。
3. 占位元信息与展开箭头视觉分离（间距 + 弱化占位符色），加载完成前不渲染无意义的 `–` 堆。
4. 取消分片卡内部滚动（正文自然展开，由页面统一滚动）；超长正文给「展开全文/折叠」。

### S6. 【P1】表单/按钮/数据加载缺少统一规范

- 初始加载即产生 **5 次重复的 `GET /v1/models`**：三个 v-show 常驻的 embeddings 实例 + rerank + similarity 各发一次；面板顶部的"已注册模型列表"与下方 model 下拉又是同数据两次展示，且列表项不可操作、纯冗余。
- models 为空时仍可提交 `model: ''`（embeddings/rerank/similarity 均无空选项禁用）。
- top_k/top_n/page size 等数字输入：清空后 `v-model.number` 得 NaN/空仍会提交，无 JS 校验。
- 危险操作（删除库/集合/记录）按钮常驻、样式为红色但无收纳；records 删除按钮默认可用且预填示例主键 `sku-1/sku-2`，不展开参数直接点删除就会尝试删示例 ID。
- 同类控件交互不一致：query 在 similarity 用 textarea、rerank 用 input；文件 accept 标准不一；MIME 下拉模板在 embeddings/similarity 重复三份。

**措施**：

1. 模型清单收敛：store 统一缓存 `GET /v1/models`（models 视图已有轮询），各面板从 store 取数，删除不可操作的重复列表区；按 type 过滤。
2. 通用表单规约：必填空值禁用提交；数字输入统一 min/max/整数校验与错误提示；危险操作按钮收纳到行尾菜单或要求先展开参数区。
3. 提取共享工具：`fileToB64`、MIME 选项、dtype 映射（overview.js 与 models.js 重复定义）等收到单一 util 模块。

### S7. 【P2】可访问性（a11y）系统性欠债

- 几乎所有 `<label>` 未通过 `for` 关联控件（30+ 处），点击 label 不聚焦；
- 可点击行/token/开关均为 `div/span + @click`：无 `role`、无 `tabindex`、无键盘事件（databases/collections 行、browse/search 字段 token、retrieval trace 开关、ops 表格行）；
- 模式按钮组无 `tablist/radiogroup` 语义与 `aria-selected/pressed`（retrieval 模式、eval 双 tab；kb 的 source/preview 有 tablist 容器但内部缺 tab 角色）；
- 进度条无 `role="progressbar"`/aria-valuenow；结果区无 `aria-live`；banner 无 `role="alert/status"`；装饰 SVG 多数未标 `aria-hidden`（models.js 少数已做到）。

**措施**：随各阶段改造顺手补齐，不单独占用阶段：label 包控件或加 for；点击型元素改为原生 `<button>` 或加完整键盘支持；进度/状态区加 ARIA；装饰元素 `aria-hidden="true"`。

---

## 2. 逐页问题与改进（21 视图）

### 概览组

**1. overview 总览首页**（`components/overview.js`）
- 版本号 0.1.0 与页脚 0.2.0 矛盾（S4）。
- 5s 轮询失败完全静默，KPI 全变 `—` 且无手动刷新按钮；面板 v-show 常驻，非本页时间隔照跑（增加可见性判断再发请求）。
- 能力卡标题截断、预热徽标竖排、`/3` 分母硬编码与实际 profiles 数脱钩。
- 信息表 URI 表头与内联 style 模板（L111）应抽类；`<th>` 缺 `scope="row"`。
- **改进**：KPI 卡片保持现有四宫格结构；能力卡标题区重排（标题两行 + 横向状态 pill）；增加刷新/错误/时间戳状态；预热计数从 profiles 实际数量派生。

**2. models 模型列表**（`components/models.js`）
- `FAMILY_LABELS` 四类技术名恒显英文（可保留——属技术字，但建议加中文 title 提示）；自动刷新按钮英文模式仍显「开/关」；`- no instance` 后缀硬编码。
- refresh/lifecycle 失败 alert、无错误持久化；failed 态卡片无快捷重试。
- **改进**：整体卡片结构保留；状态 pill 三态样式已有基础，补 failed 卡片内「重试加载」按钮；开关文案本地化；mini-dots 装饰 aria-hidden。

### 模型试验台组（结构高度相似，建议统一模板）

**3. embeddings 文本嵌入 / 4. image-embeddings / 5. multimodal-embeddings**（`components/embeddings.js`）
- 全英文硬编码；顶部模型列表与下拉重复；status 首次请求不可见（S1/S2）。
- 图像：raw JSON textarea 预填 `[]`、MIME 全局单值、文件选择无已选反馈；多模态：非字符串项静默丢弃。
- **改进**：三个实例合并为单一参数化面板模板；删除不可操作列表区，模型下拉从 store 取数；文件输入显示已选文件名/数量 + 可移除；结果区改为「向量条数/维度/耗时」摘要 + 可折叠原始 JSON + 复制按钮；修 B10。

**6. rerank 重排**（`components/rerank.js`）
- 全英文；status 模板完全不渲染；结果只有裸 JSON；模式切换（逐行/JSON）无引导，切到 JSON 必 alert；默认 topN=3 与 3 条文档 magic 耦合。
- **改进**：结果做可读排序列表（名次、分数、文档摘要）+ 折叠 JSON；补 status banner 与按钮忙碌态；输入模式切换时给对应占位示例。

**7. similarity 相似度**（`components/similarity.js`）
- 全英文 + 中英默认值混杂；模式间切换不清文件选择，图片会被静默带入；4 个文件输入无已选反馈；结果裸 JSON。
- **改进**：切模式/切 query modality 时清空相关文件；显示已选文件；结果按候选逐条显示分数横条 + 折叠 JSON；MIME 模板去重。

### 向量库组

**8. databases 数据库管理**（`components/databases.js`）
- 已接 `$t()`，但 confirm/alert 硬编码中文；删除按钮常驻、confirm 不显示影响面（集合数）；meta 同值重复（B7）；行不可键盘展开。
- **改进**：meta 改为有意义的双指标（集合数 + metadata 条目数，后者接口已有）或只显示一个；删除走模态确认并显示「将连带删除 N 个集合」；行改 button 语义 + aria-expanded。

**9. collections 集合管理**（`components/collections.js`）
- 全中文硬编码（英文模式失效）；页面级选库 + 表单内重复选库；详情失败永久转圈；新索引表单 metric/类型不联动、单 option 冗余 select、params 校验弱。
- **改进**：保留页面级库选择，移除表单内重复选择；详情加失败态；索引表单 metric 从集合详情带入、目标字段直接取向量字段名（去掉单选项 select）、params 校验为对象并给常用参数模板（HNSW M/efConstruction）；字段/索引区维持 details 折叠但加计数。

**10. records 记录**（`components/records.js`）
- 一个页面三种动词（写入 PUT / 获取 POST / 删除 POST）挤在同一表单；doFetch 结果丢弃（B2）；删除参数折叠但删除按钮常驻 + 预填示例 ID；ids/texts/fields 长度不一致无校验；primary/vector 字段名自由文本不跟 schema。
- **改进（IA 调整）**：拆为三个带标题的操作区（或 tab：写入 / 查询 / 删除）；获取结果用表格/JSON 视图呈现；删除区默认禁用，展开并填入有效参数后才可提交；字段名从集合 schema 下拉选择；写入前校验数组等长。

**11. search 检索（向量搜索试验台）**（`components/search.js`）
- 全英文硬编码；无空结果态；hits 缺字段时对 undefined 迭代；分数 ×100 语义可疑（B5）；output_fields token 不可键盘操作。
- **改进**：结果列表显示名次/主键/分数（分数按 metric 类型正确呈现：相似度显大、距离显小并加注），fields 用可展开详情；空结果显空态；字段选择改 checkbox/token 按钮（键盘可达），随集合切换重置；文件头注释中 image 模式与实现不符，删死注释。

**12. browse 浏览数据**（`components/browse.js`）
- 全英文；ID 与 DOC_ID 列同值重复；列集合只按当前页取并集致跨页表头漂移；跳页失效（B3）；跨库/跨过滤选择误删风险（B4）；查询 loading/error 不可见；三处重复表达范围信息。
- **改进**：列从 schema 定义（而非当前页并集），主键列与 fields 中同语义列去重；修跳页（独立 ref）；切库/改 filter/改 pageSize 清选择并重置 offset；全选框支持 indeterminate；合并范围信息为一处；删除按钮显示已选数量；分页/提示全部本地化。

### 知识库组（`components/knowledge-base.js`，4 个子视图）

**13. parse 文档解析**
- 全英文 label；无文件 alert；进度条无 ARIA；source/preview 切换缺 tab 角色；copy 失败永不报错。
- **改进**：保留 source/preview 双视图（这是该页正确范式，复制到 ingested）；补全 tab 语义；复制失败给提示。

**14. chunk 分片测试**
- 全英文；doChunk 无 busy/可重复提交；无空态；错误只在底部调试行；与 ingest 的分片参数是两套互不同步的 refs。
- **改进**：分片参数（strategy/size/overlap/percentile/context）抽为共享组合/对象，chunk 与 ingest 同源；按钮忙碌态；结果空态；chunks 列表卡片化（序号、chars/tokens、页码）。

**15. ingest 一键入库**
- 主提交按钮在首屏折行以下；全英文；alert 校验；流无取消；默认 `{}` metadata textarea 冗余。
- **改进**：保留 4 步进度 stepper（现有样式基础好），但提交按钮上移或步骤区压缩到首屏可见；流增加「取消」（XHR abort）；metadata 默认空字符串由 JS 处理而非 textarea 写 `{}`；进度 stage 文案本地化。

**16. ingested 入库浏览**
- 全页约 4533px：每卡头部单行塞 5 类信息；正文显示**原始 markdown 源码**（`#`、`**`、表格管道符可见）；每卡内嵌滚动条；doc_id 重复行；排序仅当前页；分页英文；底部 status 调试行常驻。
- **改进（重点页）**：卡片头改为两行（第一行：文档名/路径面包屑；第二行：分片序号、chars/tokens、页码等元信息，弱化）；正文复用 parse 页的 **source/preview 切换**（preview 渲染 markdown）；取消卡内滚动，改页面统一滚动 + 超长折叠；同一 doc_id 支持按文档分组折叠；去掉调试行或并入正式状态条；分页本地化。

**17. retrieval 分片检索**（`components/retrieval.js`）
- 全英文标签 + 多处中文残留；i18n key 已存在却不用；hybrid 默认平铺约 20 个高级控件无渐进披露；custom 死模式（B9）；dense/bm25 可同时关闭；无空结果态；流无取消、stage 不去重；分数 toFixed 崩溃风险（B6）。
- **改进（重点页）**：
  - 模式组保留 4 个但语义重整：**basic**（query/top_k/库集合）、**hybrid**（通道勾选 + fusion 两个参数，其余折叠）、**advanced**（routing/rewrite/diversity/rerank/budget/filter 分区折叠进「高级选项」手风琴）、**custom** 给真正的 JSON 原生请求编辑或暂时下线；
  - 各 fieldset 加 legend；数字/分数空值保护；流式 stage 本地化 + 去重 + 「取消检索」；空结果态；错误 banner 加重试；结果行补 fusion/rerank 分数列与 trace 区（trace 开关改 button）。

### 运维组

**18. queue 任务队列**（`components/ops-queue.js`）
- 表格无空态；分页仅上/下页、无页码无跳页、offset 原始显示；SSE 无 onerror/连接状态，且与 4s 轮询重复拉取；瞬时错误直显英文。
- **改进**：SSE 打开后停轮询、显示连接态（含自动重连提示）；空态行；分页统一为共享 pager 组件（页码 + 跳页 + 总数）；错误走 banner。

**19. reindex 索引重建**（`components/ops-reindex.js`）
- promote 按钮位置错误（B8）；无门禁时流程引导矛盾；客户端零门禁全靠服务端 409；canaryPercent 可为 0；只认第一项 gate/check；非法数字静默兜底；无法取消重建 job。
- **改进**：重建 → 评测/门禁 → 提升做成显式步骤条：每步显示可用状态，promote 按钮固定存在但条件禁用（job 未完成 / canary=0 / 门禁未过分别说明原因）；多 gate/check 时加选择器；轮询错误不写表单错误槽；增加取消 job 入口。

**20. consistency 一致性校验**（`components/ops-consistency.js`）
- 空函数 start/空 watch 死代码；扫描前无首屏引导/结果占位；修复不检查 embedder 是否加载；重扫时旧报告无 stale 标识；`OK` 硬编码。
- **改进**：删死代码；扫描前显说明面板，扫描后出结果区（总数/不一致数/可修复项）；repair 前检查 embedder 状态并引导；重扫中旧报告置灰 + 「重新扫描中」。

**21. eval 评测与门禁**（`components/ops-eval.js`）
- runDetail 吞 tab 切换（B13）；gates 无分页；questions/versions/runs/checks 子表无空态/局部 loading；KPI summary 缺失会崩；指标名硬编码两处；gate checks 无 run_id 仍显可点击样式。
- **改进**：run 详情改为覆盖层或带明确返回（回到来源 tab）；子表统一 loading/空态；KPI 加结构保护；不可点击行去掉 affordance；gates 列表补分页；scope Apply 未改动时禁用。

---

## 3. 分阶段实施计划

> 原则：每阶段独立可验收、可提交；P0 阶段不改视觉风格，只做止血，降低回归面。
> 稳定 DOM id（见 §0）全程保留；每阶段完成后跑现有 dashboard 相关测试 + 两语言逐页冒烟。

### 阶段一（P0 止血）：让界面"不出错、有反馈"

1. **修全部功能 bug**：B1 NewDbModal（一行级修复）→ B2 records fetch 落库展示 → B3 browse 跳页 → B4 选择清理 → B5/B6 崩溃与分数保护 → B7 重复 meta → B8 promote 按钮 → B9/B10/B11 → B14 监听清理。
2. **反馈链路补齐**：status/error 渲染归位（rerank、embeddings、similarity、browse、collections 详情）；提交按钮统一 busy/disabled；空 catch 全部落 error 态。
   → **已于 2026-10-01 完成**（提前落地，见 §1 S2 实施记录）；本阶段剩余条目为第 1、3、4 项。
3. **版本号单一来源**（S4）：定版、页脚与 VS_INFO 统一。
   → **已于 2026-10-01 完成**（见 §1 S4 实施记录）；本阶段剩余条目为第 4 项。
4. 删除 records 预填示例 ID 的危险默认值；删除按钮加使能条件。

验收：21 视图中/英双语点击走查无 console error；每个提交动作都有 loading/成功/失败三态可见。

### 阶段二（P1 基础）：i18n 全覆盖 + 设计令牌收口

1. 补齐两语言全部 key 并替换全部硬编码（S1）；加"双语 key 集合一致"校验；`t()` 缺 key 开发告警。
2. 字体分流（S5）：mono 收窄到技术元素；建立字号/字重层级表；修复全部截断（卡片标题、徽标、运维 label 列）。
3. 模型数据收敛（S6）：单一缓存来源，消灭 5 次重复请求与不可操作列表区；数字输入校验；共享工具提取（fileToB64、MIME、dtype、pager）。
4. 元信息占位与展开箭头解耦；分片卡取消内嵌滚动。

验收：切换 EN 后无中文残留（技术字除外），切回 ZH 无英文残留；1024 宽无文字竖排/截断。

### 阶段三（P2 信息架构）：重排拥挤页面

1. **records**：拆分为 写入/查询/删除 三区或 tab。
2. **retrieval**：四模式语义重整 + 高级选项手风琴 + custom 模式落实/下线。
3. **ingested**：卡片头两行 + source/preview 双视图 + 按文档分组。
4. **collections**：去重复选库；索引表单联动。
5. **browse**：列基于 schema、去重、跨页稳定。
6. **reindex**：步骤化提升流程；**queue/eval**：统一 pager、空态、tab 行为。
7. 三个 embedding 面板统一为参数化单模板。

验收：重点页（retrieval/ingested/records/ingest）首屏信息密度下降，核心动作首屏可达；原有功能零回归。

### 阶段四（P3 视觉提升）： polish

在不动结构的前提下统一视觉节奏：KPI/卡片间距与边框层级、空态插画/引导、按钮层级（primary/secondary/danger/ghost 使用规约）、表格行 hover/斑马纹策略、装饰元素（dim-dots、model-dots）收敛或弱化；同步更新 README 中的面板截图。

### 明确不做

暗色模式；手机端/全面响应式；引入前端构建工具链；后端接口重构；恢复此前已回退的"选中文本翻译"功能。

---

## 4. 证据与参考

- 逐页截图（21 视图 × 中/英，1440 与 1024 两档，问题模态）：走查期产物，未入库。
- 设计系统现状：`src/vector_service/static/dashboard/dashboard.css`
- 组件目录：`src/vector_service/static/dashboard/components/`
- 挂载外壳（含稳定 id 声明）：`src/vector_service/templates/dashboard.html`
