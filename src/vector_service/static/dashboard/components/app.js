// =====================================================================
// app.js -- root component + global reactive store + shared helpers
// =====================================================================
import { createApp, defineComponent, reactive, onMounted, onUnmounted } from '../vue.esm-browser.prod.js';

import OverviewPanel from './overview.js';
import ModelsPanel from './models.js';
import DatabasesPanel from './databases.js';
import CollectionsPanel from './collections.js';
import RecordsPanel from './records.js';
import SearchPanel from './search.js';
import BrowsePanel from './browse.js';
import RerankPanel from './rerank.js';
import EmbeddingsPanel from './embeddings.js';
import SimilarityPanel from './similarity.js';
import KnowledgeBasePanel from './knowledge-base.js';
import RetrievalPanel from './retrieval.js';
// Operations panels: queue / rebuild / consistency / evaluation.
import OpsQueuePanel from './ops-queue.js';
import OpsReindexPanel from './ops-reindex.js';
import OpsConsistencyPanel from './ops-consistency.js';
import OpsEvalPanel from './ops-eval.js';
import { NewDbModal, NewCollModal } from './modals.js';
// Shared feedback convention (S2): the notice strip and the modal
// confirmation are app-level hosts — panels mount the inline banner,
// busy button and empty block themselves.
import { NoticeBar, ConfirmHost, notify, dismissNotice, noticeState } from './feedback.js';

// =====================================================================
// i18n: Chinese (default) + English
// =====================================================================
// All user-facing strings live here as a flat dotted-key map. The
// `t(key)` helper picks the right translation based on `store.locale`.
// New keys: just add the same dotted name to both maps. Missing
// translations fall back to the key itself (visible in dev as a
// reminder to fill it in).

const I18N = {
  zh: {
    // Brand + topbar
    'brand.dashboard': 'Dashboard',
    'topbar.dim_unit': '维',
    'status.SVC': 'SVC',
    'status.VER': 'VER',
    'status.EMB': 'EMB',
    'status.STORE': 'STORE',
    'status.LOADED': '已加载',

    // Sidebar nav
    'nav.overview': '总览首页',
    'nav.models': '模型列表',
    'nav.embeddings': '文本嵌入',
    'nav.image_embeddings': '图像嵌入',
    'nav.multimodal_embeddings': '图文嵌入',
    'nav.rerank': '重排',
    'nav.similarity': '相似度',
    'nav.databases': '数据库管理',
    'nav.collections': '集合管理',
    'nav.records': '记录',
    'nav.search': '检索',
    'nav.browse': '浏览数据',
    'nav.parse': '文档解析',
    'nav.chunk': '分片测试',
    'nav.ingest': '一键入库',
    'nav.retrieval': '分片检索',
    'retrieval.modes.basic': '基础',
    'retrieval.modes.hybrid': '混合',
    'retrieval.modes.advanced': '高阶',
    'retrieval.modes.custom': '自定义',
    'retrieval.run': '检索',
    'retrieval.running': '检索中…',
    'retrieval.llm_hint': 'LLM 未配置',
    'retrieval.trace': '检索追踪',
    'nav.chunks_view': '入库浏览',
    'nav.Navigation': '导航 · Navigation',
    'nav.Overview': '概览',
    'nav.Models': '模型',
    'nav.Vector Store': '向量库',
    'nav.Knowledge Base': '知识库',

    // Breadcrumb cats
    'cat.overview': '概览',
    'cat.models': '模型',
    'cat.store': '向量库',
    'cat.kb': '知识库',

    // Overview panel
    'overview.version': '版本',
    'overview.uptime': '运行时长',
    'overview.uptime_sub': '进程启动以来',
    'overview.loaded_total': '已加载 / 已注册',
    'overview.loaded_total_sub': '跨所有族',
    'overview.vector_store': '向量库',
    'overview.connection_status': '向量库连接状态',
    'overview.backend': '后端',
    'overview.status_label': '状态',
    'overview.db_count': '数据库数量',
    'overview.auto_refresh': '自动刷新 · 每 5 秒',
    'overview.capabilities': '已加载能力与资源占用',
    'overview.capabilities_sub': '当前后台持有模型权重 / 会话的所有组件',
    'overview.cap_inference': '推理模型',
    'overview.cap_parser': '文档解析引擎',
    'overview.cap_inference_empty': '暂无已加载的推理模型，可在「模型列表」中加载。',

    // Parser capability cards (shared by overview + parse pane)
    'cap.warm': '已预热',
    'cap.cold': '未预热',
    'cap.model_free': '无模型 · 轻量管道（model-free）',
    'cap.warm_idle': '转换器已构建；模型管道将在首次解析时加载。',
    'cap.cold_hint': '尚未构建转换器；可点「预热」提前加载，或直接解析时自动构建。',
    'cap.warm_btn': '预热',
    'cap.evict_btn': '清空缓存',
    'cap.warming': '预热中…',
    'cap.evicting': '清空中…',
    'cap.rebuild_note': '清空缓存不会停止解析服务：下次使用该方案时转换器会自动重建（重新加载 / 下载权重）。',
    'cap.resource_hidden': '占用无法通过 torch 统计',
    'cap.onnx_note': '注：RapidOCR 基于 ONNX Runtime 运行，其权重占用无法通过 torch 内省，未计入上方数字。',
    'cap.total_weights': '权重合计',
    'cap.warm_failed': '预热失败',
    'cap.evict_failed': '清空缓存失败',
    'cap.kind.layout': 'DocLayNet 版面分析',
    'cap.kind.table': 'TableFormer 表格识别',
    'cap.kind.ocr': 'RapidOCR 文字识别',
    'cap.kind.vlm': 'Granite 视觉语言模型',
    'cap.kind.other': '辅助模型',

    // Common
    'common.refresh': '刷新',
    'common.cancel': '取消',
    'common.create': '创建',
    'common.delete': '删除',
    'common.database': '数据库',
    'common.collection': '集合',
    'common.model': '模型',
    'common.metric': '度量',
    'common.name': '名称',
    'common.results': '结果',
    'common.basic_info': '基础信息',
    'common.delete_failed': '删除失败: ',
    'common.run': '运行',
    'common.auto': '自动',
    'common.loaded': '已加载',
    'common.unloaded': '未加载',
    'common.no_models': '没有已注册模型.',
    'common.new_db': '+ 新建数据库',
    'common.click_refresh': '点击"刷新"加载数据库.',
    'common.model_id_ops': '加载 / 卸载均按 model_id 操作',

    // Models panel
    'models.auto_refresh_label': '自动刷新:',
    'models.load': '加载',
    'models.unload': '卸载',
    'models.loading': '加载中…',
    'models.unloading': '卸载中…',
    'models.refreshing': '刷新中…',
    'models.load_failed': '加载失败',
    'models.unload_failed': '卸载失败',
    'models.meta_params': '参数量',
    'models.meta_vram': '显存占用',
    'models.meta_ram': '内存占用',
    'models.meta_load_time': '加载耗时',
    'models.device_label': '设备',
    'models.dtype_label': '数据类型',

    // Databases panel
    'databases.coll_count': '集合数',
    'databases.meta_count': 'metadata 字段',
    'databases.backend_meta': '后端 metadata',
    'databases.coll_list': '集合清单',
    'databases.no_collections': '该数据库下暂无集合.',
    'databases.hint': '每个集合必须归属一个数据库;删除数据库会同时删除其下所有集合.',
    'databases.confirm_drop_title': '删除数据库',
    'databases.list_failed': '加载数据库列表失败：',
    'databases.detail_failed': '加载数据库详情失败：',
    'databases.loading': '正在加载数据库详情...',
    'databases.confirm_drop': '确认删除数据库 "{name}" 及其下所有集合?',
    'databases.dropped': '已删除数据库 {name}',

    // Modals
    'modals.new_db': '新建数据库',
    'modals.new_collection': '新建集合',
    'modals.coll_name': '集合名',
    'modals.coll_name_hint': '1-64 字符',
    'modals.pk_field': '主键字段',
    'modals.pk_field_hint': '必须出现在下方标量字段里',
    'modals.scalar_fields': '标量字段(含主键)',
    'modals.unnamed': '（未命名）',
    'modals.primary_suffix': '主键',
    'modals.dtype': 'dtype',
    'modals.max_length': 'max_length',
    'modals.is_primary': 'is_primary',
    'modals.nullable': 'nullable',
    'modals.default_value': 'default_value',
    'modals.add_scalar': '＋ 添加一个标量字段',
    'modals.preset_id_cat_price': '预设: id + category + price',
    'modals.preset_id_cat_price_desc': 'varchar 主键 + varchar 分类 + float 价格',
    'modals.preset_id_year': '预设: id + year',
    'modals.preset_id_year_desc': 'varchar 主键 + int32 年份',
    'modals.vector_field': '向量字段',
    'modals.unique': '唯一 · 不可改',
    'modals.field_name': '字段名',
    'modals.dim': '维度',
    'modals.index_params': '索引参数',
    'modals.metric_type': 'metric_type',
    'modals.index_type': 'index_type',
    'modals.params_json': 'params (JSON)',
    'modals.add_index': '＋ 添加一个索引',
    'modals.create_collection': '创建集合',
    'modals.name_hint': '1-64 字符,字母 / 数字 / 下划线',

    // Ingest panel (upload progress + result cards)
    'ingest.upload': '上传',
    'ingest.uploading': '上传中…',
    'ingest.processing': '处理中…',
    'ingest.group.file': '源文件',
    'ingest.group.parse': '解析设置',
    'ingest.group.chunk': '分片设置',
    'ingest.group.dest': '嵌入与写入',
    'ingest.stage.upload': '上传',
    'ingest.stage.parse': '解析',
    'ingest.stage.chunk': '分片',
    'ingest.stage.embed': '嵌入',
    'ingest.stage.upsert': '写入',
    'ingest.stage_hint.parse': '正在解析文档内容…',
    'ingest.stage_hint.chunk': '正在按 token 数切分文本…',
    'ingest.stage_hint.embed': '正在调用嵌入模型生成向量…',
    'ingest.stage_hint.upsert': '正在写入向量库…',
    'ingest.failed_at': '失败阶段：',
    'ingest.success': '摄取成功',
    'ingest.success_empty': '摄取完成,但文档未产生任何分片',
    'ingest.failed': '摄取失败',
    'ingest.banner_ok': '成功',
    'ingest.banner_empty': '空文档',
    'ingest.stat.chunks': '分片数',
    'ingest.stat.pages': '页数',
    'ingest.stat.tokens': 'tokens',
    'ingest.stat.duration': '耗时',
    'ingest.stat.size': '文件大小',
    'ingest.stat.model': '嵌入模型',
    'ingest.doc_id': 'doc_id',
    'ingest.copy': '复制',
    'ingest.copied': '已复制 ✓',
    'ingest.raw': '原始响应 JSON',
    'ingest.metadata_hint': '可选,仅 title / author / filename / page_count 会写入标量字段',
    'ingest.no_dbs': '（暂无数据库,请先在数据库管理中创建）',
    'ingest.err.no_db': '请先选择数据库。',
    'ingest.err.no_file': '请先选择要上传的文件。',
    'ingest.err.bad_meta': 'metadata 必须是合法的 JSON 对象,例如 {"title":"年度报告"}。',
    'ingest.err.bad_size': 'chunk size 必须在 1–8192 之间。',
    'ingest.err.bad_overlap': 'chunk overlap 必须满足 0 ≤ overlap < chunk size。',
    'ingest.err.network': '网络错误,请求未完成。',
    'ingest.parse_pages_prefix': '已解析 ',
    'ingest.parse_pages_suffix': ' 页',

    // Parse 面板（上传字节进度 + 逐页解析进度）
    'parse.upload': '解析',
    'parse.uploading': '上传中…',
    'parse.parsing': '解析中…',
    'parse.failed': '解析失败',
    'parse.pages_prefix': '已解析 ',
    'parse.pages_suffix': ' 页',
    'parse.hint.parse': '正在解析文档内容,大文件可能需要数分钟…',
    'parse.err.no_file': '请先选择要解析的文件。',
    'parse.err.network': '网络错误,请求未完成。',
    'parse.stat.chars': '文本字符',
    'parse.stat.pages': '页数',
    'parse.stat.images': '图片',
    'parse.stat.tables': '表格',
    'parse.stat.ocr': 'OCR 页数',
    'parse.stat.duration': '总耗时',
    'parse.engine_status': '解析引擎状态（预热 / 缓存）',

    // 文本分片策略（chunking 包注册的 strategy 名称）
    'chunk.strategy.fixed': '固定窗口（按 token 硬切）',
    'chunk.strategy.paragraph': '段落分片（段落优先打包）',
    'chunk.strategy.recursive': '递归分片（Markdown 感知，默认）',
    'chunk.strategy.semantic': '语义分片（嵌入断点）',
    'chunk.strategy.llm': 'LLM 分片（智能主题边界）',
    'chunk.add_context': '上下文增强（LLM 为每个分片生成上下文前缀）',
    'chunk.context_prefix': '上下文：',

    // 入库浏览
    'chunks.page_size': '每页条数',
    'chunks.doc_filter': 'doc_id 精确匹配',
    'chunks.name_filter': '文件名关键字',
    'chunks.query': '查询',
    'chunks.running': '查询中…',
    'chunks.reset': '重置',
    'chunks.count': '分片总数',
    'chunks.stat_total': '分片总数',
    'chunks.stat_page': '页码',
    'chunks.stat_returned': '本页条数',
    'chunks.no_collection': 'ingest 集合尚不存在,请先通过「一键入库」写入文档。',
    'chunks.no_match': '没有符合条件的分片。',

    // 解析 profile（PDF 管线）
    'profile.label': '解析方案',
    'profile.auto': '自动（按文档类型）',
    'profile.standard': '标准版（版面+表格+选择性 OCR）',
    'profile.native': '原生提取（快速，纯数字 PDF）',
    'profile.vlm': '视觉大模型（VLM）',
    'profile.desc.auto': '根据文档类型自动选择：带文本层的数字 PDF 使用快速原生提取；扫描件、混合 PDF 及图片使用版面分析 + OCR；Word / PPT / HTML 使用无模型快速通道。',
    'profile.desc.standard': 'DocLayNet 版面分析 + TableFormer 表格识别 + 选择性 RapidOCR：数字页面跳过 OCR，扫描页面自动识别。适用于绝大多数 PDF 与图片。',
    'profile.desc.native': '基于 docling-parse 的纯文本提取，无需加载模型、速度极快；但无法识别扫描件，表格还原较弱。仅适用于带文本层的数字 PDF。',
    'profile.desc.vlm': '端到端视觉语言模型（默认 Granite-Docling-258M）直接理解整页版面，适合复杂排版；速度较慢，首次使用需下载模型权重。',

    // Operations: queue / reindex / consistency / evaluation
    'cat.ops': '运维',
    'nav.Operations': '运维 · Operations',
    'nav.queue': '任务队列',
    'nav.reindex': '索引重建',
    'nav.consistency': '一致性校验',
    'nav.eval': '评测与门禁',
    'common.apply': '应用',
    'common.back': '返回',
    'common.close': '关闭',
    'common.next': '下一页',
    'common.prev': '上一页',
    'ops.queue.all': '全部',
    'ops.queue.attempts': '尝试次数',
    'ops.queue.cancel': '取消任务',
    'ops.queue.cancel_confirm': '确定要取消该任务吗？',
    'ops.queue.cancel_title': '取消任务',
    'ops.queue.cancel_requested': '已请求取消',
    'ops.queue.chunk_count': '分片数',
    'ops.queue.created': '创建时间',
    'ops.queue.detail': '任务详情',
    'ops.queue.filename': '文件名',
    'ops.queue.finished': '完成时间',
    'ops.queue.job_id': '任务 ID',
    'ops.queue.page_count': '页数',
    'ops.queue.progress': '进度',
    'ops.queue.stage': '阶段',
    'ops.queue.status': '状态',
    'ops.queue.tokens_used': '已用 Tokens',
    'ops.queue.updated': '更新时间',
    'ops.reindex.canary_hint': '金丝雀比例：新版本先服务该比例的请求，通过门禁后再整体提升。',
    'ops.reindex.confirm': '确定要提交重建任务吗？',
    'ops.reindex.embed_hint': '嵌入模型留空则沿用当前绑定；指定其他模型将构建全新向量索引。',
    'ops.reindex.embed_keep': '留空：沿用当前模型',
    'ops.reindex.gate': '发布门禁',
    'ops.reindex.job': '重建任务',
    'ops.reindex.no_check': '尚无门禁检查结果。',
    'ops.reindex.no_gate': '未配置门禁',
    'ops.reindex.no_gate_hint': '可直接提升金丝雀；建议先配置回归门禁再提升。',
    'ops.reindex.promote': '金丝雀提升',
    'ops.reindex.promote_confirm': '确定将金丝雀索引提升为正式索引吗？',
    'ops.reindex.promote_title': '提升金丝雀索引',
    'ops.reindex.promote_hint': '提升在单事务内完成：金丝雀引用切换为正式引用，旧物理索引随后由维护任务清理。',
    // Why the promote button is disabled. The server resolves the gate
    // for a scope itself (regression_gates is UNIQUE per scope), so the
    // client can state the exact rule it is about to be held to.
    'ops.reindex.block_no_check': '该门禁尚无评测记录，请先跑一次评测再提升。',
    'ops.reindex.block_check_failed': '最近一次门禁校验为「{status}」，通过后才能提升。',
    'ops.reindex.gate_scope_note': '门禁按「数据库 + 集合」唯一，提升时服务端读取的就是这一条。',
    'ops.reindex.submit': '提交重建',
    'ops.reindex.submitting': '提交中…',
    'ops.reindex.promoting': '提升中…',
    'ops.reindex.refreshing': '刷新中…',
    'ops.consistency.missing': 'Milvus 缺失',
    'ops.consistency.orphans': 'Milvus 孤儿',
    'ops.consistency.repair': '修复',
    'ops.consistency.repair_confirm': '确定按 SQLite 事实来源修复差异吗？（缺失补写、孤儿删除）',
    'ops.consistency.repair_title': '修复一致性差异',
    'ops.consistency.repair_hint': '修复以 SQLite 为准：补写缺失向量、删除孤儿向量。',
    'ops.consistency.report': '一致性报告',
    'ops.consistency.scan': '开始扫描',
    'ops.consistency.scanning': '扫描中…',
    'ops.consistency.repairing': '修复中…',
    'ops.consistency.show_ids': '展开 ID 列表',
    'ops.eval.all_gates': '全部门禁',
    'ops.eval.all_sets': '全部评测集',
    'ops.eval.answer': '答案',
    'ops.eval.channel_attribution': '通道归因',
    'ops.eval.checks': '检查记录',
    'ops.eval.chunks': '命中分片',
    'ops.eval.expected': '期望',
    'ops.eval.gate': '门禁',
    'ops.eval.has_answer': '含标准答案',
    'ops.eval.max_drop': '最大降幅',
    'ops.eval.min': '下限',
    'ops.eval.per_question': '逐题明细',
    'ops.eval.question': '问题',
    'ops.eval.questions': '问题列表',
    'ops.eval.rerank_compare': '重排前后对比',
    'ops.eval.run': '运行',
    'ops.eval.runs': '运行历史',
    'ops.eval.set_name': '评测集',
    'ops.eval.show_answer': '展开答案',
    'ops.eval.tab_gates': '门禁',
    'ops.eval.tab_sets': '评测集',
    'ops.eval.thresholds': '阈值',
    'ops.eval.versions': '版本',

    // Shared field labels across ops panels
    'common.total': '总计',
    'common.offset': '偏移',
    'common.scope': '范围',
    'common.error': '错误',
    'common.params': '参数',
    'common.tag': '标签',
    'common.question_count': '问题数',
    'common.doc_id': '文档 ID',
    'common.job_id': '任务 ID',
    'common.set_id': '评测集 ID',
    'common.version_id': '版本 ID',
    'common.run_id': '运行 ID',
    'common.gate_id': '门禁 ID',
    'common.check_id': '检查 ID',
    'common.question_id': '问题 ID',
    'common.candidate_ref': '候选引用',
    'common.baseline_run_id': '基线运行 ID',
    'common.baseline': '基线',
    'common.template': '模板',
    'common.include_answer': '生成答案',
    'common.canary': '金丝雀',
    'common.active': '正式',
    'common.chunks_unit': '个分片',
    'common.docs_unit': '个文档',
    'common.promoted': '已提升',
    'common.retired': '已退役',

    // Job / run / check status words
    'ops.status.queued': '排队中',
    'ops.status.parsing': '解析中',
    'ops.status.chunking': '分片中',
    'ops.status.embedding': '嵌入中',
    'ops.status.upserting': '写入中',
    'ops.status.extracting': '抽取中',
    'ops.status.graphing': '图谱构建中',
    'ops.status.evaluating': '评测中',
    'ops.status.running': '运行中',
    'ops.status.done': '已完成',
    'ops.status.failed': '失败',
    'ops.status.cancelled': '已取消',
    'ops.status.passed': '已通过',

    // Consistency extras
    'ops.consistency.overall': '总体',
    'ops.consistency.physical_indexes': '物理索引',
    'ops.consistency.sqlite_leaves': 'SQLite 叶子分片',
    'ops.consistency.source_truth': '事实来源',
    'ops.consistency.milvus_rows': 'Milvus 行数',
    'ops.consistency.repaired': '已修复',
    'ops.consistency.drift': '存在偏差',

    // Rebuild form labels
    'ops.reindex.embed_model': '嵌入模型',
    'ops.reindex.canary_percent': '金丝雀比例（%）',
    'ops.reindex.batch_size': '批大小',

    // Eval KPIs / tables
    'ops.eval.questions_kpi': '问题数',
    'ops.eval.mean_recall': '平均 Recall',
    'ops.eval.mean_mrr': '平均 MRR',
    'ops.eval.mean_ndcg': '平均 nDCG',
    'ops.eval.doc_hit_rate': '文档命中率',
    'ops.eval.pre': '重排前',
    'ops.eval.post': '重排后',
    'ops.eval.doc_label': '文档',
    'ops.eval.channel': '通道',
    'ops.eval.hit_share': '命中占比',
    'ops.eval.raw_recall': '原始召回',

    // ------------------------------------------------------------------
    // Panel strings that used to be hard-coded in the components. Keys are
    // grouped by panel; anything that is a protocol literal (metric names,
    // index types, field names like `top_k`, endpoint paths) stays as-is
    // in the templates and is deliberately NOT translated.
    // ------------------------------------------------------------------
    'common.on': '开',
    'common.off': '关',
    'common.unknown': '未知',
    'common.first': '首页',
    'common.last': '末页',
    'common.query': '查询',
    'common.reset': '重置',
    'common.seconds': '秒',
    'common.minutes': '分',
    'common.no_instance': '无实例',
    'common.mime': 'MIME',
    'common.primary_field': '主键字段',
    'common.vector_field': '向量字段',
    'common.last_op': '最近操作：',
    'common.status.ok': '成功',
    'common.status.error': '失败',

    // Operation feedback (S2): labels shared by the inline status banner,
    // the four-state empty block and the confirmation modal.
    'common.retry': '重试',
    'common.confirm': '确认',
    'common.confirm_title': '确认操作',
    'common.impact': '影响面',
    'common.rows': '行数',
    'common.deleting': '删除中…',
    'common.load_failed': '加载失败：',
    'common.state.idle': '尚未加载',
    'common.state.loading': '加载中…',
    'common.state.empty': '暂无数据',
    'common.state.error': '加载失败',

    // Embeddings panels (text / image / multimodal)
    'embeddings.kind.text': '文本',
    'embeddings.kind.image': '图像',
    'embeddings.kind.multimodal': '图文',
    'embeddings.title.text': '文本嵌入',
    'embeddings.title.image': '图像嵌入',
    'embeddings.title.multimodal': '图文嵌入',
    'embeddings.no_models': '没有可用的{kind}模型。',
    'embeddings.input_mode': '输入方式',
    'embeddings.mode.single': '单条',
    'embeddings.mode.list': '列表',
    'embeddings.input_text': '输入文本',
    'embeddings.list_json_array': '列表（JSON 数组）',
    'embeddings.select_images': '选择图片',
    'embeddings.hint.local_files': '本地文件 → 自动转 base64',
    'embeddings.json_list': 'JSON 列表',
    'embeddings.hint.each_item': '每项：data / mime',
    'embeddings.err.no_input': '请选择文件或粘贴 JSON 数组。',
    'embeddings.err.no_items': '请至少添加一项文本或图片。',
    'embeddings.err.bad_list': '批量输入的 JSON 不是数组，请修正后重试。',
    'embeddings.running': '运行中…',

    // Rerank panel
    'rerank.title': '交叉编码器重排',
    'rerank.registered': '已注册重排模型',
    'rerank.no_models': '没有可用的重排模型。',
    'rerank.suffix': '重排',
    'rerank.input_mode': '输入方式',
    'rerank.mode.lines': '每行一条',
    'rerank.mode.json': 'JSON 数组',
    'rerank.documents': '文档',
    'rerank.run': '重排',
    'rerank.results': '结果 · 命中 {n} 条',
    'rerank.err.json': 'JSON 模式需要字符串数组。',
    'rerank.running': '重排中…',

    // Similarity panel
    'similarity.title.text': '文本相似度',
    'similarity.title.image': '图像相似度',
    'similarity.title.multimodal': '图文相似度',
    'similarity.mode.text': '文本',
    'similarity.mode.image': '图像',
    'similarity.mode.multimodal': '图文',
    'similarity.query_text': '查询文本',
    'similarity.candidate_docs': '候选文档',
    'similarity.hint.one_per_line': '每行一条',
    'similarity.query_image': '查询图片',
    'similarity.candidate_images': '候选图片',
    'similarity.hint.pick_several': '可一次多选',
    'similarity.query_modality': '查询模态',
    'similarity.modality.text': '文本',
    'similarity.modality.image': '图像',
    'similarity.candidate_texts': '候选文本',
    'similarity.err.query_empty': '查询文本不能为空。',
    'similarity.err.no_docs': '请至少添加一条候选文档。',
    'similarity.err.no_query_image': '请选择查询图片。',
    'similarity.err.no_doc_images': '请至少选择一张候选图片。',
    'similarity.err.no_candidates': '请至少添加一项文本或图片候选。',
    'similarity.running': '运行中…',

    // Search panel
    'search.title': '向量检索',
    'search.query_mode': '查询方式',
    'search.mode.text': '服务端嵌入（query_text）',
    'search.mode.vector': '直接提供向量（query_vector）',
    'search.query_text': '查询文本',
    'search.query_vector': '查询向量',
    'search.filter_title': '过滤表达式（Milvus 原生）',
    'search.output_fields': '输出字段',
    'search.hint.click_toggle': '点击切换',
    'search.err.bad_vector': 'query_vector 必须是 JSON 数组。',
    'search.err.no_selection': '请选择数据库和集合。',
    'search.err.failed': '检索失败：',
    'search.run': '检索',
    'search.no_hits': '没有命中任何记录。',
    'search.running': '检索中…',
    'search.idle_hint': '设置查询条件后点击「检索」。',
    // Hit scores are metric-dependent: cosine/ip are similarities
    // (larger is closer), l2 is a distance (smaller is closer). Shown
    // next to the raw value so it is not read as a percentage.
    'search.score': '原始分数',
    'search.metric_higher': '越大越接近',
    'search.metric_lower': '越小越接近',

    // Browse panel
    'browse.title': '选择数据库 / 集合',
    'browse.sub': '集合行的分页浏览',
    'browse.query_params': '查询参数',
    'browse.query_params_sub': '按主键升序；向量字段始终隐藏',
    'browse.page_size': '每页条数',
    'browse.filter_expr': '过滤表达式',
    'browse.output_fields': '输出字段',
    'browse.hint.readonly': '只读',
    'browse.rows_range': '{from} - {to} / {total} 行',
    'browse.empty_summary': '空',
    'browse.stat_total': '总行数',
    'browse.stat_page': '页码',
    'browse.stat_returned': '本页条数',
    'browse.delete_selected': '删除选中',
    'browse.delete_by_filter': '按条件删除',
    'browse.clear_selection': '清除选择',
    'browse.delete_hint': '按条件删除作用于全集合所有匹配行，而不只是本页',
    'browse.deleted_one': '已删除 1 行。',
    'browse.deleted_n': '已删除 {n} 行。',
    'browse.delete_failed': '删除失败：',
    'browse.confirm_delete_one': '按主键删除选中的这 1 行？此操作不可撤销。',
    'browse.confirm_delete_n': '按主键删除选中的 {n} 行？此操作不可撤销。',
    'browse.confirm_delete_filter': '删除全集合中所有匹配该过滤条件的行？\n\n{expr}\n\n此操作不可撤销。',
    'browse.err.no_filter': '请先填写过滤表达式。',
    'browse.empty': '集合为空，或过滤条件没有匹配到任何行。',
    'browse.select_all': '全选 / 取消全选本页所有行',
    'browse.select_row': '选择 {id}',
    'browse.pager_info': '第 {from}-{to} 行 / 共 {total} 行 · 第 {page}/{pages} 页',
    'browse.jump_to': '跳至',
    'browse.go': '跳转',
    'browse.footer_hint': '悬停单元格查看完整 JSON 值；跨页勾选后点「删除选中」，或用「按条件删除」删除全集合匹配行；集合为空时分页器隐藏。',
    'browse.running': '查询中…',
    'browse.idle_hint': '选择集合后点击「查询」。',
    'browse.confirm_delete_title': '删除向量行',

    // Collections panel
    'collections.select_db': '选择数据库',
    'collections.current_db': '当前数据库',
    'collections.no_dbs': '（暂无数据库）',
    'collections.reload_dbs': '重新加载数据库列表',
    'collections.list_title': '集合列表',
    'collections.new': '+ 新建集合',
    'collections.pick_db_first': '请先选择数据库。',
    'collections.empty': '该数据库下暂无集合。',
    'collections.loading_detail': '正在加载集合详情...',
    'collections.stat.dim': '维度',
    'collections.stat.count': '数据量',
    'collections.stat.metric': '度量',
    'collections.stat.fields': '主键 / 向量字段',
    'collections.dim_value': '{dim} 维',
    'collections.fields_n': '字段（{n}）',
    'collections.primary': '主键',
    'collections.indexes_n': '索引（{n}）',
    'collections.new_index': '＋ 新建索引',
    'collections.target_field': '目标字段',
    'collections.vector_field_suffix': '（向量字段）',
    'collections.index_type': '索引类型',
    'collections.params_hint': 'JSON，例如 M=16, efConstruction=200',
    'collections.create_rebuild': '创建 / 重建',
    'collections.confirm_drop': '确认删除集合 "{name}"？',
    'collections.confirm_drop_index': '确认删除字段 "{field}" 上的索引？',
    'collections.drop_index_failed': '删除索引失败：',
    'collections.create_index_failed': '新建索引失败：',
    'collections.err.bad_params': 'params 必须是合法 JSON 对象。',
    'collections.confirm_drop_title': '删除集合',
    'collections.confirm_drop_index_title': '删除索引',
    'collections.list_failed': '加载集合列表失败：',
    'collections.detail_failed': '加载集合详情失败：',
    'collections.creating': '创建中…',
    'collections.dropped': '已删除集合 {name}',
    'collections.index_dropped': '已删除索引 {field}',
    'collections.index_created': '已创建索引 {field}',
    'collections.index_field': '索引字段',

    // Records panel
    'records.title': '记录写入',
    'records.input_mode': '输入方式',
    'records.mode.texts': '由服务端嵌入（texts）',
    'records.mode.vectors': '直接提供向量',
    'records.ids': '主键值数组（ids）',
    'records.texts': '待嵌入文本（texts）',
    'records.vectors': '已嵌入向量（vectors）',
    'records.fields': '附加字段（fields，按行对齐 ids）',
    'records.field_set': '字段集',
    'records.delete_params': '删除参数',
    'records.delete_params_hint': '用于下方的「按条件删除」按钮',
    'records.delete_mode': '删除方式',
    'records.del_mode.ids': '按主键列表',
    'records.del_mode.filter': '按过滤表达式',
    'records.del_ids': '主键值',
    'records.del_filter': '过滤表达式',
    'records.upsert': '写入（PUT）',
    'records.fetch': '按主键获取（POST）',
    'records.delete': '按条件删除（POST）',
    // Format hints for the now-empty inputs. Placeholders are not
    // submitted, so they cannot repeat the pre-filled-example bug.
    'records.ph.ids': '["id-1", "id-2"]',
    'records.ph.texts': '["第一段文本", "第二段文本"]',
    'records.ph.vectors': '[[0.1, 0.2, 0.3]]',
    'records.ph.fields': '[{"category": "mouse"}]',
    'records.ph.del_ids': '["id-1", "id-2"]',
    'records.ph.del_filter': 'category == \'mouse\'',
    'records.err.ids_json': 'ids 必须是 JSON 数组。',
    'records.err.texts_json': 'texts 必须是 JSON 数组。',
    'records.err.vectors_json': 'vectors 必须是 JSON 数组。',
    'records.err.fields_json': 'fields 必须是 JSON 数组。',
    'records.err.no_selection': '请选择数据库和集合。',
    'records.err.no_ids': '请填写至少 1 个主键。',
    'records.err.no_filter': '请填写过滤表达式。',
    'records.confirm_delete': '确认删除？此操作不可撤销。',
    'records.confirm_delete_title': '删除向量',
    'records.upserting': '写入中…',
    'records.fetching': '获取中…',
    'records.deleting': '删除中…',
    'records.fetched_n': '已获取 {n} 条记录',
    'records.fetched_missing': '{n} 个主键不存在',
    'records.upsert_ok': '已写入 {n} 行',
    'records.delete_ok': '已删除 {n} 行',
    'records.vector_dims': '维度 {n}',
    'records.no_vector': '未返回向量',

    // Modals (validation + index presets)
    'modals.err.db_name_required': '数据库名不能为空。',
    'modals.err.create_failed': '创建失败',
    'modals.err.one_primary': '必须且只能有一个主键字段（is_primary）。当前：{n}',
    'modals.err.primary_varchar': '主键字段必须是 varchar。当前：{dtype}',
    'modals.err.primary_name_required': '主键字段名不能为空。',
    'modals.err.varchar_max_length': 'varchar 字段 {name} 需要 max_length。',
    'modals.err.field_name_required': '每个字段都必须有名字。',
    'modals.err.primary_mismatch': '主键字段（{name}）与标量字段中的 is_primary 不一致。',
    'modals.err.dim': '向量维度必须 ≥ 1。',
    'modals.err.no_db': '请先创建数据库。',
    'modals.err.coll_name_required': '集合名不能为空。',
    'modals.preset_hnsw_cosine': '预设：HNSW + cosine',
    'modals.preset_hnsw_cosine_desc': '内存图索引，召回率高，适合中小规模',
    'modals.preset_ivf_l2': '预设：IVF_FLAT + l2',
    'modals.preset_ivf_l2_desc': '倒排聚类，欧氏距离',
    'modals.preset_diskann_ip': '预设：DISKANN + ip',
    'modals.preset_diskann_ip_desc': '磁盘索引，适合超大规模',
    'modals.preset_use_default': '预设：默认（HNSW + cosine）',
    'modals.preset_use_default_desc': 'M=16, efConstruction=200',

    // Retrieval panel
    'retrieval.query': '检索问题',
    'retrieval.query_placeholder': '输入检索问题',
    // 'custom' mode posts the textarea verbatim instead of the
    // control-driven body, so it gets its own copy.
    'retrieval.custom_json': '请求体 (JSON)',
    'retrieval.custom_hint': '按原文发送，忽略上方控件',
    'retrieval.custom_regen': '按当前控件重新生成',
    'retrieval.err.bad_json': '请求体必须是 JSON 对象：',
    'retrieval.group.channels': '通道与融合',
    'retrieval.fusion': '融合方式',
    'retrieval.group.routing': '意图路由',
    'retrieval.auto_route': '自动路由通道',
    'retrieval.use_llm': '使用 LLM 分类器',
    'retrieval.group.rewrite': '查询改写',
    'retrieval.rewrite_enabled': '启用',
    'retrieval.method.hyde': 'HyDE',
    'retrieval.method.multi_query': '多查询',
    'retrieval.method.step_back': '回溯提问',
    'retrieval.method.decompose': '问题分解',
    'retrieval.n_variants': '变体数',
    'retrieval.group.diversity': '多样性与重排',
    'retrieval.cross_encoder': '交叉编码器重排',
    'retrieval.candidate_pool': '候选池',
    'retrieval.group.context': '上下文预算与元数据过滤',
    'retrieval.max_tokens': '最大 tokens',
    'retrieval.max_tokens_ph': '按 token 预算裁剪',
    'retrieval.doc_id_ph': '精确匹配',
    'retrieval.filename_ph': '模糊匹配（SQLite 解析）',
    'retrieval.no_hits': '没有命中任何分片。',
    'retrieval.chunk_label': '分片',
    'retrieval.chars': '字符',
    'retrieval.sub_queries': '子查询',
    'retrieval.top': '命中',
    // Server-emitted stage names (retrieval/pipeline.py).
    'retrieval.stage.route': '意图路由',
    'retrieval.stage.rewrite': '查询改写',
    'retrieval.stage.recall': '召回',
    'retrieval.stage.fuse': '融合',
    'retrieval.stage.mmr': '多样性重排',
    'retrieval.stage.rerank': '重排',
    'retrieval.stage.expand': '上下文扩展',
    'retrieval.stage.compress': '上下文压缩',

    // Knowledge-base panels (parse / chunk / ingest / ingested)
    'kb.parse_title': '文档解析',
    'kb.chunk_title': '分片测试',
    'kb.ingest_title': '一键入库',
    'kb.chunks_title': '入库浏览',
    'kb.file': '文件',
    'kb.strategy': '分片策略',
    'kb.breakpoint': '断点百分位',
    'kb.chunk_size': '分片大小',
    'kb.overlap': '重叠',
    'kb.markdown': 'Markdown 文本',
    'kb.chunk_btn': '分片',
    'kb.chunks_n': '分片数：{n}',
    'kb.chars_tokens': '{chars} 字符 · {tokens} tokens',
    'kb.page': '第 {n} 页',
    'kb.embed_model': '嵌入模型',
    'kb.source': '源码',
    'kb.preview': '预览',
    'kb.copy': '复制',
    'kb.copied': '已复制',
    'kb.expand_full': '展开全文',
    'kb.collapse': '折叠',
    'kb.download_md': '下载 .md',
    'kb.metadata': '元数据 (JSON)',
    'kb.pager_info': '{from}-{to} / {total} · 第 {page}/{pages} 页',
    'kb.chunking': '分块中…',
    'kb.dbs_failed': '加载数据库列表失败：',
    'kb.models_failed': '加载模型列表失败：',
    'kb.parser_failed': '加载解析引擎状态失败：',
    // Stream cancel / watchdog outcome, shown in the panel banner.
    'kb.cancelled': '已取消。',
    'kb.err.timeout': '流已连续 {seconds} 秒无响应，已中止。',
    'models.poll_failed': '模型列表刷新失败：',
  },
  en: {
    // Brand + topbar
    'brand.dashboard': 'Dashboard',
    'topbar.dim_unit': 'dim',
    'status.SVC': 'SVC',
    'status.VER': 'VER',
    'status.EMB': 'EMB',
    'status.STORE': 'STORE',
    'status.LOADED': 'LOADED',

    // Sidebar nav
    'nav.overview': 'Overview',
    'nav.models': 'Model List',
    'nav.embeddings': 'Text Embed',
    'nav.image_embeddings': 'Image Embed',
    'nav.multimodal_embeddings': 'Multimodal Embed',
    'nav.rerank': 'Rerank',
    'nav.similarity': 'Similarity',
    'nav.databases': 'Databases',
    'nav.collections': 'Collections',
    'nav.records': 'Records',
    'nav.search': 'Search',
    'nav.browse': 'Browse',
    'nav.parse': 'Parse',
    'nav.chunk': 'Chunk Tester',
    'nav.ingest': 'One-Click Ingest',
    'nav.retrieval': 'Retrieval',
    'retrieval.modes.basic': 'Basic',
    'retrieval.modes.hybrid': 'Hybrid',
    'retrieval.modes.advanced': 'Advanced',
    'retrieval.modes.custom': 'Custom',
    'retrieval.run': 'Retrieve',
    'retrieval.running': 'Retrieving…',
    'retrieval.llm_hint': 'LLM not configured',
    'retrieval.trace': 'Retrieval trace',
    'nav.chunks_view': 'Stored Chunks',
    'nav.Navigation': 'Navigation',
    'nav.Overview': 'Overview',
    'nav.Models': 'Models',
    'nav.Vector Store': 'Vector Store',
    'nav.Knowledge Base': 'Knowledge Base',

    // Breadcrumb cats
    'cat.overview': 'Overview',
    'cat.models': 'Models',
    'cat.store': 'Vector Store',
    'cat.kb': 'Knowledge Base',

    'overview.version': 'version',
    'overview.uptime': 'uptime',
    'overview.uptime_sub': 'since startup',
    'overview.loaded_total': 'loaded / registered',
    'overview.loaded_total_sub': 'across families',
    'overview.vector_store': 'vector store',
    'overview.connection_status': 'vector store status',
    'overview.backend': 'backend',
    'overview.status_label': 'status',
    'overview.db_count': 'db count',
    'overview.auto_refresh': 'auto-refresh · every 5s',
    'overview.capabilities': 'Loaded capabilities & resources',
    'overview.capabilities_sub': 'Every component currently holding model weights / a session in the backend',
    'overview.cap_inference': 'Inference models',
    'overview.cap_parser': 'Document parsing engine',
    'overview.cap_inference_empty': 'No inference models loaded yet — load one in "Model List".',

    // Parser capability cards (shared by overview + parse pane)
    'cap.warm': 'warm',
    'cap.cold': 'cold',
    'cap.model_free': 'model-free · lightweight pipeline',
    'cap.warm_idle': 'Converter built; model pipelines load on the first parse.',
    'cap.cold_hint': 'Converter not built yet — warm it up ahead of time, or let it build on the first parse.',
    'cap.warm_btn': 'Warm up',
    'cap.evict_btn': 'Evict cache',
    'cap.warming': 'warming…',
    'cap.evicting': 'evicting…',
    'cap.rebuild_note': 'Evicting the cache does NOT stop parsing: the converter rebuilds automatically on the next use of this profile (reloading / re-downloading weights).',
    'cap.resource_hidden': 'resource usage invisible to torch introspection',
    'cap.onnx_note': 'Note: RapidOCR runs on ONNX Runtime; its weight footprint is invisible to torch introspection and is not included in the figures above.',
    'cap.total_weights': 'total weights',
    'cap.warm_failed': 'Warm-up failed',
    'cap.evict_failed': 'Evict failed',
    'cap.kind.layout': 'DocLayNet layout analysis',
    'cap.kind.table': 'TableFormer table recognition',
    'cap.kind.ocr': 'RapidOCR text recognition',
    'cap.kind.vlm': 'Granite vision-language model',
    'cap.kind.other': 'helper model',

    'common.refresh': 'refresh',
    'common.cancel': 'cancel',
    'common.create': 'create',
    'common.delete': 'delete',
    'common.database': 'database',
    'common.collection': 'collection',
    'common.model': 'model',
    'common.metric': 'metric',
    'common.name': 'name',
    'common.results': 'results',
    'common.basic_info': 'basic info',
    'common.delete_failed': 'delete failed: ',
    'common.run': 'run',
    'common.auto': 'auto',
    'common.loaded': 'loaded',
    'common.unloaded': 'unloaded',
    'common.no_models': 'no models registered.',
    'common.new_db': '+ new database',
    'common.click_refresh': 'click "refresh" to load databases.',
    'common.model_id_ops': 'load / unload is keyed by model_id',

    'models.auto_refresh_label': 'auto-refresh:',
    'models.load': 'load',
    'models.unload': 'unload',
    'models.loading': 'loading…',
    'models.unloading': 'unloading…',
    'models.refreshing': 'refreshing…',
    'models.load_failed': 'load failed',
    'models.unload_failed': 'unload failed',
    'models.meta_params': 'parameters',
    'models.meta_vram': 'VRAM',
    'models.meta_ram': 'RAM',
    'models.meta_load_time': 'load time',
    'models.device_label': 'device',
    'models.dtype_label': 'dtype',

    'databases.coll_count': 'collection count',
    'databases.meta_count': 'metadata fields',
    'databases.backend_meta': 'backend metadata',
    'databases.coll_list': 'collection list',
    'databases.no_collections': 'no collections yet.',
    'databases.hint': 'each collection belongs to a database; dropping a database drops its collections too.',
    'databases.confirm_drop_title': 'Drop database',
    'databases.list_failed': 'failed to load the database list: ',
    'databases.detail_failed': 'failed to load database detail: ',
    'databases.loading': 'loading database detail...',
    'databases.confirm_drop': 'drop database "{name}" and all its collections?',
    'databases.dropped': 'dropped database {name}',

    'modals.new_db': 'new database',
    'modals.new_collection': 'new collection',
    'modals.coll_name': 'collection name',
    'modals.coll_name_hint': '1-64 chars',
    'modals.pk_field': 'primary key field',
    'modals.pk_field_hint': 'must appear in scalar fields',
    'modals.scalar_fields': 'scalar fields (with primary)',
    'modals.unnamed': '(unnamed)',
    'modals.primary_suffix': 'primary',
    'modals.dtype': 'dtype',
    'modals.max_length': 'max_length',
    'modals.is_primary': 'is_primary',
    'modals.nullable': 'nullable',
    'modals.default_value': 'default_value',
    'modals.add_scalar': '+ add scalar field',
    'modals.preset_id_cat_price': 'preset: id + category + price',
    'modals.preset_id_cat_price_desc': 'varchar PK + varchar cat + float price',
    'modals.preset_id_year': 'preset: id + year',
    'modals.preset_id_year_desc': 'varchar PK + int32 year',
    'modals.vector_field': 'vector field',
    'modals.unique': 'unique - immutable',
    'modals.field_name': 'field name',
    'modals.dim': 'dim',
    'modals.index_params': 'index params',
    'modals.metric_type': 'metric_type',
    'modals.index_type': 'index_type',
    'modals.params_json': 'params (JSON)',
    'modals.add_index': '+ add index',
    'modals.create_collection': 'create collection',
    'modals.name_hint': '1-64 chars, alphanumeric / underscore',

    // Ingest panel (upload progress + result cards)
    'ingest.upload': 'upload',
    'ingest.uploading': 'uploading…',
    'ingest.processing': 'processing…',
    'ingest.group.file': 'source file',
    'ingest.group.parse': 'parsing',
    'ingest.group.chunk': 'chunking',
    'ingest.group.dest': 'embedding & storage',
    'ingest.stage.upload': 'upload',
    'ingest.stage.parse': 'parse',
    'ingest.stage.chunk': 'chunk',
    'ingest.stage.embed': 'embed',
    'ingest.stage.upsert': 'upsert',
    'ingest.stage_hint.parse': 'Parsing the document…',
    'ingest.stage_hint.chunk': 'Splitting text into token chunks…',
    'ingest.stage_hint.embed': 'Calling the embedding model…',
    'ingest.stage_hint.upsert': 'Writing vectors to the store…',
    'ingest.failed_at': 'Failed at stage: ',
    'ingest.success': 'Ingest succeeded',
    'ingest.success_empty': 'Ingest finished, but no chunks were produced',
    'ingest.failed': 'Ingest failed',
    'ingest.banner_ok': 'SUCCESS',
    'ingest.banner_empty': 'EMPTY',
    'ingest.stat.chunks': 'chunks',
    'ingest.stat.pages': 'pages',
    'ingest.stat.tokens': 'tokens',
    'ingest.stat.duration': 'duration',
    'ingest.stat.size': 'file size',
    'ingest.stat.model': 'embed model',
    'ingest.doc_id': 'doc_id',
    'ingest.copy': 'copy',
    'ingest.copied': 'copied ✓',
    'ingest.raw': 'raw response JSON',
    'ingest.metadata_hint': 'optional; only title / author / filename / page_count stored as scalar fields',
    'ingest.no_dbs': '(no databases — create one in Databases first)',
    'ingest.err.no_db': 'Select a database first.',
    'ingest.err.no_file': 'Select a file to upload.',
    'ingest.err.bad_meta': 'metadata must be a valid JSON object, e.g. {"title":"annual report"}.',
    'ingest.err.bad_size': 'chunk size must be between 1 and 8192.',
    'ingest.err.bad_overlap': 'chunk overlap must satisfy 0 ≤ overlap < chunk size.',
    'ingest.err.network': 'Network error — the request did not complete.',
    'ingest.parse_pages_prefix': 'parsed ',
    'ingest.parse_pages_suffix': ' pages',

    // Parse panel (byte-upload progress + per-page parse progress)
    'parse.upload': 'parse',
    'parse.uploading': 'uploading…',
    'parse.parsing': 'parsing…',
    'parse.failed': 'Parse failed',
    'parse.pages_prefix': 'parsed ',
    'parse.pages_suffix': ' pages',
    'parse.hint.parse': 'Parsing the document — large files can take a few minutes…',
    'parse.err.no_file': 'Select a file to parse.',
    'parse.err.network': 'Network error — the request did not complete.',
    'parse.stat.chars': 'text chars',
    'parse.stat.pages': 'pages',
    'parse.stat.images': 'images',
    'parse.stat.tables': 'tables',
    'parse.stat.ocr': 'OCR pages',
    'parse.stat.duration': 'total time',
    'parse.engine_status': 'Parsing engine status (warm / cache)',

    // Chunking strategies (names registered by the chunking package)
    'chunk.strategy.fixed': 'fixed windows (hard token split)',
    'chunk.strategy.paragraph': 'paragraph (paragraph-first packing)',
    'chunk.strategy.recursive': 'recursive (markdown-aware, default)',
    'chunk.strategy.semantic': 'semantic (embedding breakpoints)',
    'chunk.strategy.llm': 'LLM (smart topic boundaries)',
    'chunk.add_context': 'contextual enrichment (LLM context prefix per chunk)',
    'chunk.context_prefix': 'Context: ',

    // Ingested-chunks browser
    'chunks.page_size': 'per page',
    'chunks.doc_filter': 'doc_id (exact)',
    'chunks.name_filter': 'filename keyword',
    'chunks.query': 'query',
    'chunks.running': 'querying…',
    'chunks.reset': 'reset',
    'chunks.count': 'total chunks',
    'chunks.stat_total': 'total chunks',
    'chunks.stat_page': 'page',
    'chunks.stat_returned': 'returned',
    'chunks.no_collection': 'The ingest collection does not exist yet — ingest a document with "One-Click Ingest" first.',
    'chunks.no_match': 'No chunks match the filter.',

    // Parse profiles (PDF pipelines)
    'profile.label': 'profile',
    'profile.auto': 'auto (by document type)',
    'profile.standard': 'standard (layout + tables + selective OCR)',
    'profile.native': 'native extraction (fast, digital PDF)',
    'profile.vlm': 'vision-language model (VLM)',
    'profile.desc.auto': 'Picks automatically by document type: digital PDFs with a text layer use fast native extraction; scanned/mixed PDFs and images use layout analysis + OCR; Word / PPT / HTML use the model-free fast path.',
    'profile.desc.standard': 'DocLayNet layout analysis + TableFormer table recognition + selective RapidOCR: digital pages skip OCR while scanned pages are recognized automatically. Fits most PDFs and images.',
    'profile.desc.native': 'Plain text extraction via docling-parse — no model loading, near-instant; cannot read scans and table recovery is weak. Digital PDFs with a text layer only.',
    'profile.desc.vlm': 'End-to-end vision-language model (Granite-Docling-258M by default) reads whole pages directly — great for complex layouts; slower, and weights download on first use.',

    // Operations: queue / reindex / consistency / evaluation
    'cat.ops': 'Operations',
    'nav.Operations': 'Operations',
    'nav.queue': 'Job Queue',
    'nav.reindex': 'Index Rebuild',
    'nav.consistency': 'Consistency',
    'nav.eval': 'Evaluation & Gates',
    'common.apply': 'Apply',
    'common.back': 'Back',
    'common.close': 'Close',
    'common.next': 'Next',
    'common.prev': 'Prev',
    'ops.queue.all': 'All',
    'ops.queue.attempts': 'Attempts',
    'ops.queue.cancel': 'Cancel job',
    'ops.queue.cancel_confirm': 'Cancel this job?',
    'ops.queue.cancel_title': 'Cancel job',
    'ops.queue.cancel_requested': 'Cancel requested',
    'ops.queue.chunk_count': 'Chunks',
    'ops.queue.created': 'Created',
    'ops.queue.detail': 'Job detail',
    'ops.queue.filename': 'Filename',
    'ops.queue.finished': 'Finished',
    'ops.queue.job_id': 'Job ID',
    'ops.queue.page_count': 'Pages',
    'ops.queue.progress': 'Progress',
    'ops.queue.stage': 'Stage',
    'ops.queue.status': 'Status',
    'ops.queue.tokens_used': 'Tokens used',
    'ops.queue.updated': 'Updated',
    'ops.reindex.canary_hint': 'Canary percentage: the new build serves this share of traffic first; promote it fully after gates pass.',
    'ops.reindex.confirm': 'Submit this rebuild job?',
    'ops.reindex.embed_hint': 'Leave the embedding model empty to keep the current binding; a different model builds a brand-new vector index.',
    'ops.reindex.embed_keep': 'Empty: keep current model',
    'ops.reindex.gate': 'Release gate',
    'ops.reindex.job': 'Rebuild job',
    'ops.reindex.no_check': 'No gate check yet.',
    'ops.reindex.no_gate': 'No gate configured',
    'ops.reindex.no_gate_hint': 'You can promote the canary directly; configuring a regression gate first is recommended.',
    'ops.reindex.promote': 'Promote canary',
    'ops.reindex.promote_confirm': 'Promote the canary index to active?',
    'ops.reindex.promote_title': 'Promote canary index',
    'ops.reindex.promote_hint': 'Promotion runs in one transaction: the canary ref becomes active; the old physical index is later cleaned up by maintenance.',
    'ops.reindex.block_no_check': 'this gate has no evaluation check yet; run one before promoting.',
    'ops.reindex.block_check_failed': 'the latest gate check is "{status}"; it must pass before promotion.',
    'ops.reindex.gate_scope_note': 'a gate is unique per database + collection; the server reads exactly this row when promoting.',
    'ops.reindex.submit': 'Submit rebuild',
    'ops.reindex.submitting': 'submitting…',
    'ops.reindex.promoting': 'promoting…',
    'ops.reindex.refreshing': 'refreshing…',
    'ops.consistency.missing': 'Missing in Milvus',
    'ops.consistency.orphans': 'Orphans in Milvus',
    'ops.consistency.repair': 'Repair',
    'ops.consistency.repair_confirm': 'Repair differences against the SQLite source of truth? (rewrite missing, delete orphans)',
    'ops.consistency.repair_title': 'Repair consistency drift',
    'ops.consistency.repair_hint': 'Repair takes SQLite as truth: rewrite missing vectors and delete orphan vectors.',
    'ops.consistency.report': 'Consistency report',
    'ops.consistency.scan': 'Run scan',
    'ops.consistency.scanning': 'scanning…',
    'ops.consistency.repairing': 'repairing…',
    'ops.consistency.show_ids': 'Show IDs',
    'ops.eval.all_gates': 'All gates',
    'ops.eval.all_sets': 'All sets',
    'ops.eval.answer': 'Answer',
    'ops.eval.channel_attribution': 'Channel attribution',
    'ops.eval.checks': 'Checks',
    'ops.eval.chunks': 'Hit chunks',
    'ops.eval.expected': 'Expected',
    'ops.eval.gate': 'Gate',
    'ops.eval.has_answer': 'with gold answer',
    'ops.eval.max_drop': 'Max drop',
    'ops.eval.min': 'Min',
    'ops.eval.per_question': 'Per-question detail',
    'ops.eval.question': 'Question',
    'ops.eval.questions': 'Questions',
    'ops.eval.rerank_compare': 'Rerank before/after',
    'ops.eval.run': 'Run',
    'ops.eval.runs': 'Runs',
    'ops.eval.set_name': 'Eval set',
    'ops.eval.show_answer': 'Show answer',
    'ops.eval.tab_gates': 'Gates',
    'ops.eval.tab_sets': 'Sets',
    'ops.eval.thresholds': 'Thresholds',
    'ops.eval.versions': 'Versions',

    // Shared field labels across ops panels
    'common.total': 'total',
    'common.offset': 'offset',
    'common.scope': 'scope',
    'common.error': 'error',
    'common.params': 'params',
    'common.tag': 'tag',
    'common.question_count': 'questions',
    'common.doc_id': 'Doc ID',
    'common.job_id': 'Job ID',
    'common.set_id': 'Set ID',
    'common.version_id': 'Version ID',
    'common.run_id': 'Run ID',
    'common.gate_id': 'Gate ID',
    'common.check_id': 'Check ID',
    'common.question_id': 'Question ID',
    'common.candidate_ref': 'Candidate ref',
    'common.baseline_run_id': 'Baseline run ID',
    'common.baseline': 'Baseline',
    'common.template': 'Template',
    'common.include_answer': 'Include answer',
    'common.canary': 'canary',
    'common.active': 'active',
    'common.chunks_unit': 'chunks',
    'common.docs_unit': 'docs',
    'common.promoted': 'promoted',
    'common.retired': 'retired',

    // Job / run / check status words
    'ops.status.queued': 'queued',
    'ops.status.parsing': 'parsing',
    'ops.status.chunking': 'chunking',
    'ops.status.embedding': 'embedding',
    'ops.status.upserting': 'upserting',
    'ops.status.extracting': 'extracting',
    'ops.status.graphing': 'graphing',
    'ops.status.evaluating': 'evaluating',
    'ops.status.running': 'running',
    'ops.status.done': 'done',
    'ops.status.failed': 'failed',
    'ops.status.cancelled': 'cancelled',
    'ops.status.passed': 'passed',

    // Consistency extras
    'ops.consistency.overall': 'overall',
    'ops.consistency.physical_indexes': 'physical indexes',
    'ops.consistency.sqlite_leaves': 'SQLite leaves',
    'ops.consistency.source_truth': 'source of truth',
    'ops.consistency.milvus_rows': 'Milvus rows',
    'ops.consistency.repaired': 'repaired',
    'ops.consistency.drift': 'DRIFT',

    // Rebuild form labels
    'ops.reindex.embed_model': 'embed_model',
    'ops.reindex.canary_percent': 'canary_percent',
    'ops.reindex.batch_size': 'batch_size',

    // Eval KPIs / tables
    'ops.eval.questions_kpi': 'questions',
    'ops.eval.mean_recall': 'mean recall',
    'ops.eval.mean_mrr': 'mean MRR',
    'ops.eval.mean_ndcg': 'mean nDCG',
    'ops.eval.doc_hit_rate': 'doc hit rate',
    'ops.eval.pre': 'pre',
    'ops.eval.post': 'post',
    'ops.eval.doc_label': 'doc',
    'ops.eval.channel': 'channel',
    'ops.eval.hit_share': 'hit share',
    'ops.eval.raw_recall': 'raw recall',

    // Panel strings that used to be hard-coded in the components — see
    // the matching block in `zh` for the grouping rationale.
    'common.on': 'on',
    'common.off': 'off',
    'common.unknown': 'unknown',
    'common.first': 'first',
    'common.last': 'last',
    'common.query': 'query',
    'common.reset': 'reset',
    'common.seconds': 's',
    'common.minutes': 'min',
    'common.no_instance': 'no instance',
    'common.mime': 'MIME',
    'common.primary_field': 'primary key field',
    'common.vector_field': 'vector field',
    'common.last_op': 'last operation: ',
    'common.status.ok': 'ok',
    'common.status.error': 'failed',

    // Operation feedback (S2): labels shared by the inline status banner,
    // the four-state empty block and the confirmation modal.
    'common.retry': 'retry',
    'common.confirm': 'confirm',
    'common.confirm_title': 'confirm action',
    'common.impact': 'impact',
    'common.rows': 'rows',
    'common.deleting': 'deleting…',
    'common.load_failed': 'load failed: ',
    'common.state.idle': 'not loaded yet',
    'common.state.loading': 'loading…',
    'common.state.empty': 'no data',
    'common.state.error': 'failed to load',

    'embeddings.kind.text': 'text',
    'embeddings.kind.image': 'image',
    'embeddings.kind.multimodal': 'multimodal',
    'embeddings.title.text': 'text embeddings',
    'embeddings.title.image': 'image embeddings',
    'embeddings.title.multimodal': 'multimodal embeddings',
    'embeddings.no_models': 'no {kind} models.',
    'embeddings.input_mode': 'input mode',
    'embeddings.mode.single': 'single',
    'embeddings.mode.list': 'list',
    'embeddings.input_text': 'input text',
    'embeddings.list_json_array': 'list (JSON array)',
    'embeddings.select_images': 'select images',
    'embeddings.hint.local_files': 'local files → auto base64',
    'embeddings.json_list': 'JSON list',
    'embeddings.hint.each_item': 'each item: data / mime',
    'embeddings.err.no_input': 'select files or paste a JSON array.',
    'embeddings.err.no_items': 'add at least one text or image item.',
    'embeddings.err.bad_list': 'the batch JSON is not an array; fix it and retry.',
    'embeddings.running': 'running…',

    'rerank.title': 'cross-encoder rerank',
    'rerank.registered': 'registered rerankers',
    'rerank.no_models': 'no reranker models.',
    'rerank.suffix': 'reranker',
    'rerank.input_mode': 'input mode',
    'rerank.mode.lines': 'one per line',
    'rerank.mode.json': 'JSON array',
    'rerank.documents': 'documents',
    'rerank.run': 'rerank',
    'rerank.results': 'results · hits {n}',
    'rerank.err.json': 'JSON mode requires an array of strings.',
    'rerank.running': 'reranking…',

    'similarity.title.text': 'text similarity',
    'similarity.title.image': 'image similarity',
    'similarity.title.multimodal': 'multimodal similarity',
    'similarity.mode.text': 'text',
    'similarity.mode.image': 'image',
    'similarity.mode.multimodal': 'multimodal',
    'similarity.query_text': 'query text',
    'similarity.candidate_docs': 'candidate documents',
    'similarity.hint.one_per_line': 'one per line',
    'similarity.query_image': 'query image',
    'similarity.candidate_images': 'candidate images',
    'similarity.hint.pick_several': 'pick several at once',
    'similarity.query_modality': 'query modality',
    'similarity.modality.text': 'text',
    'similarity.modality.image': 'image',
    'similarity.candidate_texts': 'candidate texts',
    'similarity.err.query_empty': 'query text must not be empty.',
    'similarity.err.no_docs': 'add at least one candidate document.',
    'similarity.err.no_query_image': 'select a query image.',
    'similarity.err.no_doc_images': 'select at least one candidate image.',
    'similarity.err.no_candidates': 'add at least one text or image candidate.',
    'similarity.running': 'running…',

    'search.title': 'vector search',
    'search.query_mode': 'query mode',
    'search.mode.text': 'server-side embed (query_text)',
    'search.mode.vector': 'direct vector (query_vector)',
    'search.query_text': 'query text',
    'search.query_vector': 'query vector',
    'search.filter_title': 'filter expression (Milvus native)',
    'search.output_fields': 'output fields',
    'search.hint.click_toggle': 'click to toggle',
    'search.err.bad_vector': 'query_vector must be a JSON array.',
    'search.err.no_selection': 'select database and collection.',
    'search.err.failed': 'search failed: ',
    'search.run': 'search',
    'search.no_hits': 'no matching rows.',
    'search.running': 'searching…',
    'search.idle_hint': 'set the query and hit search.',
    'search.score': 'raw score',
    'search.metric_higher': 'higher is closer',
    'search.metric_lower': 'lower is closer',

    'browse.title': 'Select database / collection',
    'browse.sub': 'paginated browse of collection rows',
    'browse.query_params': 'query params',
    'browse.query_params_sub': 'sorted by primary key asc; vector field always hidden',
    'browse.page_size': 'page size',
    'browse.filter_expr': 'filter expression',
    'browse.output_fields': 'output fields',
    'browse.hint.readonly': 'read-only',
    'browse.rows_range': '{from} - {to} / {total} rows',
    'browse.empty_summary': 'empty',
    'browse.stat_total': 'total rows',
    'browse.stat_page': 'page',
    'browse.stat_returned': 'returned',
    'browse.delete_selected': 'delete selected',
    'browse.delete_by_filter': 'delete by filter',
    'browse.clear_selection': 'clear selection',
    'browse.delete_hint': 'filter delete hits ALL matching rows collection-wide, not only this page',
    'browse.deleted_one': 'deleted 1 row.',
    'browse.deleted_n': 'deleted {n} rows.',
    'browse.delete_failed': 'delete failed: ',
    'browse.confirm_delete_one': 'Delete the selected row by primary key? This cannot be undone.',
    'browse.confirm_delete_n': 'Delete {n} selected rows by primary key? This cannot be undone.',
    'browse.confirm_delete_filter': 'Delete ALL rows matching this filter from the whole collection?\n\n{expr}\n\nThis cannot be undone.',
    'browse.err.no_filter': 'Please fill in the filter expression first.',
    'browse.empty': 'collection is empty or filter matches nothing.',
    'browse.select_all': 'select / unselect all rows on this page',
    'browse.select_row': 'select {id}',
    'browse.pager_info': 'rows {from}-{to} of {total} / page {page} of {pages}',
    'browse.jump_to': 'jump to',
    'browse.go': 'go',
    'browse.footer_hint': 'hover a cell to see the full JSON value; tick rows across pages then "delete selected", or delete every match collection-wide with "delete by filter"; pager hides when the collection is empty.',
    'browse.running': 'querying…',
    'browse.idle_hint': 'pick a collection and hit query.',
    'browse.confirm_delete_title': 'Delete rows',

    'collections.select_db': 'Select database',
    'collections.current_db': 'current database',
    'collections.no_dbs': '(no databases)',
    'collections.reload_dbs': 'reload database list',
    'collections.list_title': 'Collection list',
    'collections.new': '+ new collection',
    'collections.pick_db_first': 'select a database first.',
    'collections.empty': 'no collections in this database.',
    'collections.loading_detail': 'loading collection detail...',
    'collections.stat.dim': 'dimension',
    'collections.stat.count': 'rows',
    'collections.stat.metric': 'metric',
    'collections.stat.fields': 'primary / vector field',
    'collections.dim_value': 'dim {dim}',
    'collections.fields_n': 'fields ({n})',
    'collections.primary': 'primary',
    'collections.indexes_n': 'indexes ({n})',
    'collections.new_index': '+ new index',
    'collections.target_field': 'target field',
    'collections.vector_field_suffix': '(vector field)',
    'collections.index_type': 'index type',
    'collections.params_hint': 'JSON, e.g. M=16, efConstruction=200',
    'collections.create_rebuild': 'Create / rebuild',
    'collections.confirm_drop': 'Drop collection "{name}"?',
    'collections.confirm_drop_index': 'Drop the index on field "{field}"?',
    'collections.drop_index_failed': 'drop index failed: ',
    'collections.create_index_failed': 'create index failed: ',
    'collections.err.bad_params': 'params must be a valid JSON object.',
    'collections.confirm_drop_title': 'Drop collection',
    'collections.confirm_drop_index_title': 'Drop index',
    'collections.list_failed': 'failed to load the collection list: ',
    'collections.detail_failed': 'failed to load collection detail: ',
    'collections.creating': 'creating…',
    'collections.dropped': 'dropped collection {name}',
    'collections.index_dropped': 'dropped index {field}',
    'collections.index_created': 'created index {field}',
    'collections.index_field': 'index field',

    'records.title': 'Write records',
    'records.input_mode': 'input mode',
    'records.mode.texts': 'server-side embed (texts)',
    'records.mode.vectors': 'direct vectors',
    'records.ids': 'primary keys (ids)',
    'records.texts': 'texts to embed (texts)',
    'records.vectors': 'precomputed vectors (vectors)',
    'records.fields': 'extra fields (fields, row-aligned with ids)',
    'records.field_set': 'fields',
    'records.delete_params': 'delete params',
    'records.delete_params_hint': 'used by the "delete by filter" button below',
    'records.delete_mode': 'delete mode',
    'records.del_mode.ids': 'by primary key list',
    'records.del_mode.filter': 'by filter expression',
    'records.del_ids': 'primary keys',
    'records.del_filter': 'filter expression',
    'records.upsert': 'write (PUT)',
    'records.fetch': 'fetch by id (POST)',
    'records.delete': 'delete by filter (POST)',
    'records.ph.ids': '["id-1", "id-2"]',
    'records.ph.texts': '["first text", "second text"]',
    'records.ph.vectors': '[[0.1, 0.2, 0.3]]',
    'records.ph.fields': '[{"category": "mouse"}]',
    'records.ph.del_ids': '["id-1", "id-2"]',
    'records.ph.del_filter': 'category == \'mouse\'',
    'records.err.ids_json': 'ids must be a JSON array.',
    'records.err.texts_json': 'texts must be a JSON array.',
    'records.err.vectors_json': 'vectors must be a JSON array.',
    'records.err.fields_json': 'fields must be a JSON array.',
    'records.err.no_selection': 'select a database and collection.',
    'records.err.no_ids': 'enter at least one primary key.',
    'records.err.no_filter': 'enter a filter expression.',
    'records.confirm_delete': 'Delete? This cannot be undone.',
    'records.confirm_delete_title': 'Delete vectors',
    'records.upserting': 'upserting…',
    'records.fetching': 'fetching…',
    'records.deleting': 'deleting…',
    'records.fetched_n': 'fetched {n} record(s)',
    'records.fetched_missing': '{n} primary key(s) not found',
    'records.upsert_ok': 'wrote {n} row(s)',
    'records.delete_ok': 'deleted {n} row(s)',
    'records.vector_dims': '{n} dimensions',
    'records.no_vector': 'no vector returned',

    'modals.err.db_name_required': 'database name is required.',
    'modals.err.create_failed': 'create failed',
    'modals.err.one_primary': 'exactly one primary key (is_primary) required. current: {n}',
    'modals.err.primary_varchar': 'primary must be varchar. current: {dtype}',
    'modals.err.primary_name_required': 'primary key field name is required.',
    'modals.err.varchar_max_length': 'varchar field {name} needs max_length.',
    'modals.err.field_name_required': 'every field must have a name.',
    'modals.err.primary_mismatch': 'primary field ({name}) does not match is_primary field.',
    'modals.err.dim': 'vector dim must be >= 1.',
    'modals.err.no_db': 'create a database first.',
    'modals.err.coll_name_required': 'collection name is required.',
    'modals.preset_hnsw_cosine': 'preset: HNSW + cosine',
    'modals.preset_hnsw_cosine_desc': 'in-memory graph index, high recall, good for small/medium sets',
    'modals.preset_ivf_l2': 'preset: IVF_FLAT + l2',
    'modals.preset_ivf_l2_desc': 'inverted-file clustering, Euclidean distance',
    'modals.preset_diskann_ip': 'preset: DISKANN + ip',
    'modals.preset_diskann_ip_desc': 'on-disk index, for very large collections',
    'modals.preset_use_default': 'preset: default (HNSW + cosine)',
    'modals.preset_use_default_desc': 'M=16, efConstruction=200',

    'retrieval.query': 'query',
    'retrieval.query_placeholder': 'Enter your query',
    'retrieval.custom_json': 'request body (JSON)',
    'retrieval.custom_hint': 'sent verbatim; the controls above are ignored',
    'retrieval.custom_regen': 'rebuild from current controls',
    'retrieval.err.bad_json': 'request body must be a JSON object: ',
    'retrieval.group.channels': 'Channels & fusion',
    'retrieval.fusion': 'fusion',
    'retrieval.group.routing': 'Intent routing',
    'retrieval.auto_route': 'auto-route channels',
    'retrieval.use_llm': 'use LLM classifier',
    'retrieval.group.rewrite': 'Query rewrite',
    'retrieval.rewrite_enabled': 'enabled',
    'retrieval.method.hyde': 'HyDE',
    'retrieval.method.multi_query': 'multi-query',
    'retrieval.method.step_back': 'step-back',
    'retrieval.method.decompose': 'decompose',
    'retrieval.n_variants': 'n variants',
    'retrieval.group.diversity': 'Diversity & rerank',
    'retrieval.cross_encoder': 'cross-encoder rerank',
    'retrieval.candidate_pool': 'candidate pool',
    'retrieval.group.context': 'Context budget & metadata filter',
    'retrieval.max_tokens': 'max tokens',
    'retrieval.max_tokens_ph': 'trim to a token budget',
    'retrieval.doc_id_ph': 'exact match',
    'retrieval.filename_ph': 'fuzzy match (SQLite)',
    'retrieval.no_hits': 'no matching chunks.',
    'retrieval.chunk_label': 'chunk',
    'retrieval.chars': 'chars',
    'retrieval.sub_queries': 'sub-queries',
    'retrieval.top': 'top',
    'retrieval.stage.route': 'routing',
    'retrieval.stage.rewrite': 'rewrite',
    'retrieval.stage.recall': 'recall',
    'retrieval.stage.fuse': 'fusion',
    'retrieval.stage.mmr': 'MMR',
    'retrieval.stage.rerank': 'rerank',
    'retrieval.stage.expand': 'expand',
    'retrieval.stage.compress': 'compress',

    'kb.parse_title': 'parse',
    'kb.chunk_title': 'chunk',
    'kb.ingest_title': 'ingest',
    'kb.chunks_title': 'chunks',
    'kb.file': 'file',
    'kb.strategy': 'strategy',
    'kb.breakpoint': 'breakpoint %',
    'kb.chunk_size': 'chunk size',
    'kb.overlap': 'overlap',
    'kb.markdown': 'markdown text',
    'kb.chunk_btn': 'chunk',
    'kb.chunks_n': 'chunks: {n}',
    'kb.chars_tokens': '{chars} chars · {tokens} tokens',
    'kb.page': 'p.{n}',
    'kb.embed_model': 'embed model',
    'kb.source': 'source',
    'kb.preview': 'preview',
    'kb.copy': 'copy',
    'kb.copied': 'copied',
    'kb.expand_full': 'show full text',
    'kb.collapse': 'collapse',
    'kb.download_md': 'download .md',
    'kb.metadata': 'metadata (JSON)',
    'kb.pager_info': '{from}-{to} / {total} · page {page} of {pages}',
    'kb.chunking': 'chunking…',
    'kb.dbs_failed': 'failed to load the database list: ',
    'kb.models_failed': 'failed to load the model list: ',
    'kb.parser_failed': 'failed to load parser engine status: ',
    'kb.cancelled': 'cancelled.',
    'kb.err.timeout': 'the stream sent nothing for {seconds}s; aborted.',
    'models.poll_failed': 'failed to refresh the model list: ',
  },
};

// Keys already reported as missing, so the dev-time warning fires
// once per key instead of once per render.
const _warnedKeys = Object.create(null);

// Translate `key` for the active locale. An unknown key renders as the
// key itself (never blank) and warns once in the console. `vars`
// substitutes `{name}` placeholders — plain string replace, no
// expression evaluation.
export function t(key, vars) {
  const dict = I18N[store.locale] || I18N.zh;
  let s = dict[key];
  if (s === undefined) {
    if (!_warnedKeys[key]) {
      _warnedKeys[key] = true;
      console.warn('[i18n] missing key:', key, 'locale:', store.locale);
    }
    s = key;
  }
  if (vars) {
    for (const k of Object.keys(vars)) {
      s = s.split('{' + k + '}').join(String(vars[k]));
    }
  }
  return s;
}

export async function api(method, path, body) {
  const init = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) init.body = JSON.stringify(body);
  const resp = await fetch(path, init);
  let payload = null;
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    try { payload = await resp.json(); } catch (_e) { payload = null; }
  } else {
    payload = await resp.text().catch(() => null);
  }
  if (!resp.ok) {
    const msg = (payload && payload.error && payload.error.message) || 'HTTP ' + resp.status;
    throw new Error(msg);
  }
  return { payload, status: resp.status };
}

export function enc(s) { return encodeURIComponent(s); }
export function extractApiError(e, fallback) {
  if (!e) return fallback || '请求失败';
  if (typeof e === 'string') return e;
  if (e && e.message) return e.message;
  return String(e);
}

// One alert per id+error message: while polling repeats the same
// `failed` row every few seconds we must not re-pop the alert. Keyed by
// model id; cleared when that id leaves the failed state (retry / load
// success / unload).
const alertedLoadError = Object.create(null);

// Reconcile a fresh GET /v1/models payload into the store:
//  - drives per-card optimistic busy flags (busy[id] holds the verb of
//    the request WE fired: 'load' | 'unload'), clearing each only after
//    the server-polled state settles for THAT verb,
//  - alerts exactly once on a loading -> failed transition,
//  - keeps store.embedderDim in sync (previously only the poller did),
//  - returns true while at least one row is still loading.
export function applyModelsData(rows) {
  const prev = new Map((store.models.data || []).map(m => [m.id, m]));
  store.models.data = rows || [];
  let anyLoading = false;
  for (const m of store.models.data) {
    const before = prev.get(m.id);
    // The busy verb drives OPTIMISTIC UI: the card switches to its
    // loading state the instant the button is clicked, before the POST
    // (or the confirming GET) has returned. It must survive a stale poll
    // — e.g. a GET fired just before the click that comes back showing
    // the old unloaded row must not flip the card back mid-request.
    const busyVerb = store.models.busy[m.id];
    if (m.load_status === 'loading' || busyVerb === 'load') anyLoading = true;

    if (m.load_status === 'failed') {
      // Announce when WE watched this id fail: either the polled row just
      // transitioned out of loading, or our own optimistic load is in
      // flight and the server already reports failure. Opening the
      // dashboard on a stale failure shows the inline error, no notice.
      // The notice strip (not alert()) because this lands while the
      // operator may be on any view — the model card itself carries the
      // detail, the strip is what makes it visible from elsewhere.
      const watched =
        (before && before.load_status === 'loading') || busyVerb === 'load';
      if (watched && alertedLoadError[m.id] !== m.load_error) {
        alertedLoadError[m.id] = m.load_error;
        notify('error', t('models.load_failed') + ': ' + (m.load_error || t('common.unknown')));
      }
    } else if (alertedLoadError[m.id] !== undefined) {
      // Any non-failed row (incl. 'loading' during a retry) rearms the
      // one-shot alert for the next failure.
      delete alertedLoadError[m.id];
    }

    if (busyVerb === 'load') {
      // Settle once the server confirms the instance (or the failure);
      // 'unloaded' while our POST is in flight is a stale poll.
      if (m.loaded || m.load_status === 'loaded' || m.load_status === 'failed') {
        store.models.busy[m.id] = false;
      }
    } else if (busyVerb === 'unload') {
      // Settle once the live instance is gone. A 'loading' row means
      // someone else started loading this family — also release.
      if (!m.loaded && m.load_status !== 'loading') {
        store.models.busy[m.id] = false;
      }
    }
  }
  const loaded = store.models.data.find(m => m.type === 'embedder' && m.loaded);
  store.embedderDim = loaded && loaded.dimensions ? loaded.dimensions : 0;
  return anyLoading;
}

export const store = reactive({
  view: 'overview',
  locale: 'zh',                                 // current UI language: 'zh' | 'en'
  health: { healthz: 'unknown', readyz: 'unknown' },
  // Service identity for the statusbar, filled by the same 5s tick as the
  // health LEDs. Empty until the first successful GET /v1/system/status;
  // the statusbar renders '—' instead of a hardcoded literal so a version
  // bump (or a reconfigured backend) shows up with no code edit.
  service: { version: '', embeddingBackend: '', storeBackend: '' },
  embedderDim: 0,
  models: { data: [], busy: Object.create(null), autoRefresh: true },
  databases: { list: [], detailCache: Object.create(null), expanded: new Set() },
  collections: { list: [], detailCache: Object.create(null), expanded: new Set() },
  modals: { newDb: false, newColl: false },
  modalErr: { newDb: '', newColl: '' },
});

let healthTimer = null;
async function pollHealth() {
  try {
    const r = await fetch('/healthz');
    store.health.healthz = r.ok ? 'ok' : 'err';
  } catch (_e) { store.health.healthz = 'err'; }
  try {
    const r = await fetch('/readyz');
    store.health.readyz = r.ok ? 'ok' : 'err';
  } catch (_e) { store.health.readyz = 'err'; }
  // The statusbar's service identity (version + configured backends) comes
  // from the API rather than literals baked into the template. Fail-open:
  // keep the last good values. No banner on failure by design — this hits
  // the same app as the two probes above, so an unreachable service has
  // already turned both LEDs red, and a notice here would just be a second
  // copy of that signal on a strip the operator has to dismiss.
  try {
    const { payload } = await api('GET', '/v1/system/status');
    const svc = (payload && payload.service) || {};
    if (svc.version) store.service.version = svc.version;
    if (svc.embedding_backend) store.service.embeddingBackend = svc.embedding_backend;
    if (svc.vector_store_backend) store.service.storeBackend = svc.vector_store_backend;
  } catch (_e) { /* see above: the LEDs carry this poll's reachability signal */ }
}
function startHealthPolling() {
  if (healthTimer) return;
  healthTimer = setInterval(pollHealth, 5000);
  pollHealth();
}
// Adaptive poller: 5s cadence at rest, tightened to 2s while any card
// reports load_status='loading' so a background load completes visually
// within ~2s instead of waiting out a full 5s tick. Self-rescheduling
// setTimeout chain (not setInterval) so the delay can change per tick.
let modelsTimer = null;
let modelsPollStopped = false;
// The poller runs whatever view the operator is on, so a failure has no
// panel to land in. Report the *transition* into failing (once) on the
// app-level notice strip and clear it on recovery — notifying on every
// 5s tick would re-arm a notice the operator just dismissed.
let modelsPollFailing = false;
let modelsPollNotice = '';
function startModelsAutoRefresh() {
  if (modelsTimer !== null) return;
  modelsPollStopped = false;
  const schedule = (delay) => {
    modelsTimer = setTimeout(async () => {
      modelsTimer = null;
      if (modelsPollStopped) return;
      let anyLoading = false;
      if (store.models.autoRefresh) {
        try {
          const { payload } = await api('GET', '/v1/models');
          anyLoading = applyModelsData((payload && payload.data) || []);
          if (modelsPollFailing) {
            modelsPollFailing = false;
            // Only retract our own notice: the operator may have raised
            // a different one while this poll was failing.
            if (noticeState.text === modelsPollNotice) dismissNotice();
            modelsPollNotice = '';
          }
        } catch (e) {
          if (!modelsPollFailing) {
            modelsPollFailing = true;
            modelsPollNotice = t('models.poll_failed')
              + extractApiError(e, t('common.unknown'));
            notify('error', modelsPollNotice);
          }
        }
      }
      if (!modelsPollStopped) schedule(anyLoading ? 2000 : 5000);
    }, delay);
  };
  schedule(5000);
}
function stopModelsAutoRefresh() {
  modelsPollStopped = true;
  if (modelsTimer !== null) {
    clearTimeout(modelsTimer);
    modelsTimer = null;
  }
}

const NAV_LABELS = {
  'overview':            { catKey: 'cat.overview',  subKey: 'nav.overview' },
  'models':              { catKey: 'cat.models',    subKey: 'nav.models' },
  'embeddings':          { catKey: 'cat.models',    subKey: 'nav.embeddings' },
  'image-embeddings':    { catKey: 'cat.models',    subKey: 'nav.image_embeddings' },
  'multimodal-embeddings':{ catKey: 'cat.models',   subKey: 'nav.multimodal_embeddings' },
  'rerank':              { catKey: 'cat.models',    subKey: 'nav.rerank' },
  'similarity':          { catKey: 'cat.models',    subKey: 'nav.similarity' },
  'databases':           { catKey: 'cat.store',     subKey: 'nav.databases' },
  'collections':         { catKey: 'cat.store',     subKey: 'nav.collections' },
  'records':             { catKey: 'cat.store',     subKey: 'nav.records' },
  'search':              { catKey: 'cat.store',     subKey: 'nav.search' },
  'browse':              { catKey: 'cat.store',     subKey: 'nav.browse' },
  'parse':               { catKey: 'cat.kb',        subKey: 'nav.parse' },
  'chunk':               { catKey: 'cat.kb',        subKey: 'nav.chunk' },
  'ingest':              { catKey: 'cat.kb',        subKey: 'nav.ingest' },
  'ingested':            { catKey: 'cat.kb',        subKey: 'nav.chunks_view' },
  'retrieval':           { catKey: 'cat.kb',        subKey: 'nav.retrieval' },
  'queue':               { catKey: 'cat.ops',       subKey: 'nav.queue' },
  'reindex':             { catKey: 'cat.ops',       subKey: 'nav.reindex' },
  'consistency':         { catKey: 'cat.ops',       subKey: 'nav.consistency' },
  'eval':                { catKey: 'cat.ops',       subKey: 'nav.eval' },
};

const App = defineComponent({
  name: 'DashboardApp',
  components: {
    OverviewPanel, ModelsPanel, DatabasesPanel, CollectionsPanel,
    RecordsPanel, SearchPanel, BrowsePanel, RerankPanel,
    EmbeddingsPanel, SimilarityPanel, KnowledgeBasePanel, RetrievalPanel,
    OpsQueuePanel, OpsReindexPanel, OpsConsistencyPanel, OpsEvalPanel,
    NewDbModal, NewCollModal,
    NoticeBar, ConfirmHost,
  },
  setup() {
    onMounted(() => {
      startHealthPolling();
      startModelsAutoRefresh();
    });
    onUnmounted(() => {
      if (healthTimer) clearInterval(healthTimer);
      stopModelsAutoRefresh();
    });
    return { store, t };
  },
  template: `
    <div class="app">

      <header class="topbar">
        <div class="brand">
          <span class="brand-glyph">
            <svg viewBox="0 0 32 32" fill="none">
              <circle cx="10" cy="16" r="2" fill="#0891B2"/>
              <circle cx="16" cy="12" r="2" fill="#0891B2" opacity=".7"/>
              <circle cx="16" cy="20" r="2" fill="#0891B2" opacity=".7"/>
              <circle cx="22" cy="16" r="2" fill="#0891B2" opacity=".4"/>
            </svg>
          </span>
          <span class="brand-name">vector-service</span>
          <span class="brand-sep">·</span>
          <span class="brand-tag">{{ t('brand.dashboard') }}</span>
        </div>
        <div class="signature">
          <div class="dim-dots" id="dim-dots" aria-label="dim-dots">
            <span v-for="i in store.embedderDim" :key="'a'+i" class="dim-dot dim-dot--accent"></span>
            <span v-for="i in Math.max(16 - store.embedderDim, 0)" :key="'m'+i" class="dim-dot"></span>
          </div>
          <span class="dim-dots-label"><span class="num">{{ store.embedderDim || '—' }}</span> {{ t('topbar.dim_unit') }}</span>
        </div>
        <div class="topbar-spacer"></div>
        <div class="indicators">
          <div :class="'led ' + (store.health.healthz === 'ok' ? 'ok' : 'err')" id="led-healthz"><span>healthz</span></div>
          <div :class="'led ' + (store.health.readyz === 'ok' ? 'ok' : 'err')" id="led-readyz"><span>readyz</span></div>
        </div>
        <div class="topbar-actions">
          <button class="lang-btn" @click="store.locale = store.locale === 'zh' ? 'en' : 'zh'" :title="store.locale === 'zh' ? 'Switch to English' : '切换中文'">
            {{ store.locale === 'zh' ? 'EN' : '中' }}
          </button>
        </div>
      </header>

      <nav class="sidebar" aria-label="main nav">
        <div class="sidebar-head">{{ t('nav.Navigation') }}</div>
        <div class="nav">
          <div class="nav-group" data-group="overview">
            <div class="nav-group-label"><span>{{ t('nav.Overview') }}</span></div>
            <div class="nav-items">
              <div :class="['nav-item', store.view === 'overview' ? 'active' : '']" data-view="overview" @click="store.view = 'overview'">
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
                  <rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/>
                  <rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/>
                </svg>
                <span>{{ t('nav.overview') }}</span>
              </div>
            </div>
          </div>

          <div class="nav-group" data-group="models">
            <div class="nav-group-label"><span>{{ t('nav.Models') }}</span></div>
            <div class="nav-items">
              <div :class="['nav-item', store.view === 'models' ? 'active' : '']" data-view="models" @click="store.view = 'models'"><span>{{ t('nav.models') }}</span></div>
              <div :class="['nav-item', store.view === 'embeddings' ? 'active' : '']" data-view="embeddings" @click="store.view = 'embeddings'"><span>{{ t('nav.embeddings') }}</span></div>
              <div :class="['nav-item', store.view === 'image-embeddings' ? 'active' : '']" data-view="image-embeddings" @click="store.view = 'image-embeddings'"><span>{{ t('nav.image_embeddings') }}</span></div>
              <div :class="['nav-item', store.view === 'multimodal-embeddings' ? 'active' : '']" data-view="multimodal-embeddings" @click="store.view = 'multimodal-embeddings'"><span>{{ t('nav.multimodal_embeddings') }}</span></div>
              <div :class="['nav-item', store.view === 'rerank' ? 'active' : '']" data-view="rerank" @click="store.view = 'rerank'"><span>{{ t('nav.rerank') }}</span></div>
              <div :class="['nav-item', store.view === 'similarity' ? 'active' : '']" data-view="similarity" @click="store.view = 'similarity'"><span>{{ t('nav.similarity') }}</span></div>
            </div>
          </div>

          <div class="nav-group" data-group="store">
            <div class="nav-group-label"><span>{{ t('nav.Vector Store') }}</span></div>
            <div class="nav-items">
              <div :class="['nav-item', store.view === 'databases' ? 'active' : '']" data-view="databases" @click="store.view = 'databases'"><span>{{ t('nav.databases') }}</span></div>
              <div :class="['nav-item', store.view === 'collections' ? 'active' : '']" data-view="collections" @click="store.view = 'collections'"><span>{{ t('nav.collections') }}</span></div>
              <div :class="['nav-item', store.view === 'records' ? 'active' : '']" data-view="records" @click="store.view = 'records'"><span>{{ t('nav.records') }}</span></div>
              <div :class="['nav-item', store.view === 'search' ? 'active' : '']" data-view="search" @click="store.view = 'search'"><span>{{ t('nav.search') }}</span></div>
              <div :class="['nav-item', store.view === 'browse' ? 'active' : '']" data-view="browse" @click="store.view = 'browse'"><span>{{ t('nav.browse') }}</span></div>
            </div>
          </div>

          <div class="nav-group" data-group="kb">
            <div class="nav-group-label"><span>{{ t('nav.Knowledge Base') }}</span></div>
            <div class="nav-items">
              <div :class="['nav-item', store.view === 'parse' ? 'active' : '']" data-view="parse" @click="store.view = 'parse'"><span>{{ t('nav.parse') }}</span></div>
              <div :class="['nav-item', store.view === 'chunk' ? 'active' : '']" data-view="chunk" @click="store.view = 'chunk'"><span>{{ t('nav.chunk') }}</span></div>
              <div :class="['nav-item', store.view === 'ingest' ? 'active' : '']" data-view="ingest" @click="store.view = 'ingest'"><span>{{ t('nav.ingest') }}</span></div>
              <div :class="['nav-item', store.view === 'ingested' ? 'active' : '']" data-view="ingested" @click="store.view = 'ingested'"><span>{{ t('nav.chunks_view') }}</span></div>
              <div :class="['nav-item', store.view === 'retrieval' ? 'active' : '']" data-view="retrieval" @click="store.view = 'retrieval'"><span>{{ t('nav.retrieval') }}</span></div>
            </div>
          </div>

          <div class="nav-group" data-group="ops">
            <div class="nav-group-label"><span>{{ t('nav.Operations') }}</span></div>
            <div class="nav-items">
              <div :class="['nav-item', store.view === 'queue' ? 'active' : '']" data-view="queue" @click="store.view = 'queue'">{{ t('nav.queue') }}</div>
              <div :class="['nav-item', store.view === 'reindex' ? 'active' : '']" data-view="reindex" @click="store.view = 'reindex'">{{ t('nav.reindex') }}</div>
              <div :class="['nav-item', store.view === 'consistency' ? 'active' : '']" data-view="consistency" @click="store.view = 'consistency'">{{ t('nav.consistency') }}</div>
              <div :class="['nav-item', store.view === 'eval' ? 'active' : '']" data-view="eval" @click="store.view = 'eval'">{{ t('nav.eval') }}</div>
            </div>
          </div>
        </div>
      </nav>

      <main class="main">
        <!-- App-level feedback for failures that belong to no panel
             (background model loads, polling that starts erroring). -->
        <notice-bar />

        <nav class="breadcrumb">
          <span>vector-service</span>
          <span class="crumb-sep">/</span>
          <span id="crumb-cat">{{ navLabel(store.view).cat }}</span>
          <span class="crumb-sep" v-if="navLabel(store.view).sub">/</span>
          <span class="crumb-current" id="crumb-sub" v-if="navLabel(store.view).sub">{{ navLabel(store.view).sub }}</span>
        </nav>

        <div class="content">
          <overview-panel v-show="store.view === 'overview'" />
          <models-panel v-show="store.view === 'models'" />
          <embeddings-panel v-show="store.view === 'embeddings'" kind="text" />
          <embeddings-panel v-show="store.view === 'image-embeddings'" kind="image" />
          <embeddings-panel v-show="store.view === 'multimodal-embeddings'" kind="multimodal" />
          <rerank-panel v-show="store.view === 'rerank'" />
          <similarity-panel v-show="store.view === 'similarity'" />
          <databases-panel v-show="store.view === 'databases'" />
          <collections-panel v-show="store.view === 'collections'" />
          <records-panel v-show="store.view === 'records'" />
          <search-panel v-show="store.view === 'search'" />
          <browse-panel v-show="store.view === 'browse'" />
          <knowledge-base-panel v-show="store.view === 'parse' || store.view === 'chunk' || store.view === 'ingest' || store.view === 'ingested'" :view="store.view" />
          <retrieval-panel v-show="store.view === 'retrieval'" />
          <ops-queue-panel v-show="store.view === 'queue'" />
          <ops-reindex-panel v-show="store.view === 'reindex'" />
          <ops-consistency-panel v-show="store.view === 'consistency'" />
          <ops-eval-panel v-show="store.view === 'eval'" />
        </div>
      </main>

      <footer class="statusbar">
        <span class="seg"><span class="key">{{ t('status.SVC') }}</span><span class="val">vector-service</span></span>
        <span class="seg"><span class="key">{{ t('status.VER') }}</span><span class="val accent">{{ store.service.version || '—' }}</span></span>
        <span class="seg"><span class="key">{{ t('status.EMB') }}</span><span class="val">{{ store.service.embeddingBackend || '—' }}</span></span>
        <span class="seg"><span class="key">{{ t('status.STORE') }}</span><span class="val">{{ store.service.storeBackend || '—' }}</span></span>
        <span class="right">
          <span class="seg"><span class="key">{{ t('status.LOADED') }}</span><span class="val">{{ loadedCount }}/{{ store.models.data.length }}</span></span>
        </span>
      </footer>

      <new-db-modal v-if="store.modals.newDb" />
      <new-coll-modal v-if="store.modals.newColl" />
      <confirm-host />
    </div>
  `,
  methods: {
    navLabel(view) {
      const m = NAV_LABELS[view];
      if (!m) return { cat: '', sub: '' };
      return { cat: t(m.catKey), sub: t(m.subKey) };
    },
  },
  computed: {
    loadedCount() { return (store.models.data || []).filter(m => m.loaded).length; },
  },
});

// Expose `t` as `$t` in templates. Component templates can then
// call `{{ $t('nav.browse') }}` without each setup() having to
// re-declare it.
const _app = createApp(App);
_app.config.globalProperties.$t = t;
_app.mount('#app');

