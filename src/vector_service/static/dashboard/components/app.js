// =====================================================================
// app.js -- root component + global reactive store + shared helpers
// =====================================================================
import { createApp, defineComponent, reactive, computed, onMounted, onUnmounted } from '../vue.esm-browser.prod.js';

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
// Operations panels (TODO §7): queue / rebuild / consistency / evaluation.
import OpsQueuePanel from './ops-queue.js';
import OpsReindexPanel from './ops-reindex.js';
import OpsConsistencyPanel from './ops-consistency.js';
import OpsEvalPanel from './ops-eval.js';
import { NewDbModal, NewCollModal } from './modals.js';

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
    'retrieval.title': '分片检索',
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
    'common.status_label': '状态:',
    'common.auto': '自动',
    'common.loaded': '已加载',
    'common.unloaded': '未加载',
    'common.no_models': '没有已注册模型.',
    'common.new_db': '+ 新建数据库',
    'common.new_collection': '+ 新建集合',
    'common.click_refresh': '点击"刷新"加载数据库.',
    'common.model_id_ops': '加载 / 卸载均按 model_id 操作',
    'common.params_label_pre': 'params ',

    // Models panel
    'models.auto_refresh_label': '自动刷新:',
    'models.load': '加载',
    'models.unload': '卸载',
    'models.loading': '加载中…',
    'models.unloading': '卸载中…',
    'models.load_failed': '加载失败',
    'models.unload_failed': '卸载失败',
    'models.meta_params': '参数量',
    'models.meta_vram': '显存占用',
    'models.meta_ram': '内存占用',
    'models.meta_load_time': '加载耗时',

    // Databases panel
    'databases.coll_count': '集合数',
    'databases.meta_count': 'metadata 字段',
    'databases.backend_meta': '后端 metadata',
    'databases.coll_list': '集合清单',
    'databases.no_collections': '该数据库下暂无集合.',
    'databases.hint': '每个集合必须归属一个数据库;删除数据库会同时删除其下所有集合.',
    'databases.loading': '正在加载数据库详情...',
    'databases.confirm_drop_pre': '确认删除数据库 "',
    'databases.and_collections': '及其下所有集合?',

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

    // Operations (TODO §7): queue / reindex / consistency / evaluation
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
    'ops.reindex.promote_hint': '提升在单事务内完成：金丝雀引用切换为正式引用，旧物理索引随后由维护任务清理。',
    'ops.reindex.submit': '提交重建',
    'ops.consistency.missing': 'Milvus 缺失',
    'ops.consistency.orphans': 'Milvus 孤儿',
    'ops.consistency.repair': '修复',
    'ops.consistency.repair_confirm': '确定按 SQLite 事实来源修复差异吗？（缺失补写、孤儿删除）',
    'ops.consistency.repair_hint': '修复以 SQLite 为准：补写缺失向量、删除孤儿向量。',
    'ops.consistency.report': '一致性报告',
    'ops.consistency.scan': '开始扫描',
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
    'retrieval.title': 'Retrieval',
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
    'common.status_label': 'status:',
    'common.auto': 'auto',
    'common.loaded': 'loaded',
    'common.unloaded': 'unloaded',
    'common.no_models': 'no models registered.',
    'common.new_db': '+ new database',
    'common.new_collection': '+ new collection',
    'common.click_refresh': 'click "refresh" to load databases.',
    'common.model_id_ops': 'load / unload is keyed by model_id',
    'common.params_label_pre': 'params ',

    'models.auto_refresh_label': 'auto-refresh:',
    'models.load': 'load',
    'models.unload': 'unload',
    'models.loading': 'loading…',
    'models.unloading': 'unloading…',
    'models.load_failed': 'load failed',
    'models.unload_failed': 'unload failed',
    'models.meta_params': 'parameters',
    'models.meta_vram': 'VRAM',
    'models.meta_ram': 'RAM',
    'models.meta_load_time': 'load time',

    'databases.coll_count': 'collection count',
    'databases.meta_count': 'metadata fields',
    'databases.backend_meta': 'backend metadata',
    'databases.coll_list': 'collection list',
    'databases.no_collections': 'no collections yet.',
    'databases.hint': 'each collection belongs to a database; dropping a database drops its collections too.',
    'databases.loading': 'loading database detail...',
    'databases.confirm_drop_pre': 'drop database "',
    'databases.and_collections': '" and all its collections?',

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

    // Operations (TODO §7): queue / reindex / consistency / evaluation
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
    'ops.reindex.promote_hint': 'Promotion runs in one transaction: the canary ref becomes active; the old physical index is later cleaned up by maintenance.',
    'ops.reindex.submit': 'Submit rebuild',
    'ops.consistency.missing': 'Missing in Milvus',
    'ops.consistency.orphans': 'Orphans in Milvus',
    'ops.consistency.repair': 'Repair',
    'ops.consistency.repair_confirm': 'Repair differences against the SQLite source of truth? (rewrite missing, delete orphans)',
    'ops.consistency.repair_hint': 'Repair takes SQLite as truth: rewrite missing vectors and delete orphan vectors.',
    'ops.consistency.report': 'Consistency report',
    'ops.consistency.scan': 'Run scan',
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
  },
};

export function t(key) {
  const dict = I18N[store.locale] || I18N.zh;
  return dict[key] || key;
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
      // Alert when WE watched this id fail: either the polled row just
      // transitioned out of loading, or our own optimistic load is in
      // flight and the server already reports failure. Opening the
      // dashboard on a stale failure shows the inline error, no pop-up.
      const watched =
        (before && before.load_status === 'loading') || busyVerb === 'load';
      if (watched && alertedLoadError[m.id] !== m.load_error) {
        alertedLoadError[m.id] = m.load_error;
        alert(t('models.load_failed') + ': ' + (m.load_error || 'unknown'));
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
        } catch (_e) { /* logged */ }
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
        <span class="seg"><span class="key">SVC</span><span class="val">vector-service</span></span>
        <span class="seg"><span class="key">VER</span><span class="val accent">0.2.0</span></span>
        <span class="seg"><span class="key">EMB</span><span class="val">bge-m3</span></span>
        <span class="seg"><span class="key">STORE</span><span class="val">milvus</span></span>
        <span class="right">
          <span class="seg"><span class="key">{{ t('status.LOADED') }}</span><span class="val">{{ loadedCount }}/{{ store.models.data.length }}</span></span>
        </span>
      </footer>

      <new-db-modal v-if="store.modals.newDb" />
      <new-coll-modal v-if="store.modals.newColl" />
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

