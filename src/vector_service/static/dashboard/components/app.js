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
    'nav.text_similarity': '文本相似度',
    'nav.image_similarity': '图像相似度',
    'nav.mm_similarity': '图文相似度',
    'nav.databases': '数据库管理',
    'nav.collections': '集合管理',
    'nav.records': '记录',
    'nav.search': '检索',
    'nav.browse': '浏览数据',
    'nav.parse': '文档解析',
    'nav.chunk': '文本分片',
    'nav.ingest': '一体化摄取',
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
  },
  en: {
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
  },
  en: {
    'brand.dashboard': 'Dashboard',
    'topbar.dim_unit': 'dim',
    'status.SVC': 'SVC',
    'status.VER': 'VER',
    'status.EMB': 'EMB',
    'status.STORE': 'STORE',
    'status.LOADED': 'LOADED',

    'nav.overview': 'Overview',
    'nav.models': 'Model List',
    'nav.embeddings': 'Text Embed',
    'nav.image_embeddings': 'Image Embed',
    'nav.multimodal_embeddings': 'Multimodal Embed',
    'nav.rerank': 'Rerank',
    'nav.text_similarity': 'Text Similarity',
    'nav.image_similarity': 'Image Similarity',
    'nav.mm_similarity': 'Multimodal Similarity',
    'nav.databases': 'Databases',
    'nav.collections': 'Collections',
    'nav.records': 'Records',
    'nav.search': 'Search',
    'nav.browse': 'Browse',
    'nav.parse': 'Parse',
    'nav.chunk': 'Chunk',
    'nav.ingest': 'Ingest',
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
    'nav.Navigation': 'Navigation',
    'nav.Overview': 'Overview',
    'nav.Models': 'Models',
    'nav.Vector Store': 'Vector Store',
    'nav.Knowledge Base': 'Knowledge Base',

    'cat.overview': 'Overview',
    'cat.models': 'Models',
    'cat.store': 'Vector Store',
    'cat.kb': 'Knowledge Base',
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

export const store = reactive({
  view: 'overview',
  locale: 'zh',                                 // current UI language: 'zh' | 'en'
  health: { healthz: 'unknown', readyz: 'unknown' },
  embedderDim: 0,
  models: { data: [], busy: Object.create(null), autoRefresh: true, timer: null },
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
function startModelsAutoRefresh() {
  if (store.models.timer) return;
  store.models.timer = setInterval(async () => {
    if (!store.models.autoRefresh) return;
    try {
      const { payload } = await api('GET', '/v1/models');
      store.models.data = (payload && payload.data) || [];
      const loaded = (store.models.data || []).find(m => m.type === 'embedder' && m.loaded);
      store.embedderDim = loaded && loaded.dimensions ? loaded.dimensions : 0;
    } catch (_e) {}
  }, 5000);
}

const NAV_LABELS = {
  'overview':            { catKey: 'cat.overview',  subKey: 'nav.overview' },
  'models':              { catKey: 'cat.models',    subKey: 'nav.models' },
  'embeddings':          { catKey: 'cat.models',    subKey: 'nav.embeddings' },
  'image-embeddings':    { catKey: 'cat.models',    subKey: 'nav.image_embeddings' },
  'multimodal-embeddings':{ catKey: 'cat.models',   subKey: 'nav.multimodal_embeddings' },
  'rerank':              { catKey: 'cat.models',    subKey: 'nav.rerank' },
  'text-similarity':     { catKey: 'cat.models',    subKey: 'nav.text_similarity' },
  'image-similarity':    { catKey: 'cat.models',    subKey: 'nav.image_similarity' },
  'mm-similarity':       { catKey: 'cat.models',    subKey: 'nav.mm_similarity' },
  'databases':           { catKey: 'cat.store',     subKey: 'nav.databases' },
  'collections':         { catKey: 'cat.store',     subKey: 'nav.collections' },
  'records':             { catKey: 'cat.store',     subKey: 'nav.records' },
  'search':              { catKey: 'cat.store',     subKey: 'nav.search' },
  'browse':              { catKey: 'cat.store',     subKey: 'nav.browse' },
  'parse':               { catKey: 'cat.kb',        subKey: 'nav.parse' },
  'chunk':               { catKey: 'cat.kb',        subKey: 'nav.chunk' },
  'ingest':              { catKey: 'cat.kb',        subKey: 'nav.ingest' },
  'retrieval':           { catKey: 'cat.kb',        subKey: 'nav.retrieval' },
};

const App = defineComponent({
  name: 'DashboardApp',
  components: {
    OverviewPanel, ModelsPanel, DatabasesPanel, CollectionsPanel,
    RecordsPanel, SearchPanel, BrowsePanel, RerankPanel,
    EmbeddingsPanel, SimilarityPanel, KnowledgeBasePanel, RetrievalPanel,
    NewDbModal, NewCollModal,
  },
  setup() {
    onMounted(() => {
      startHealthPolling();
      startModelsAutoRefresh();
    });
    onUnmounted(() => {
      if (healthTimer) clearInterval(healthTimer);
      if (store.models.timer) clearInterval(store.models.timer);
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
              <div :class="['nav-item', store.view === 'text-similarity' ? 'active' : '']" data-view="text-similarity" @click="store.view = 'text-similarity'"><span>{{ t('nav.text_similarity') }}</span></div>
              <div :class="['nav-item', store.view === 'image-similarity' ? 'active' : '']" data-view="image-similarity" @click="store.view = 'image-similarity'"><span>{{ t('nav.image_similarity') }}</span></div>
              <div :class="['nav-item', store.view === 'mm-similarity' ? 'active' : '']" data-view="mm-similarity" @click="store.view = 'mm-similarity'"><span>{{ t('nav.mm_similarity') }}</span></div>
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
              <div :class="['nav-item', store.view === 'retrieval' ? 'active' : '']" data-view="retrieval" @click="store.view = 'retrieval'"><span>{{ t('nav.retrieval') }}</span></div>
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
          <similarity-panel v-show="store.view === 'text-similarity'" kind="text" />
          <similarity-panel v-show="store.view === 'image-similarity'" kind="image" />
          <similarity-panel v-show="store.view === 'mm-similarity'" kind="multimodal" />
          <databases-panel v-show="store.view === 'databases'" />
          <collections-panel v-show="store.view === 'collections'" />
          <records-panel v-show="store.view === 'records'" />
          <search-panel v-show="store.view === 'search'" />
          <browse-panel v-show="store.view === 'browse'" />
          <knowledge-base-panel v-show="store.view === 'parse' || store.view === 'chunk' || store.view === 'ingest'" :view="store.view" />
          <retrieval-panel v-show="store.view === 'retrieval'" />
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

