// =====================================================================
// pager.js -- shared first/prev/info/next/last pager (S6).
//   Browse rows and the kb chunk list had byte-identical pager markup
//   with different ids. Each instance passes its DOM ids in as props so
//   the automation anchors (btn-brw-first, chunks-pager, ...) stay where
//   they were; the page-jump box is browse-only (show-jump).
// =====================================================================
import { defineComponent, ref, watch } from '../vue.esm-browser.prod.js';

export default defineComponent({
  name: 'UiPager',
  props: {
    page: { type: Number, required: true },
    pages: { type: Number, required: true },
    busy: Boolean,
    showJump: Boolean,
    infoText: { type: String, default: '' },
    // DOM ids, per instance. Plain attributes (not bindings) at the call
    // site keep the pinned literals on disk.
    rootId: String,
    infoId: String,
    firstId: String,
    prevId: String,
    nextId: String,
    lastId: String,
    jumpId: String,
    jumpBtnId: String,
  },
  emits: ['first', 'prev', 'next', 'last', 'go'],
  setup(props, { emit }) {
    // Mirrors the parent's page number so half-typed input never moves
    // the view; the parent's update re-syncs the box (clamped jump echo).
    const jumpPage = ref(props.page);
    watch(() => props.page, (n) => { jumpPage.value = n; });
    function go() {
      const n = Math.floor(Number(jumpPage.value));
      const target = Math.min(Math.max(1, Number.isFinite(n) && n > 0 ? n : 1), props.pages);
      jumpPage.value = target;
      emit('go', target);
    }
    return { jumpPage, go };
  },
  template: `
    <div class="pager" :id="rootId">
      <button class="btn sm" :id="firstId" :disabled="page <= 1 || busy"
              @click="$emit('first')">« {{ $t('common.first') }}</button>
      <button class="btn sm" :id="prevId" :disabled="page <= 1 || busy"
              @click="$emit('prev')">‹ {{ $t('common.prev') }}</button>
      <span class="info" :id="infoId">{{ infoText }}</span>
      <button class="btn sm" :id="nextId" :disabled="page >= pages || busy"
              @click="$emit('next')">{{ $t('common.next') }} ›</button>
      <button class="btn sm" :id="lastId" :disabled="page >= pages || busy"
              @click="$emit('last')">{{ $t('common.last') }} »</button>
      <template v-if="showJump">
        <span class="info">{{ $t('browse.jump_to') }}</span>
        <input type="number" :id="jumpId" :min="1" :max="pages" v-model.number="jumpPage"
               :aria-label="$t('browse.stat_page')" style="width:72px;" />
        <button class="btn sm" :id="jumpBtnId" :disabled="busy"
                @click="go">{{ $t('browse.go') }}</button>
      </template>
    </div>
  `,
});
