// Shared operation-feedback primitives (S2).
//
// One convention for "the user did something — what happened?":
//
//   StatusBanner  inline role=status / role=alert bar for the outcome of
//                 a submit, replacing the native alert() calls.
//   BusyButton    submit button that disables itself and swaps its label
//                 while a request is in flight, so a double click cannot
//                 fire the same request twice.
//   EmptyState    four-state block for a list/table area: loading (spinner),
//                 empty (guidance), error (reason + retry) and idle
//                 (not run yet) — previously all four rendered the same
//                 "collection is empty or filter matches nothing." copy.
//   NoticeBar     app-level strip for failures that belong to no panel
//                 (a background model load, a poll that starts failing).
//   ConfirmHost   modal confirmation with the blast radius spelled out,
//                 replacing window.confirm().
//
// This module imports nothing from app.js on purpose: app.js mounts the
// two hosts, so an app.js import here would close an import cycle. All
// user-visible text is passed in already translated by the caller; the
// few generic labels come from `$t` inside the templates.
import {
  defineComponent, reactive, ref, computed, watch, nextTick,
  onMounted, onUnmounted,
} from '../vue.esm-browser.prod.js';

// ---------------------------------------------------------------------------
// Inline status banner
// ---------------------------------------------------------------------------
// `kind` picks the colour and the ARIA role: an error interrupts, the
// other kinds are announced politely. `retry` is optional — panels pass
// it only when re-running the same action could plausibly succeed, so a
// form-validation complaint never offers a pointless button.
export const StatusBanner = defineComponent({
  name: 'StatusBanner',
  props: {
    kind: { type: String, default: 'info' },   // error | success | info | warn
    text: { type: String, default: '' },
    retry: { type: Function, default: null },
    retryLabel: { type: String, default: '' },
  },
  computed: {
    ariaRole() { return this.kind === 'error' ? 'alert' : 'status'; },
    glyph() {
      if (this.kind === 'error') return '!';
      if (this.kind === 'success') return '✓';
      if (this.kind === 'warn') return '!';
      return 'i';
    },
  },
  template: `
    <div v-if="text" :class="['banner', kind]" :role="ariaRole">
      <span class="banner-icon" aria-hidden="true">{{ glyph }}</span>
      <span class="banner-text">{{ text }}</span>
      <button v-if="retry" class="btn sm banner-retry" type="button" @click="retry">
        {{ retryLabel || $t('common.retry') }}
      </button>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// Busy submit button
// ---------------------------------------------------------------------------
// Wraps the plain `.btn` markup so `:disabled` and the spinner can never
// be forgotten at a call site. `emits: ['click']` is load-bearing: without
// it Vue would also fall the parent's @click through to the root element
// and the handler would run twice.
export const BusyButton = defineComponent({
  name: 'BusyButton',
  props: {
    busy: { type: Boolean, default: false },
    label: { type: String, default: '' },
    busyLabel: { type: String, default: '' },
    variant: { type: String, default: '' },   // primary | danger | ghost | '' (secondary)
    disabled: { type: Boolean, default: false },
  },
  emits: ['click'],
  computed: {
    isDisabled() { return this.busy || this.disabled; },
    shown() { return this.busy ? (this.busyLabel || this.label) : this.label; },
  },
  template: `
    <button type="button" :class="['btn', variant, busy ? 'is-busy' : '']"
            :disabled="isDisabled" :aria-busy="busy ? 'true' : 'false'"
            @click="$emit('click', $event)">
      <span v-if="busy" class="btn-spinner" aria-hidden="true"></span>{{ shown }}
    </button>
  `,
});

// ---------------------------------------------------------------------------
// Four-state empty block
// ---------------------------------------------------------------------------
// `state` is the same internal vocabulary the panels already use for run
// state, so a panel can pass its `status` ref straight in.
export const EmptyState = defineComponent({
  name: 'EmptyState',
  props: {
    state: { type: String, default: 'empty' },   // idle | loading | empty | error
    text: { type: String, default: '' },
    hint: { type: String, default: '' },
    retry: { type: Function, default: null },
  },
  computed: {
    ariaRole() { return this.state === 'error' ? 'alert' : 'status'; },
    label() { return this.text || this.$t('common.state.' + this.state); },
  },
  template: `
    <div :class="['empty', state]" :role="ariaRole">
      <span v-if="state === 'loading'" class="spinner" aria-hidden="true"></span>
      <!-- Decorative placeholder ring for the no-data states; the
           text/hint carry the actual message (stage 4). -->
      <span v-else-if="state === 'empty' || state === 'idle'" class="empty-glyph" aria-hidden="true"></span>
      <span class="empty-text">{{ label }}</span>
      <span v-if="hint" class="hint">{{ hint }}</span>
      <button v-if="retry && state === 'error'" class="btn sm empty-retry"
              type="button" @click="retry">{{ $t('common.retry') }}</button>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// App-level notice
// ---------------------------------------------------------------------------
// For feedback that has no panel to live in: a background model load that
// failed while the operator was looking at another view, a poll that just
// started erroring. Stays until dismissed or replaced — unlike a toast it
// cannot be missed by looking away.
export const noticeState = reactive({ kind: '', text: '' });

export function notify(kind, text) {
  noticeState.kind = kind;
  noticeState.text = text;
}

export function dismissNotice() {
  noticeState.kind = '';
  noticeState.text = '';
}

export const NoticeBar = defineComponent({
  name: 'NoticeBar',
  setup() { return { noticeState, dismissNotice }; },
  template: `
    <div v-if="noticeState.text" :class="['notice-bar', noticeState.kind]"
         :role="noticeState.kind === 'error' ? 'alert' : 'status'">
      <span class="banner-icon" aria-hidden="true">{{ noticeState.kind === 'error' ? '!' : 'i' }}</span>
      <span class="banner-text">{{ noticeState.text }}</span>
      <button class="notice-close" type="button" :aria-label="$t('common.close')"
              @click="dismissNotice">x</button>
    </div>
  `,
});

// ---------------------------------------------------------------------------
// Modal confirmation
// ---------------------------------------------------------------------------
// `window.confirm()` cannot show what is about to be destroyed. The modal
// takes an explicit `details` list — database, collection, the number of
// rows, the filter expression — rendered as a labelled table so the blast
// radius is read before the button is pressed.
const _dialog = reactive({
  open: false,
  title: '',
  message: '',
  details: [],
  confirmLabel: '',
  danger: true,
});

// Resolver of the in-flight askConfirm(). Kept outside the reactive object
// so Vue never proxies the callback.
let _pending = null;

// Ask the operator to confirm a destructive action. Resolves true when
// confirmed, false when cancelled, dismissed or superseded by a newer
// request (which is resolved first so no caller is left hanging).
export function askConfirm(opts = {}) {
  if (_pending) {
    const previous = _pending;
    _pending = null;
    previous(false);
  }
  _dialog.title = opts.title || '';
  _dialog.message = opts.message || '';
  _dialog.details = Array.isArray(opts.details) ? opts.details : [];
  _dialog.confirmLabel = opts.confirmLabel || '';
  _dialog.danger = opts.danger !== false;
  _dialog.open = true;
  return new Promise(resolve => { _pending = resolve; });
}

export function settleConfirm(ok) {
  const resolve = _pending;
  _pending = null;
  _dialog.open = false;
  if (resolve) resolve(!!ok);
}

export const ConfirmHost = defineComponent({
  name: 'ConfirmHost',
  setup() {
    const cancelBtn = ref(null);
    function onKeydown(ev) {
      if (!_dialog.open) return;
      if (ev.key === 'Escape') { ev.preventDefault(); settleConfirm(false); }
    }
    onMounted(() => window.addEventListener('keydown', onKeydown));
    onUnmounted(() => window.removeEventListener('keydown', onKeydown));
    // Focus the safe button: a stray Enter must never confirm a delete.
    watch(() => _dialog.open, (open) => {
      if (!open) return;
      nextTick(() => { if (cancelBtn.value) cancelBtn.value.focus(); });
    });
    return { dialog: _dialog, settleConfirm, cancelBtn };
  },
  template: `
    <div v-if="dialog.open" class="modal-overlay" id="modal-confirm"
         @click.self="settleConfirm(false)">
      <div class="modal" role="alertdialog" aria-modal="true"
           aria-labelledby="modal-confirm-title">
        <header class="modal-head">
          <h3 id="modal-confirm-title">{{ dialog.title || $t('common.confirm_title') }}</h3>
        </header>
        <div class="modal-body">
          <p v-if="dialog.message" class="confirm-msg">{{ dialog.message }}</p>
          <div v-if="dialog.details.length" class="confirm-impact">
            <div class="confirm-impact-label">{{ $t('common.impact') }}</div>
            <table class="info-table">
              <tr v-for="d in dialog.details" :key="d.label">
                <th>{{ d.label }}</th><td>{{ d.value }}</td>
              </tr>
            </table>
          </div>
        </div>
        <footer class="modal-foot">
          <button ref="cancelBtn" class="btn" type="button"
                  @click="settleConfirm(false)">{{ $t('common.cancel') }}</button>
          <button :class="['btn', dialog.danger ? 'danger' : 'primary']" type="button"
                  @click="settleConfirm(true)">{{ dialog.confirmLabel || $t('common.confirm') }}</button>
        </footer>
      </div>
    </div>
  `,
});
