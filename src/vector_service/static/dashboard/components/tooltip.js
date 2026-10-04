// =====================================================================
// tooltip.js -- shared hover gloss for technical terms (v-tip).
//
// Usage in any component template:
//     <label v-tip="$t('tip.bm25')">bm25</label>
// The directive paints the affordance (dotted underline + help cursor)
// on the host element and shows one fixed-positioned bubble appended to
// <body>, so overflow:auto panels cannot clip it (an ::after bubble
// would be). Text is plain strings only — tooltips explain protocol
// literals and jargon, they never carry markup.
//
// Keyboard: hosts that are not already interactive get tabindex=0 so
// the gloss is reachable by Tab; focus/blur mirror hover.
// Locale switch: templates re-render with a new $t() binding value and
// the `updated` hook swaps the bubble text in place.
// =====================================================================

// One bubble for the whole app; lazily appended on first show.
let bubble = null;
let showTimer = null;
// The host we are currently anchored to — a hide that races a quick
// re-enter on a neighbour must not blank the new anchor's bubble.
let currentEl = null;

// Hover-intent delay: a tooltip that pops instantly makes skimming a
// dense form feel twitchy. Focus (keyboard) shows immediately — the
// user already committed to the element.
const SHOW_DELAY_MS = 160;

function ensureBubble() {
  if (bubble) return bubble;
  bubble = document.createElement('div');
  bubble.className = 'tip-bubble';
  bubble.setAttribute('role', 'tooltip');
  bubble.hidden = true;
  document.body.appendChild(bubble);
  return bubble;
}

function placeBubble(el) {
  const b = ensureBubble();
  const r = el.getBoundingClientRect();
  const br = b.getBoundingClientRect();
  const margin = 10;

  // Above by default, flipped below when the headroom is missing;
  // horizontally clamped to the viewport.
  let top = r.top - br.height - margin;
  if (top < margin) top = r.bottom + margin;
  let left = r.left + r.width / 2 - br.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - br.width - margin));

  b.style.top = Math.round(top) + 'px';
  b.style.left = Math.round(left) + 'px';
}

function show(el, text, immediate) {
  if (!text) return;
  const run = () => {
    const b = ensureBubble();
    currentEl = el;
    b.textContent = text;
    b.hidden = false;
    placeBubble(el);
    el.classList.add('tip-open');
  };
  clearTimeout(showTimer);
  if (immediate) run();
  else showTimer = setTimeout(run, SHOW_DELAY_MS);
}

function hide() {
  clearTimeout(showTimer);
  showTimer = null;
  if (currentEl) currentEl.classList.remove('tip-open');
  currentEl = null;
  if (bubble) bubble.hidden = true;
}

// Scrolling (panel or window) moves the host out from under the bubble.
function onScroll() { hide(); }

function bind(el, text) {
  if (el.__tipHandlers) unbind(el);
  const onEnter = () => show(el, el.__tipText, false);
  const onLeave = () => hide();
  const onFocus = () => show(el, el.__tipText, true);
  const onBlur = () => hide();
  const handlers = { onEnter, onLeave, onFocus, onBlur };
  el.__tipHandlers = handlers;
  el.__tipText = text;
  el.classList.add('has-tip');
  // Non-interactive hosts need a focus stop to expose the gloss to
  // keyboard users; buttons/inputs already are focusable.
  if (!/^(button|input|select|textarea|a|summary)$/i.test(el.tagName)) {
    el.setAttribute('tabindex', '0');
    el.__tipAddedTabindex = true;
  }
  el.addEventListener('mouseenter', onEnter);
  el.addEventListener('mouseleave', onLeave);
  el.addEventListener('focus', onFocus);
  el.addEventListener('blur', onBlur);
}

function unbind(el) {
  const h = el.__tipHandlers;
  if (!h) return;
  el.removeEventListener('mouseenter', h.onEnter);
  el.removeEventListener('mouseleave', h.onLeave);
  el.removeEventListener('focus', h.onFocus);
  el.removeEventListener('blur', h.onBlur);
  if (el.__tipAddedTabindex) el.removeAttribute('tabindex');
  el.classList.remove('has-tip', 'tip-open');
  el.__tipHandlers = null;
  el.__tipText = '';
}

// window capture so panel-level scrolls bubble here too.
window.addEventListener('scroll', onScroll, true);
window.addEventListener('resize', hide);

export const tipDirective = {
  mounted(el, binding) { bind(el, binding.value || ''); },
  updated(el, binding) {
    el.__tipText = binding.value || '';
    // A host with a dynamic :class binding gets its className
    // reassigned by Vue's class patch whenever the computed string
    // changes (e.g. the retrieval mode tabs toggling 'on'), wiping the
    // affordance class added at mount. Directive hooks run after the
    // DOM patch, so re-assert it here.
    if (el.__tipHandlers) {
      el.classList.add('has-tip');
      if (currentEl === el) el.classList.add('tip-open');
    }
    if (currentEl === el && bubble && !bubble.hidden) {
      bubble.textContent = el.__tipText;
      placeBubble(el);
    }
  },
  unmounted(el) {
    if (currentEl === el) hide();
    unbind(el);
  },
};
