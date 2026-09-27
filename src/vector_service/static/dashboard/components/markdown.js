// Tiny, dependency-free Markdown -> HTML renderer for the parse-result
// "preview" tab. It only needs to render markdown produced by the
// Docling/markdown parsers well enough for an operator to preview it:
// headings, GFM pipe tables, ordered/unordered lists (one nesting
// level), fenced code, blockquotes, hr, inline code/links/emphasis.
//
// Security: input is HTML-escaped before any formatting is applied and
// raw HTML is NEVER passed through, so the result is safe for v-html.
// Link targets are restricted to http(s)/mailto/relative URLs.

const ENTITY_ESCAPES = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ENTITY_ESCAPES[ch]);
}

function safeUrl(raw) {
  // The text has already been entity-escaped at this point; undo the
  // escapes for the URL check then re-escape for the attribute.
  const u = raw.replace(/&amp;/g, '&').replace(/&quot;/g, '"').replace(/&#39;/g, "'").trim();
  if (/^(https?:|mailto:|#|\/)/i.test(u)) return escapeHtml(u);
  return null;
}

// Inline spans: code, links, images, bold/italic/strike. Inline code is
// extracted first via sentinel placeholders so its content is neither
// escaped twice nor formatted. Private-use code points keep the
// placeholders invisible and collision-free.
function renderInline(text) {
  let s = escapeHtml(text);
  const codeSpans = [];
  s = s.replace(/`([^`\n]+)`/g, (_m, code) => {
    codeSpans.push(code);
    return `${codeSpans.length - 1}`;
  });

  // URLs may not contain whitespace or parentheses (parenthesised URLs
  // must be %-encoded / escaped); this keeps javascript:alert(1) intact
  // as literal text instead of half-consuming the trailing ')'.
  // images ![alt](url)
  s = s.replace(/!\[([^\]]*)\]\(([^()\s]+)(?:\s+&quot;[^&]*?&quot;)?\)/g, (m, alt, url) => {
    const u = safeUrl(url);
    return u ? `<img src="${u}" alt="${alt}">` : m;
  });
  // links [text](url)
  s = s.replace(/\[([^\]]+)\]\(([^()\s]+)(?:\s+&quot;[^&]*?&quot;)?\)/g, (m, label, url) => {
    const u = safeUrl(url);
    return u
      ? `<a href="${u}" target="_blank" rel="noopener noreferrer">${label}</a>`
      : m;
  });

  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
       .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
       .replace(/(^|[\s([])\*([^*\n]+)\*/g, '$1<em>$2</em>')
       .replace(/(^|[\s([])_([^_\n]+)_/g, '$1<em>$2</em>')
       .replace(/~~([^~\n]+)~~/g, '<del>$1</del>');

  return s.replace(/(\d+)/g, (_m, i) => `<code>${codeSpans[Number(i)]}</code>`);
}

const FENCE_RE = /^(\s*)(`{3,}|~{3,})(.*)$/;
const LIST_RE = /^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$/;
const BLOCKQUOTE_RE = /^\s*>\s?/;
const HEADING_RE = /^(#{1,6})\s+(.*?)\s*#*\s*$/;

function isBlockStart(line) {
  return (
    !line.trim()
    || FENCE_RE.test(line)
    || HEADING_RE.test(line)
    || LIST_RE.test(line)
    || BLOCKQUOTE_RE.test(line)
    || /^\s*([-*_])\s*(?:\1\s*){2,}$/.test(line)
  );
}

function alignAttr(spec) {
  if (/^:-+$/.test(spec)) return ' style="text-align:left"';
  if (/^-+:$/.test(spec)) return ' style="text-align:right"';
  if (/^:-+:$/.test(spec)) return ' style="text-align:center"';
  return '';
}

function splitTableRow(row) {
  let s = row.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|')) s = s.slice(0, -1);
  // Docling escapes pipes inside cells as \|.
  return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, '|'));
}

function parseList(lines, i, indent) {
  const ordered = /^\s*\d+[.)]\s/.test(lines[i]);
  const tag = ordered ? 'ol' : 'ul';
  let out = `<${tag}>`;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // A single blank line before another item is just list spacing.
      if (i + 1 < lines.length && LIST_RE.test(lines[i + 1])) { i += 1; continue; }
      break;
    }
    const m = line.match(LIST_RE);
    if (!m) {
      // Indented continuation text belongs to the current item.
      if (/^\s+/.test(line)) { out += ' ' + renderInline(line.trim()); i += 1; continue; }
      break;
    }
    const curIndent = m[1].length;
    if (curIndent < indent) break;
    if (curIndent > indent) break;
    // A switch between bullet / numbered markers at the same level ends
    // this list; the outer loop starts a fresh <ol>/<ul>.
    const itemOrdered = /^\s*\d+[.)]\s/.test(line);
    if (itemOrdered !== ordered) break;
    i += 1;
    const next = lines[i] && lines[i].match(LIST_RE);
    if (next && next[1].length > indent) {
      const sub = parseList(lines, i, next[1].length);
      out += `<li>${renderInline(m[2])}${sub.html}</li>`;
      i = sub.i;
    } else {
      out += `<li>${renderInline(m[2])}</li>`;
    }
  }
  return { html: `${out}</${tag}>`, i };
}

export function renderMarkdown(src) {
  if (!src) return '';
  const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    // Fenced code block -----------------------------------------------
    const fence = line.match(FENCE_RE);
    if (fence) {
      const marker = fence[2][0];
      const minLen = fence[2].length;
      const buf = [];
      i += 1;
      while (i < lines.length) {
        const close = lines[i].match(/^(\s*)(`{3,}|~{3,})\s*$/);
        if (close && close[2][0] === marker && close[2].length >= minLen) { i += 1; break; }
        buf.push(lines[i]);
        i += 1;
      }
      out.push(`<pre class="md-pre"><code>${escapeHtml(buf.join('\n'))}</code></pre>`);
      continue;
    }

    // Heading ----------------------------------------------------------
    const heading = line.match(HEADING_RE);
    if (heading) {
      const lvl = heading[1].length;
      out.push(`<h${lvl}>${renderInline(heading[2])}</h${lvl}>`);
      i += 1;
      continue;
    }

    // Horizontal rule --------------------------------------------------
    if (/^\s*([-*_])\s*(?:\1\s*){2,}$/.test(line)) {
      out.push('<hr>');
      i += 1;
      continue;
    }

    // Blockquote (group consecutive '>' lines, recurse for nesting) ----
    if (BLOCKQUOTE_RE.test(line)) {
      const buf = [];
      while (i < lines.length && BLOCKQUOTE_RE.test(lines[i])) {
        buf.push(lines[i].replace(BLOCKQUOTE_RE, ''));
        i += 1;
      }
      out.push(`<blockquote>${renderMarkdown(buf.join('\n'))}</blockquote>`);
      continue;
    }

    // GFM pipe table ---------------------------------------------------
    if (
      line.includes('|')
      && i + 1 < lines.length
      && /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/.test(lines[i + 1])
    ) {
      const headers = splitTableRow(line);
      const aligns = splitTableRow(lines[i + 1]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      let table = '<table class="md-table"><thead><tr>';
      table += headers
        .map((h, idx) => `<th${alignAttr(aligns[idx] || '')}>${renderInline(h)}</th>`)
        .join('');
      table += '</tr></thead><tbody>';
      for (const r of rows) {
        table += '<tr>';
        table += headers
          .map((_h, idx) => `<td${alignAttr(aligns[idx] || '')}>${renderInline(r[idx] || '')}</td>`)
          .join('');
        table += '</tr>';
      }
      table += '</tbody></table>';
      out.push(table);
      continue;
    }

    // List -------------------------------------------------------------
    if (LIST_RE.test(line)) {
      const parsed = parseList(lines, i, line.match(LIST_RE)[1].length);
      out.push(parsed.html);
      i = parsed.i;
      continue;
    }

    // Paragraph: consume lines until the next block boundary -----------
    const buf = [];
    while (i < lines.length && !isBlockStart(lines[i])) {
      buf.push(lines[i]);
      i += 1;
    }
    out.push(`<p>${renderInline(buf.join(' '))}</p>`);
  }
  return out.join('\n');
}
