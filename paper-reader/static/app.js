'use strict';
const $ = id => document.getElementById(id);
const icons = {
  library: 'M3 4v16 M7 4v16 M11 5v15 M15 4l5 15',
  search: 'M20 20l-5-5 M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0',
  plus: 'M12 5v14 M5 12h14',
  minus: 'M5 12h14',
  drive: 'M3 6h18v12H3z M6 14h1 M10 14h1',
  file: 'M6 3h8l4 4v14H6z M14 3v5h4 M9 13h6 M9 16h5',
  text: 'M3 5h18 M8 5v15 M3 20h10 M15 10h6 M18 10v10',
  external: 'M13 4h7v7 M20 4l-10 10 M10 5H4v15h15v-6',
  pointer: 'M4 3l6 17 3-7 7-3z',
  left: 'M15 5l-7 7 7 7',
  right: 'M9 5l7 7-7 7',
  spark: 'M12 3l2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3z',
  layers: 'M3 8l9-5 9 5-9 5z M3 12l9 5 9-5 M3 16l9 5 9-5',
  chat: 'M4 4h16v12H9l-5 4z',
  globe: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M3 12h18 M12 3c5 5 5 13 0 18-5-5-5-13 0-18',
  up: 'M12 19V5 M5 12l7-7 7 7',
  close: 'M5 5l14 14 M19 5L5 19',
  settings: 'M4 6h16 M4 12h16 M4 18h16 M8 3v6 M16 9v6 M10 15v6',
  expand: 'M9 3H3v6 M15 3h6v6 M3 15v6h6 M21 15v6h-6',
  upload: 'M12 16V3 M6 9l6-6 6 6 M4 16v5h16v-5',
  flow: 'M4 5h6v5H4z M14 15h6v5h-6z M7 10v7h7',
  code: 'M8 6l-6 6 6 6 M16 6l6 6-6 6 M14 3l-4 18',
  check: 'M3 12l5 5L21 4',
  book: 'M3 4c4-1 7 0 9 2 2-2 5-3 9-2v14c-4-1-7 0-9 2-2-2-5-3-9-2z M12 6v14',
  flag: 'M5 21V3 M5 4c5-4 9 4 15 0v9c-6 4-10-4-15 0',
  link: 'M9 14l6-6 M7 9L5 11a4 4 0 0 0 6 6l2-2 M17 13l2-2a4 4 0 0 0-6-6l-2 2'
};
const ns = 'http://www.w3.org/2000/svg';

function svgNode(tag, attrs = {}) {
  const e = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e
}

function icon(name) {
  const s = svgNode('svg', {
    class: 'ico',
    viewBox: '0 0 24 24',
    'aria-hidden': 'true'
  });
  s.append(svgNode('path', {
    d: icons[name] || icons.file
  }));
  return s
}
for (const e of document.querySelectorAll('[data-icon]')) e.append(icon(e.dataset.icon));

function el(tag, cls = '', text = '') {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e
}

function btn(text, cls, fn) {
  const b = el('button', cls, text);
  b.type = 'button';
  b.onclick = fn;
  return b
}

function pressed(e, on) {
  e.setAttribute('aria-pressed', String(!!on))
}

function safeUrl(v) {
  try {
    const u = new URL(v);
    return ['https:', 'http:'].includes(u.protocol) ? u.href : ''
  } catch {
    return ''
  }
}

function link(label, url) {
  const a = el('a', '', label);
  a.href = safeUrl(url);
  a.target = '_blank';
  a.rel = 'noopener noreferrer';
  return a
}
async function api(path, body = null) {
  const opts = body === null ? {} : {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Paperdesk': '1'
    },
    body: JSON.stringify(body)
  };
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw Error(data.error || '无法完成请求');
  return data
}
const state = {
  books: [],
  id: null,
  doc: null,
  page: 1,
  pageData: new Map(),
  sizes: [],
  pageEls: new Map(),
  loading: new Set(),
  observer: null,
  positionTimer: null,
  scrollRaf: 0,
  mode: localStorage.getItem('paperdesk-mode') || 'original',
  zoom: 1,
  assistZoom: 1,
  bookZoom: 1,
  tab: 'ask',
  mine: false,
  selection: null,
  draft: null,
  paperSeq: 0,
  busy: false
};
const paperAPI = (tail = '', id = state.id) => '/api/papers/' + id + tail;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
let toastTimer, poll = null, dockTimer = null;
const savedAssistZoom = Number(localStorage.getItem('paperdesk-assist-zoom'));
const savedBookZoom = Number(localStorage.getItem('paperdesk-book-zoom'));
state.assistZoom = savedAssistZoom >= .8 && savedAssistZoom <= 1.6 ? savedAssistZoom : 1;
state.bookZoom = savedBookZoom >= .7 && savedBookZoom <= 1.8 ? savedBookZoom : 1;
document.documentElement.style.setProperty('--assist-zoom', state.assistZoom);

function say(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').hidden = true, 4000)
}

function setIcon(node, name) {
  const svg = node.querySelector('svg.ico');
  if (!svg) {
    node.prepend(icon(name));
    return
  }
  const path = svg.querySelector('path');
  if (path) path.setAttribute('d', icons[name] || icons.file);
  else svg.replaceChildren(svgNode('path', {d: icons[name] || icons.file}));
}

function openLibrary(focusSearch = false) {
  document.body.classList.add('library-open');
  $('library-pane').setAttribute('aria-hidden', 'false');
  $('library-pane').inert = false;
  $('library-button').setAttribute('aria-expanded', 'true');
  $('scrim').hidden = false;
  $('ask-selection').hidden = true;
  clearReaderHot();
  hideDock();
  if (focusSearch) $('filter').focus({preventScroll: true});
}

function closeLibrary() {
  document.body.classList.remove('library-open');
  $('library-pane').setAttribute('aria-hidden', 'true');
  $('library-pane').inert = true;
  $('library-button').setAttribute('aria-expanded', 'false');
  $('scrim').hidden = true;
}

$('library-button').onclick = () => openLibrary(false);
$('close-library').onclick = closeLibrary;
$('scrim').onclick = closeLibrary;

function showDock(hold = false) {
  if (document.body.classList.contains('library-open')) return;
  document.body.classList.add('dock-open');
  $('reading-dock').inert = false;
  clearTimeout(dockTimer);
  if (!hold) dockTimer = setTimeout(hideDock, 2800);
}

function hideDock() {
  clearTimeout(dockTimer);
  document.body.classList.remove('dock-open');
  $('reading-dock').inert = true;
}

function toggleDock() {
  if (document.body.classList.contains('dock-open')) hideDock();
  else showDock(true);
}
$('reading-tools').onclick = toggleDock;
$('reading-dock').addEventListener('pointerenter', () => clearTimeout(dockTimer));
$('reading-dock').addEventListener('pointerleave', () => {
  if (!document.body.classList.contains('library-open')) dockTimer = setTimeout(hideDock, 700);
});
$('canvas').addEventListener('wheel', hideDock, {passive: true});
$('insights').addEventListener('wheel', hideDock, {passive: true});

function clearReaderHot() {
  document.body.classList.remove('reader-hot-library', 'reader-hot-tools')
}

function trackReaderHot(e) {
  if (e.pointerType && e.pointerType !== 'mouse') return;
  const box = $('reader-pane').getBoundingClientRect();
  const inside = e.clientX >= box.left && e.clientX <= box.right && e.clientY >= box.top && e.clientY <= box.bottom;
  document.body.classList.toggle('reader-hot-library', inside && e.clientX <= box.left + 150 && e.clientY <= box.top + 90);
  document.body.classList.toggle('reader-hot-tools', inside && e.clientX <= box.left + 190 && e.clientY >= box.bottom - 90)
}
document.addEventListener('pointermove', trackReaderHot, {passive: true});
document.addEventListener('pointerleave', clearReaderHot);

function wheelDelta(e, el) {
  return e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * el.clientHeight : e.deltaY
}

function canScroll(el) {
  return el.scrollHeight > el.clientHeight + 1
}

function applyBookZoom() {
  const doc = $('book-frame').contentDocument;
  if (!doc?.body) return;
  doc.body.style.zoom = state.bookZoom
}

function setBookZoom(value) {
  const next = Math.max(.7, Math.min(1.8, Math.round(value * 10) / 10));
  if (next === state.bookZoom) return;
  state.bookZoom = next;
  localStorage.setItem('paperdesk-book-zoom', String(next));
  applyBookZoom()
}

function setAssistZoom(value) {
  const next = Math.max(.8, Math.min(1.6, Math.round(value * 10) / 10));
  if (next === state.assistZoom) return;
  state.assistZoom = next;
  const insights = $('insights'),
    progress = (insights.scrollTop + insights.clientHeight / 2) / Math.max(1, insights.scrollHeight);
  document.documentElement.style.setProperty('--assist-zoom', next);
  localStorage.setItem('paperdesk-assist-zoom', String(next));
  requestAnimationFrame(() => {
    insights.scrollTop = Math.max(0, progress * insights.scrollHeight - insights.clientHeight / 2)
  })
}

const zoomAccum = {article: 0, assistant: 0};

function wheelZoom(pane, deltaY) {
  const direction = deltaY < 0 ? 1 : deltaY > 0 ? -1 : 0;
  if (!direction) return;
  zoomAccum[pane] = (zoomAccum[pane] || 0) + Math.abs(deltaY) * direction;
  if (Math.abs(zoomAccum[pane]) < 40) return;
  zoomAccum[pane] = 0;
  if (pane === 'assistant') {
    setAssistZoom(state.assistZoom + direction * .1);
    return
  }
  if (state.mode === 'book') setBookZoom(state.bookZoom + direction * .1);
  else if (state.doc?.count) zoom(state.zoom + direction * .1)
}

document.addEventListener('wheel', e => {
  if (e.defaultPrevented) return;
  const target = e.target instanceof Element ? e.target : null;
  if (!target || target.closest('dialog, #reading-dock, #ask-selection, #toast')) return;
  if (e.ctrlKey || e.metaKey) {
    if (target.closest('.assistant')) {
      e.preventDefault();
      hideDock();
      wheelZoom('assistant', e.deltaY)
    } else if (target.closest('.reader')) {
      e.preventDefault();
      hideDock();
      wheelZoom('article', e.deltaY)
    } else if (target.closest('#pane-splitter')) {
      e.preventDefault();
      const box = $('pane-splitter').getBoundingClientRect();
      wheelZoom(e.clientX < box.left + box.width / 2 ? 'article' : 'assistant', e.deltaY)
    }
    return
  }
  const field = target.closest('textarea, select');
  if (field && canScroll(field)) return;
  if (target.closest('.assistant')) {
    const insights = $('insights');
    if (!target.closest('.insights') && canScroll(insights)) {
      e.preventDefault();
      insights.scrollTop += wheelDelta(e, insights);
      hideDock()
    }
    return
  }
  if (target.closest('.library')) {
    const books = $('books');
    if (!target.closest('.books') && canScroll(books)) {
      e.preventDefault();
      books.scrollTop += wheelDelta(e, books)
    }
    return
  }
  if (target.closest('.reader') && !target.closest('.canvas') && !target.closest('iframe')) {
    const canvas = $('canvas');
    if (!canvas.hidden && canScroll(canvas)) {
      e.preventDefault();
      canvas.scrollTop += wheelDelta(e, canvas);
      hideDock()
    }
  }
}, {passive: false, capture: true});

$('fullscreen').hidden = !document.fullscreenEnabled;
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    say('当前浏览器无法进入全屏。');
  }
};
document.addEventListener('fullscreenchange', () => {
  const label = document.fullscreenElement ? '退出全屏' : '进入全屏';
  $('fullscreen').setAttribute('aria-label', label);
  $('fullscreen').title = label;
  setIcon($('fullscreen'), document.fullscreenElement ? 'close' : 'expand');
});

const splitter = $('pane-splitter');
const splitMedia = matchMedia('(max-width: 860px)');
let splitDrag = null;

function splitTotal(mobile = splitMedia.matches) {
  const box = document.querySelector('.workspace');
  return mobile ? box.clientHeight : box.clientWidth
}

function splitLimits(mobile, total) {
  return mobile
    ? {min: Math.max(220, total * .25), max: total * .65}
    : {min: 300, max: Math.max(300, Math.min(720, total - 429))}
}

function applySplitSize(size, persist = true) {
  const mobile = splitMedia.matches,
    total = splitTotal(mobile);
  if (!total) return null;
  const limits = splitLimits(mobile, total),
    value = Math.max(limits.min, Math.min(limits.max, size));
  if (mobile) {
    const pct = Math.round(value / total * 10000) / 100;
    document.documentElement.style.setProperty('--assist-h', pct + '%');
    splitter.setAttribute('aria-valuenow', String(Math.round(pct)));
    if (persist) localStorage.setItem('paperdesk-split-h', String(pct));
  } else {
    const width = Math.round(value),
      pct = Math.round(width / total * 10000) / 100;
    document.documentElement.style.setProperty('--assist-w', width + 'px');
    splitter.setAttribute('aria-valuenow', String(Math.round(pct)));
    if (persist) localStorage.setItem('paperdesk-split-w', String(pct));
  }
  return value
}

function applySplitSettings() {
  const mobile = splitMedia.matches,
    total = splitTotal(mobile),
    limits = splitLimits(mobile, total);
  splitter.setAttribute('aria-orientation', mobile ? 'horizontal' : 'vertical');
  splitter.setAttribute('aria-label', mobile ? '拖动调整正文与问答高度' : '拖动调整正文与问答宽度');
  splitter.setAttribute('aria-valuemin', String(Math.round(limits.min / total * 100)));
  splitter.setAttribute('aria-valuemax', String(Math.round(limits.max / total * 100)));
  if (mobile) {
    const saved = Number(localStorage.getItem('paperdesk-split-h')),
      pct = saved >= 20 && saved <= 70 ? saved : null;
    if (pct) document.documentElement.style.setProperty('--assist-h', pct + '%');
    else document.documentElement.style.removeProperty('--assist-h');
    splitter.setAttribute('aria-valuenow', String(Math.round(pct || 40)));
    return
  }
  const saved = Number(localStorage.getItem('paperdesk-split-w'));
  if (saved >= 10 && saved <= 90) {
    applySplitSize(total * saved / 100, false)
  } else if (saved >= 200) {
    const applied = applySplitSize(saved, false);
    if (applied != null) localStorage.setItem('paperdesk-split-w', String(Math.round(applied / total * 10000) / 100))
  } else {
    document.documentElement.style.removeProperty('--assist-w');
    requestAnimationFrame(() => {
      if (splitMedia.matches) return;
      const width = $('assistant-pane').getBoundingClientRect().width;
      const current = splitTotal(false);
      if (width && current) splitter.setAttribute('aria-valuenow', String(Math.round(width / current * 100)));
    })
  }
}

function currentSplitSize() {
  const box = $('assistant-pane').getBoundingClientRect();
  return splitMedia.matches ? box.height : box.width
}

function resetSplit() {
  localStorage.removeItem('paperdesk-split-w');
  localStorage.removeItem('paperdesk-split-h');
  document.documentElement.style.removeProperty('--assist-w');
  document.documentElement.style.removeProperty('--assist-h');
  applySplitSettings()
}

splitter.addEventListener('pointerdown', e => {
  if (e.button !== 0) return;
  const mobile = splitMedia.matches,
    page = state.pageEls.get(state.page),
    canvas = $('canvas');
  splitDrag = {
    id: e.pointerId,
    mobile,
    startX: e.clientX,
    startY: e.clientY,
    startSize: currentSplitSize(),
    page: state.page,
    offset: page ? (canvas.scrollTop - page.offsetTop) / Math.max(1, page.offsetHeight) : 0,
    last: null
  };
  try {
    splitter.setPointerCapture(e.pointerId)
  } catch {}
  document.body.classList.add('resizing');
  document.body.dataset.resize = mobile ? 'y' : 'x';
  hideDock();
  $('ask-selection').hidden = true;
  e.preventDefault()
});

splitter.addEventListener('pointermove', e => {
  if (!splitDrag || e.pointerId !== splitDrag.id) return;
  const delta = splitDrag.mobile ? e.clientY - splitDrag.startY : e.clientX - splitDrag.startX;
  splitDrag.last = applySplitSize(splitDrag.startSize - delta, false)
});

function endSplit(e) {
  if (!splitDrag || e.pointerId !== splitDrag.id) return;
  const drag = splitDrag;
  splitDrag = null;
  if (drag.last != null) applySplitSize(drag.last, true);
  document.body.classList.remove('resizing');
  delete document.body.dataset.resize;
  try {
    splitter.releasePointerCapture(e.pointerId)
  } catch {}
  if (!drag.mobile && state.doc?.count && state.mode !== 'book') {
    requestAnimationFrame(() => {
      const page = state.pageEls.get(drag.page);
      if (!page) return;
      $('canvas').scrollTop = Math.max(0, page.offsetTop + drag.offset * page.offsetHeight);
      updatePageFromScroll()
    })
  }
}
splitter.addEventListener('pointerup', endSplit);
splitter.addEventListener('pointercancel', endSplit);
splitter.addEventListener('dblclick', resetSplit);
splitter.addEventListener('keydown', e => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
  e.preventDefault();
  const grow = e.key === 'ArrowLeft' || e.key === 'ArrowUp',
    step = e.shiftKey ? 48 : 16;
  applySplitSize(currentSplitSize() + (grow ? step : -step), true)
});
addEventListener('resize', () => {
  if (!splitDrag) applySplitSettings()
});
splitMedia.addEventListener('change', applySplitSettings);
applySplitSettings();

function renderBooks() {
  const area = $('books');
  const q = $('filter').value.toLowerCase().trim();
  area.replaceChildren();
  $('count').textContent = state.books.length + ' 篇';
  $('count').title = state.books.some(p=>p.course) ? '含总论、论文、项目与共用附录' : '书库条目';
  const matches = state.books.filter(p => (!state.mine || (!p.course && !p.slug)) && [p.short, p.title, p.venue, p.year,p.course?.stage,p.course?.kind]
    .join(' ').toLowerCase().includes(q));
  let lastGroup = null;
  for (const p of matches) {
    const c=p.course;
    const group=c ? c.stage || c.kind : '我的导入';
    if(group!==lastGroup){area.append(el('h3','book-group',group));lastGroup=group}
    const b = btn('', 'book' + (p.id === state.id ? ' active' : ''), () => choose(p.id));
    b.title = p.title;
    b.setAttribute('aria-current', p.id === state.id ? 'page' : 'false');
    const num=c?.order>=1 && c.order<=41 ? String(c.order).padStart(2,'0')+'　' : '';
    b.append(
      el('strong','',num+p.short),
      el('small','',[c?.kind,p.venue,p.year].filter(Boolean).join(' · ')||'我的导入')
    );
    area.append(b)
  }
  if (!matches.length) area.append(el('p', 'empty', q ? '没有匹配的论文。' : state.mine ? '还没有导入的论文。' :
    '点击「添加论文」开始阅读。'));
  pressed($('filter-all'), !state.mine);
  pressed($('filter-mine'), state.mine)
}
$('filter').oninput = renderBooks;
$('filter-all').onclick = () => {
  state.mine = false;
  renderBooks()
};
$('filter-mine').onclick = () => {
  state.mine = true;
  renderBooks()
};
async function library() {
  const data = await api('/api/library');
  state.books = data.papers;
  const c = $('connection');
  $('connection-text').textContent = data.connected ? '已登录' : '未登录';
  c.classList.toggle('good', data.connected);
  c.title = data.connected ? '通过本机 Codex 登录调用 ChatGPT' : '请在终端运行 codex login';
  renderBooks()
}

function activeJob(doc) {
  return [doc.analysis, ...doc.messages].some(j => j && ['queued', 'working'].includes(j.state))
}

function watch() {
  clearTimeout(poll);
  if (!state.doc || !activeJob(state.doc)) return;
  const id = state.id;
  poll = setTimeout(async () => {
    try {
      const d = await api(paperAPI('', id));
      if (id !== state.id) return;
      const old = state.doc;
      state.doc = d;
      if (d.analysis?.state === 'done' && old.analysis?.state !== 'done' && d.guide?.ocr_pages
        ?.includes(state.page)) refreshPage(state.page, true);
      if (d.title !== old.title) {
        const paper = state.books.find(x => x.id === id);
        if (paper) {
          paper.title = d.title;
          paper.short = d.short;
          renderBooks()
        }
        head()
      }
      if (JSON.stringify(old.analysis) !== JSON.stringify(d.analysis) || JSON.stringify(old
          .messages) !== JSON.stringify(d.messages)) insights();
    } catch (e) {
      if (id === state.id) say(e.message)
    }
    if (id === state.id) watch()
  }, 1900)
}

function head() {
  const p = state.doc;
  if (!p) return;
  document.title = p.short + ' · 论文阅读';
  $('reader-pane').setAttribute('aria-label', p.short + ' 阅读区');
  $('pdf-link').href = paperAPI('/pdf');
  $('page-count').textContent = '/ ' + (p.count || '—');
  updatePager()
}

function updatePager() {
  if (!state.doc) return;
  if (document.activeElement !== $('pageno')) $('pageno').value = state.page;
  $('prev').disabled = state.page <= 1;
  $('next').disabled = !state.doc.count || state.page >= state.doc.count;
  $('page-count').textContent = '/ ' + (state.doc.count || '—');
  $('composer-page').textContent = state.mode === 'book' ? '本章正文' : (state.doc.count ? 'P. ' + state.page : '')
}

function updateHash() {
  if (!state.doc || !state.doc.count || state.mode === 'book') return;
  history.replaceState(null, '', '/#pdf/' + state.id + '/' + state.page)
}

function queuePosition() {
  clearTimeout(state.positionTimer);
  state.positionTimer = setTimeout(() => {
    if (!state.doc?.count) return;
    api(paperAPI('/position'), {page: state.page}).catch(() => {})
  }, 1000)
}

function paperAppearance() {
  const p = $('paper');
  p.style.width = (state.zoom * 100) + '%';
  p.style.maxWidth = Math.round(980 * state.zoom) + 'px';
  $('fit').textContent = Math.round(state.zoom * 100) + '%';
  $('minus').disabled = state.zoom <= .7;
  $('plus').disabled = state.zoom >= 1.8;
  pressed($('mode-original'), state.mode === 'original');
  pressed($('mode-text'), state.mode === 'text')
}

function scalePage(page) {
  if (!page || page.classList.contains('text-page')) return;
  const w = page.getBoundingClientRect().width;
  if (!w) return;
  for (const s of page.querySelectorAll('.textlayer span')) s.style.fontSize = (w * s.dataset.ht) + 'px'
}

function scaleLayers() {
  for (const page of state.pageEls.values()) {
    if (page.classList.contains('text-page')) {
      const text = page.querySelector('.plainpage');
      if (text) text.style.fontSize = (17 * state.zoom) + 'px'
    } else scalePage(page)
  }
}
new ResizeObserver(scaleLayers).observe($('paper'));

function renderPage(n) {
  const page = state.pageEls.get(n),
    d = state.pageData.get(n);
  if (!page || !d) return;
  page.classList.remove('loading');
  page.classList.toggle('text-page', state.mode === 'text');
  if (state.mode === 'text') {
    page.style.aspectRatio = 'auto';
    page.replaceChildren();
    if (d.ocr) page.append(el('p', 'page-note', '本页由页图识别，请对照原版核对符号。'));
    const text = el('div', 'plainpage', d.text || '这一页没有可提取的文本，可以切回原版查看图表。');
    text.style.fontSize = (17 * state.zoom) + 'px';
    page.append(text);
    return
  }
  page.style.aspectRatio = d.width + ' / ' + d.height;
  page.replaceChildren();
  const img = el('img', 'pageimg');
  img.alt = '第 ' + n + ' 页';
  img.draggable = false;
  img.decoding = 'async';
  img.src = paperAPI('/pages/' + n + '/image');
  img.onerror = () => {
    if (!img.isConnected) return;
    page.replaceChildren(el('p', 'page-status error', '原版页面未能加载。'), btn('重新载入', 'small-action', () =>
      refreshPage(n, true)))
  };
  const layer = el('div', 'textlayer');
  layer.setAttribute('aria-label', '可选中的第 ' + n + ' 页原文');
  const ctx = document.createElement('canvas').getContext('2d');
  ctx.font = '100px "Times New Roman", serif';
  const fragment = document.createDocumentFragment();
  for (const [x, y, x2, y2, t] of d.lines) {
    const span = el('span', '', t + ' ');
    span.style.left = (100 * x / d.width) + '%';
    span.style.top = (100 * y / d.height) + '%';
    span.dataset.ht = (y2 - y) / d.width;
    span.dataset.ratio = (x2 - x) / Math.max(.1, y2 - y);
    span.style.transform = 'scaleX(' + (span.dataset.ratio * 100 / Math.max(1, ctx.measureText(t)
      .width)) + ')';
    fragment.append(span)
  }
  layer.append(fragment);
  page.append(img, layer);
  scalePage(page)
}

async function refreshPage(n, force = false) {
  if (!state.doc?.count || state.mode === 'book') return;
  const page = state.pageEls.get(n);
  if (!page) return;
  if (!force && state.pageData.has(n)) {
    renderPage(n);
    return
  }
  if (state.loading.has(n)) return;
  const id = state.id;
  state.loading.add(n);
  page.classList.add('loading');
  if (!state.pageData.has(n)) {
    page.replaceChildren(el('p', 'page-status', '正在载入第 ' + n + ' 页…'))
  }
  try {
    const d = await api(paperAPI('/pages/' + n));
    if (id !== state.id) return;
    state.pageData.set(n, d);
    renderPage(n)
  } catch (e) {
    if (id !== state.id) return;
    if (state.pageData.has(n)) say(e.message);
    else page.replaceChildren(el('p', 'page-status error', e.message), btn('重新载入', 'small-action', () =>
      refreshPage(n, true)))
  } finally {
    state.loading.delete(n);
    page.classList.remove('loading')
  }
}

async function setupPages() {
  if (!state.doc?.count) return;
  const id = state.id;
  state.observer?.disconnect();
  state.pageEls = new Map();
  state.sizes = [];
  state.loading.clear();
  try {
    const data = await api(paperAPI('/pages'));
    if (id !== state.id) return;
    if (Array.isArray(data.pages) && data.pages.length === state.doc.count) state.sizes = data.pages
  } catch {
    // Older local servers do not have the sizes endpoint; a uniform fallback still scrolls.
  }
  if (id !== state.id) return;
  if (!state.sizes.length) {
    let ratio = 612 / 792;
    const first = state.pageData.get(1);
    if (first?.width && first?.height) ratio = first.width / first.height;
    else {
      try {
        const d = await api(paperAPI('/pages/1'));
        if (id !== state.id) return;
        state.pageData.set(1, d);
        if (d.width && d.height) ratio = d.width / d.height
      } catch {}
    }
    state.sizes = Array.from({length: state.doc.count}, () => ({width: 1000, height: 1000 / ratio}))
  }
  const fragment = document.createDocumentFragment();
  for (let n = 1; n <= state.doc.count; n++) {
    const size = state.sizes[n - 1] || {width: 1000, height: 1294},
      page = el('article', 'paper-page');
    page.dataset.page = String(n);
    page.style.aspectRatio = size.width + ' / ' + size.height;
    page.setAttribute('aria-label', '第 ' + n + ' 页');
    fragment.append(page);
    state.pageEls.set(n, page)
  }
  $('paper').replaceChildren(fragment);
  paperAppearance();
  state.observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) refreshPage(Number(entry.target.dataset.page))
  }, {root: $('canvas'), rootMargin: '1400px 0px'});
  for (const page of state.pageEls.values()) state.observer.observe(page);
  refreshPage(state.page)
}

function flashPage(n) {
  const page = state.pageEls.get(n);
  if (!page) return;
  page.classList.remove('flash');
  void page.offsetWidth;
  page.classList.add('flash');
  setTimeout(() => page.classList.remove('flash'), 1300)
}

function go(n, save = true) {
  if (!state.doc?.count || state.mode === 'book') return;
  n = Math.max(1, Math.min(state.doc.count, Math.floor(Number(n) || 1)));
  const page = state.pageEls.get(n);
  if (!page) return;
  state.page = n;
  state.doc.current_page = n;
  state.selection = null;
  $('ask-selection').hidden = true;
  $('canvas').scrollTo({
    top: Math.max(0, page.offsetTop - 10),
    behavior: save && !reduceMotion.matches ? 'smooth' : 'auto'
  });
  updatePager();
  updateHash();
  refreshPage(n);
  if (save) {
    flashPage(n);
    queuePosition()
  }
}

function updatePageFromScroll() {
  state.scrollRaf = 0;
  if (!state.doc?.count || state.mode === 'book') return;
  const top = $('canvas').scrollTop + 44;
  let found = state.page;
  for (const [n, page] of state.pageEls) {
    if (page.offsetTop <= top) found = n;
    else break
  }
  if (found === state.page) return;
  state.page = found;
  state.doc.current_page = found;
  updatePager();
  updateHash();
  queuePosition()
}

$('canvas').addEventListener('scroll', () => {
  $('ask-selection').hidden = true;
  if (!state.scrollRaf) state.scrollRaf = requestAnimationFrame(updatePageFromScroll)
}, {passive: true});

async function choose(id, location = {}) {
  const seq = ++state.paperSeq;
  clearTimeout(poll);
  state.id = id;
  state.doc = null;
  state.page = 1;
  state.pageData = new Map();
  state.sizes = [];
  state.pageEls = new Map();
  state.loading.clear();
  state.observer?.disconnect();
  state.draft = null;
  state.selection = null;
  state.busy = false;
  state.zoom = 1;
  $('book-frame').hidden = true;
  $('book-frame').dataset.key = '';
  $('ask-selection').hidden = true;
  $('question').value = '';
  $('composer-page').textContent = '';
  $('paper').replaceChildren(el('p', 'empty', '正在打开论文…'));
  $('insights').replaceChildren(el('p', 'status-note', '正在打开论文…'));
  closeLibrary();
  renderBooks();
  chip();
  try {
    const doc = await api(paperAPI('', id));
    if (seq !== state.paperSeq) return;
    state.doc = doc;
    localStorage.setItem('paperdesk-active', id);
    const saved = localStorage.getItem('paperdesk-mode') === 'text' ? 'text' : 'original';
    state.mode = doc.course && !location.page ? 'book' : (doc.count ? saved : 'original');
    state.tab = 'ask';
    head();
    setView('ask');
    courseControls();
    if (doc.course && !location.page) {
      openChapter(location.anchor)
    } else if (doc.count) {
      await setupPages();
      if (seq !== state.paperSeq) return;
      go(location.page || doc.current_page || 1, false)
    } else {
      $('paper').replaceChildren(el('p', 'empty', '这篇论文没有可显示的页面。'))
    }
    watch()
  } catch (e) {
    if (seq === state.paperSeq) {
      $('paper').replaceChildren(el('p', 'error empty', e.message));
      say(e.message)
    }
  }
}

async function mode(m) {
  if (!state.doc?.count) return;
  const fromBook = state.mode === 'book',
    keep = Math.max(1, Math.min(state.doc.count, state.doc.current_page || state.page || 1));
  if (!fromBook && state.mode === m) return;
  state.mode = m;
  localStorage.setItem('paperdesk-mode', m);
  courseControls();
  if (fromBook) {
    state.page = keep;
    await setupPages();
    go(state.page, false)
  } else {
    for (const n of state.pageData.keys()) renderPage(n);
    requestAnimationFrame(() => go(state.page, false))
  }
  updatePager()
}
$('mode-original').onclick = () => mode('original');
$('mode-text').onclick = () => mode('text');
$('prev').onclick = () => go(state.page - 1);
$('next').onclick = () => go(state.page + 1);
$('pageno').onkeydown = e => {
  if (e.key === 'Enter') {
    $('pageno').blur()
  }
};
$('pageno').onchange = () => go($('pageno').value);

function zoom(z) {
  if (!state.doc?.count) return;
  state.zoom = Math.max(.7, Math.min(1.8, Math.round(z * 10) / 10));
  paperAppearance();
  scaleLayers();
  const keep = state.page;
  requestAnimationFrame(() => go(keep, false))
}
$('plus').onclick = () => zoom(state.zoom + .1);
$('minus').onclick = () => zoom(state.zoom - .1);
$('fit').onclick = () => zoom(1);

function pageElementFor(node) {
  const e = node?.nodeType === 1 ? node : node?.parentElement;
  return e?.closest?.('.paper-page') || null
}

function selected() {
  const s = window.getSelection();
  if (!s || !s.rangeCount || s.isCollapsed) return;
  const r = s.getRangeAt(0),
    p = $('paper');
  if (!p.contains(r.startContainer) || !p.contains(r.endContainer)) return;
  const start = pageElementFor(r.startContainer),
    end = pageElementFor(r.endContainer);
  if (!start || !end) return;
  const q = s.toString().trim();
  if (!q) return;
  if (q.length > 8000) {
    say('请选取不超过 8000 字的段落。');
    return
  }
  if (start !== end) {
    state.selection = null;
    $('ask-selection').hidden = true;
    say('请在同一页内选择文字，跨页选文暂不支持。');
    return
  }
  const rect = r.getBoundingClientRect();
  state.selection = {
    quote: q,
    page: Number(start.dataset.page)
  };
  const b = $('ask-selection');
  b.hidden = false;
  b.style.left = Math.min(innerWidth - 116, Math.max(8, rect.left)) + 'px';
  b.style.top = Math.min(innerHeight - 42, Math.max(8, rect.top - 37)) + 'px'
}
$('paper').addEventListener('pointerup', () => setTimeout(selected, 0));
$('paper').addEventListener('keyup', selected);
document.addEventListener('selectionchange', () => {
  const s = getSelection();
  if (!s.rangeCount || s.isCollapsed) $('ask-selection').hidden = true
});
$('ask-selection').onpointerdown = e => e.preventDefault();
$('ask-selection').onclick = () => {
  if (!state.selection) return;
  state.draft = state.selection;
  state.selection = null;
  $('ask-selection').hidden = true;
  getSelection().removeAllRanges();
  if(state.draft.origin==='book')$('book-frame').contentWindow?.getSelection()?.removeAllRanges();
  setView('ask');
  chip();
  $('question').focus()
};

function chip() {
  const c = $('quote-chip');
  c.hidden = !state.draft;
  c.replaceChildren();
  if (state.draft) {
    const label = el('span', '', (state.draft.origin==='book'?'拆解 · '+(state.draft.section||'正文'):'P. '+state.draft.page)+'　'+state.draft.quote);
    label.title = state.draft.quote;
    const clear = btn('×', '', () => {
      state.draft = null;
      chip()
    });
    clear.setAttribute('aria-label', '移除选文');
    c.append(label, clear)
  }
}

function setView(view) {
  const guide = view === 'guide';
  state.tab = guide ? 'guide' : 'ask';
  const assistant = $('assistant-pane');
  assistant.classList.toggle('guide-open', guide);
  const toggle = $('assist-toggle');
  toggle.setAttribute('aria-pressed', String(guide));
  toggle.title = guide ? '返回问答（G）' : '打开导读（G）';
  $('assist-toggle-label').textContent = guide ? '问答' : '导读';
  setIcon(toggle, guide ? 'chat' : 'layers');
  $('ask-form').hidden = guide;
  if (guide) $('insights').scrollTop = 0;
  insights()
}
$('assist-toggle').onclick = () => setView(state.tab === 'guide' ? 'ask' : 'guide');

function jumpToPage(n) {
  if (!state.doc?.count) return;
  if (state.mode === 'book') mode('original').then(() => go(n)).catch(() => {});
  else go(n)
}

function citations(container, pages = [], sources = []) {
  const c = el('div', 'citations');
  for (const n of pages || []) c.append(btn('原文 P. ' + n, '', () => jumpToPage(n)));
  for (const [i, s] of(sources || []).entries())
    if (safeUrl(s.url)) c.append(link('[' + (i + 1) + '] ' + (s.title || new URL(s.url).hostname), s
      .url));
  if (c.childNodes.length) container.append(c)
}
let graphId = 0;

function diagram(graph) {
  if (!graph || !graph.nodes?.length) return null;
  const nodes = graph.nodes,
    edges = graph.edges || [],
    card = el('figure', 'figure-card');
  const cap = el('figcaption', 'figure-head');
  cap.append(icon('flow'), document.createTextNode('算法图解'));
  cap.title = graph.title;
  card.append(cap);
  const gid = 'arrow-' + (++graphId),
    w = 284,
    step = 68,
    h = nodes.length * step + 4,
    svg = svgNode('svg', {
      class: 'diagram',
      viewBox: '0 0 ' + w + ' ' + h,
      role: 'img',
      'aria-label': graph.title || '算法步骤图'
    });
  const title = svgNode('title');
  title.textContent = graph.title || '算法流程';
  svg.append(title);
  const defs = svgNode('defs'),
    marker = svgNode('marker', {
      id: gid,
      viewBox: '0 0 10 10',
      refX: '8',
      refY: '5',
      markerWidth: '6',
      markerHeight: '6',
      orient: 'auto'
    });
  marker.append(svgNode('path', {
    d: 'M0 1L9 5 0 9z',
    fill: '#22b46a'
  }));
  defs.append(marker);
  svg.append(defs);

  function label(x, y, t, attrs = {}) {
    const e = svgNode('text', {
      x,
      y,
      fill: '#3f4a44',
      'font-size': '10',
      ...attrs
    });
    e.textContent = t;
    svg.append(e)
  }
  for (const [i, e] of edges.entries()) {
    if (!nodes[e.start - 1] || !nodes[e.end - 1]) continue;
    const a = (e.start - 1) * step + 24,
      b = (e.end - 1) * step + 24,
      straight = e.end === e.start + 1,
      left = e.end > e.start;
    const rail = left ? 8 + (i % 3) * 5 : 276 - (i % 3) * 5;
    const d = straight ? 'M142 ' + (a + 17) + ' V' + (b - 18) : 'M' + (left ? 33 : 251) + ' ' + a +
      ' H' + rail + ' V' + b + ' H' + (left ? 32 : 252);
    const path = svgNode('path', {
      d,
      fill: 'none',
      stroke: '#8eb9a1',
      'stroke-width': '1.4',
      'stroke-dasharray': straight ? 'none' : '3 2',
      'marker-end': 'url(#' + gid + ')'
    });
    const tip = svgNode('title');
    tip.textContent = e.start + ' → ' + e.end + ' ' + e.label;
    path.append(tip);
    svg.append(path);
    if (straight) label(150, a + 37, Array.from(e.label || '').slice(0, 12).join(''), {
      'font-size': 8.9
    })
  }
  const fills = ['#f7fbf8', '#e8f8ee', '#eef5ff', '#f2f3f5'];
  nodes.forEach((name, i) => {
    const y = i * step + 7;
    svg.append(svgNode('rect', {
      x: 33,
      y,
      width: 218,
      height: 34,
      rx: 5,
      fill: fills[i % fills.length],
      stroke: '#d7e6dc'
    }));
    label(45, y + 21, String(i + 1).padStart(2, '0'), {
      'font-family': 'monospace',
      'font-size': 10,
      fill: '#8a9099'
    });
    const text = Array.from(name);
    label(69, y + 22, text.length > 19 ? text.slice(0, 18).join('') + '…' : name, {
      'font-size': 12,
      'font-weight': 600
    });
  });
  card.append(svg);
  const extra = edges.filter(e => e.end !== e.start + 1);
  if (extra.length) card.append(el('p', 'fine', extra.map(e => e.start + ' → ' + e.end + '　' + e.label)
    .join('　·　')));
  return card
}

function fold(label, name, sections) {
  if (!sections?.length) return null;
  const d = el('details', 'fold');
  d.dataset.fold = label;
  const summary = el('summary');
  summary.append(icon(name), document.createTextNode(label));
  d.append(summary);
  const c = el('div', 'fold-content');
  for (const sec of sections) {
    const a = el('article');
    if (sec.title) a.append(el('h4', '', sec.title));
    a.append(el('p', '', sec.body));
    citations(a, sec.pages, sec.sources);
    c.append(a)
  }
  d.append(c);
  return d
}

function answerText(r) {
  const p = el('p', 'message-text'),
    str = r.answer;
  for (const part of str.split(/(\*\*[^*]+\*\*|\x60[^\x60]+\x60|\[(?:\d+|C\d+|P\d+-\d+)\])/g)) {
    if (part.startsWith('**') && part.endsWith('**')) p.append(el('strong', '', part.slice(2, -2)));
    else if (part.charCodeAt(0) === 96 && part.charCodeAt(part.length - 1) === 96) p.append(el('code',
      '', part.slice(1, -1)));
    else if ((r.evidence||[]).some(x=>'['+x.id+']'===part))p.append(evidenceLink(r.evidence.find(x=>'['+x.id+']'===part), part));
    else if (/^\[\d+\]$/.test(part) && safeUrl(r.sources?.[Number(part.slice(1, -1)) - 1]?.url)) p
      .append(link(part, r.sources[Number(part.slice(1, -1)) - 1].url));
    else p.append(document.createTextNode(part))
  }
  return p
}

function stateLine(j) {
  const p = el('p', 'status-note' + (j.state === 'failed' ? ' error' : ''), j.state === 'failed' ? j
    .error : j.progress || '处理中…');
  p.setAttribute('role', 'status');
  return p
}

function blockPending(j) {
  return j && ['queued', 'working'].includes(j.state)
}
async function retryGuide() {
  const id = state.id;
  try {
    await api(paperAPI('/guide', id), {
      search: $('web-search').checked
    });
    if (id === state.id) {
      const doc = await api(paperAPI('', id));
      if (id !== state.id) return;
      state.doc = doc;
      insights();
      watch()
    }
  } catch (e) {
    say(e.message)
  }
}

function guide(area, d) {
  const g = d.guide,
    j = d.analysis;
  if (j && j.state !== 'done') {
    area.append(stateLine(j));
    if (j.state === 'failed') area.append(btn('重新拆解', 'small-action', retryGuide))
  }
  if (!g) {
    if(d.course){courseOutline(area,d);return}
    if (!j) area.append(btn('生成论文拆解', 'small-action', retryGuide));
    return
  }
  const line = el('div', 'meta-line');
  line.append(el('span', '', g.origin), el('span', '', g.steps.length + ' 个算法步骤'));
  area.append(line, el('h3', '', g.algorithm || g.title), el('p', 'body-copy', g.overview));
  const pic = diagram(g.figure);
  if (pic) area.append(pic);
  const blocks = [
    ['版本与原文', 'file', [...(g.bibliography || []), {
      title: '本地 PDF',
      body: d.count + ' 个物理页 · SHA-256 ' + d.id
    }]],
    ['先修知识', 'book', g.prerequisites?.map(t => ({
      body: t
    }))],
    ['逐步读算法', 'layers', g.steps],
    ['为什么这样设计', 'flow', g.design],
    ['源码怎样实现', 'code', g.implementation],
    ['复现记录与条件', 'check', g.reproduction],
    ['剩余缺口与验收', 'flag', g.gaps]
  ];
  for (const [text, name, secs] of blocks) {
    const f = fold(text, name, secs);
    if (f) area.append(f)
  }
  citations(area, [], g.sources);
  if (g.ocr_pages?.length) area.append(el('p', 'fine', '以下页使用页图识别，转录请与原图核对：' + g.ocr_pages.join('、')));
  if (g.missing_pages?.length) area.append(el('p', 'status-note', '无文本层的页码：' + g.missing_pages.join(
    '、') + '。请对照原版页图。'));
  if (g.origin === 'ChatGPT 拆解') area.append(el('p', 'fine', '检索记录：' + (g.searches?.join('；') ||
    '本次未执行网页检索')))
}

function chat(area, d) {
  if (!d.messages.length) {
    const empty = el('div', 'assistant-empty');
    const mark = el('span', 'assistmark');
    mark.append(icon('spark'));
    empty.append(mark, el('strong', '', state.draft ? '围绕选文提问' : '问答助手'));
    empty.append(el('p', '', state.draft
      ? '选文已放入下方输入框，写下你想追问的内容。'
      : '选中论文中的文字，点击「就此提问」；也可以直接在下方输入问题。'));
    empty.append(el('p', 'fine shortcut-hint', '按 / 打开书库 · 按 G 查看导读'));
    area.append(empty);
    return
  }
  for (const j of d.messages) {
    const sec = el('article', 'question-block');
    const fromBook=j.request.origin==='book';
    sec.append(el('div','meta-line',fromBook?'拆解正文 · '+j.request.section:'第 '+j.request.page+' 页 · '+(j.request.quote?'选文问答':'论文问答')));
    if (j.request.quote) sec.append(btn(j.request.quote, 'quote-box', () => {
      if(fromBook) openChapter(j.request.anchor);else jumpToPage(j.request.page)
    }));
    sec.append(el('h4', '', j.request.question));
    if (j.state === 'done') {
      const r = j.result;
      sec.append(answerText(r));
      const fig = diagram(r.figure);
      if (fig) sec.append(fig);
      courseEvidence(sec,r);
      if(fromBook && !r.evidence?.length)sec.append(el('p','fine','本回答依据拆解选文，未引用可定位的论文页或源码行。'));
      citations(sec, r.pages, r.sources);
      sec.append(el('p', 'status-note', r.searches?.length ? '已检索：' + r.searches.join('；') :
        '依据论文与已有拆解 · 未执行网页检索'))
    } else {
      sec.append(stateLine(j));
      if (j.state === 'failed') sec.append(btn('重新提问', 'small-action', () => {
        state.draft = j.request.origin==='book' || j.request.quote ? {...j.request} : null;
        chip();
        $('question').value = j.request.question;
        $('web-search').checked = j.request.search;
        $('question').focus()
      }))
    }
    area.append(sec)
  }
}

function insights() {
  const area = $('insights'),
    opens = [...area.querySelectorAll('details[open]')].map(d => d.dataset.fold),
    top = area.scrollTop;
  area.replaceChildren();
  if (!state.doc) {
    updatePager();
    return
  }
  if (state.tab === 'guide') guide(area, state.doc);
  else chat(area, state.doc);
  for (const f of area.querySelectorAll('details'))
    if (opens.includes(f.dataset.fold)) f.open = true;
  area.scrollTop = top;
  $('send').disabled = state.busy || state.doc.messages.some(blockPending);
  updatePager()
}
$('ask-form').onsubmit = async e => {
  e.preventDefault();
  const q = $('question').value.trim(),
    id = state.id;
  if (!q || !state.doc || state.busy) return;
  const draft = state.draft || (state.mode==='book' ? bookRequest() : {page:state.page,quote:''});
  state.busy = true;
  $('send').disabled = true;
  try {
    await api(paperAPI('/ask', id), {
      ...draft,
      question: q,
      search: $('web-search').checked
    });
    if (id !== state.id) return;
    state.draft = null;
    chip();
    $('question').value = '';
    const d = await api(paperAPI('', id));
    if (id !== state.id) return;
    state.doc = d;
    setView('ask');
    $('insights').scrollTop = $('insights').scrollHeight;
    watch()
  } catch (err) {
    say(err.message)
  } finally {
    if (id === state.id) {
      state.busy = false;
      insights()
    }
  }
};
$('question').onkeydown = e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    $('ask-form').requestSubmit();
    e.preventDefault()
  }
};

$('import').onclick = () => {
  $('upload-status').textContent = '';
  $('upload-dialog').showModal()
};
$('close-import').onclick = () => $('upload-dialog').close();
let uploading = false;
async function upload(file) {
  if (!file || uploading) return;
  if (file.size > 64 * 1024 * 1024) {
    $('upload-status').textContent = '文件超过 64 MiB。';
    return
  }
  uploading = true;
  $('upload-status').textContent = '正在导入…';
  $('file').disabled = true;
  try {
    const r = await fetch('/api/import', {
      method: 'POST',
      body: file,
      headers: {
        'Content-Type': 'application/pdf',
        'X-Filename': encodeURIComponent(file.name),
        'X-Paperdesk': '1',
        'X-Research': $('import-search').checked ? '1' : '0'
      }
    });
    const j = await r.json();
    if (!r.ok) throw Error(j.error || '导入失败');
    $('upload-dialog').close();
    $('filter').value = '';
    state.mine = false;
    await library();
    await choose(j.id);
    say(j.duplicate ? '这篇论文已在书库中。' : '已加入书库，正在拆解论文。')
  } catch (e) {
    $('upload-status').textContent = e.message
  } finally {
    uploading = false;
    $('file').disabled = false;
    $('file').value = ''
  }
}
$('file').onchange = e => upload(e.target.files[0]);
for (const type of ['dragenter', 'dragover', 'dragleave', 'drop']) $('drop').addEventListener(type,
e => {
  e.preventDefault();
  $('drop').classList.toggle('drag', type === 'dragover' || type === 'dragenter');
  if (type === 'drop') upload(e.dataTransfer.files[0])
});
function readingKeys(e) {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || e.shiftKey || $('upload-dialog').open) return;
  if (e.key === 'Escape') {
    if (document.body.classList.contains('library-open')) closeLibrary();
    else if (state.tab === 'guide') setView('ask');
    else {
      hideDock();
      $('ask-selection').hidden = true
    }
    return
  }
  if (e.target?.closest?.('input, textarea, select, [contenteditable="true"]')) return;
  if (e.key === '/') {
    e.preventDefault();
    openLibrary(true);
    return
  }
  if (e.key.toLowerCase() === 'g') {
    e.preventDefault();
    setView(state.tab === 'guide' ? 'ask' : 'guide');
    return
  }
  if (e.key.toLowerCase() === 't') {
    e.preventDefault();
    toggleDock()
  }
}
document.addEventListener('keydown', readingKeys);
(async () => {
  try {
    await library();
    const id = localStorage.getItem('paperdesk-active');
    const active = state.books.find(p => p.id === id) || state.books[0];
    if (active) {if(!(await handleCourseRoute()))await choose(active.id)}
    else {
      const empty = el('div', 'empty');
      empty.append(el('strong', 'empty-title', '还没有论文'), el('p', '', '导入 PDF 后即可连续阅读，并随时向问答助手提问。'));
      empty.append(btn('添加论文', 'small-action', () => $('import').click()));
      $('paper').replaceChildren(empty);
      openLibrary(false)
    }
  } catch (e) {
    say(e.message)
  }
})();
