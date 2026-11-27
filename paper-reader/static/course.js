'use strict';
// Body markup stays in a scriptless, same-origin frame. Its anchors match the published book.
let frameMarkers = [],
    pendingAnchor = null,
    positionTimer;
function courseControls() {
    const doc = state.doc,
        book = state.mode === 'book' && !!doc?.course;
    $('mode-book').hidden = !doc?.course;
    pressed($('mode-book'), book);
    $('mode-original').hidden = $('mode-text').hidden = !doc?.count;
    pressed($('mode-original'), !book && state.mode === 'original');
    pressed($('mode-text'), !book && state.mode === 'text');
    $('chapter-menu').hidden = $('chapter-nav').hidden = $('book-frame').hidden = !book;
    $('pdf-nav').hidden = $('canvas').hidden = $('pdf-tools').hidden = book || !doc?.count;
    if (book) {
        const chapters = state.books.filter((x) => x.course);
        const idx = chapters.findIndex((x) => x.id === state.id);
        for (const [id, next] of [
            ['chapter-prev', chapters[idx - 1]],
            ['chapter-next', chapters[idx + 1]]
        ]) {
            const b = $(id);
            b.disabled = !next;
            b.title = next?.short || '';
            b.onclick = () => {
                if (next) followRoute('/#read/' + next.course.key);
            };
        }
    }
}
function openChapter(anchor) {
    const c = state.doc?.course;
    if (!c) return;
    state.mode = 'book';
    state.observer?.disconnect();
    state.selection = null;
    $('ask-selection').hidden = true;
    courseControls();
    updatePager();
    const frame = $('book-frame'),
        key = c.key;
    const requested = anchor || localStorage.getItem('paperdesk-locus-' + key) || key;
    history.replaceState(null, '', '/#read/' + key + '/' + encodeURIComponent(requested));
    const menu = $('chapter-jump');
    menu.replaceChildren(new Option('章内目录', ''));
    for (const h of c.outline) menu.append(new Option(h.title, h.anchor));
    if (frame.dataset.key === key && frame.contentDocument?.body?.dataset.course === key) {
        jumpToBlock(requested);
        return;
    }
    pendingAnchor = requested;
    frameMarkers = [];
    frame.dataset.key = key;
    frame.src = '/course/' + key;
}
function jumpToBlock(anchor) {
    const frame = $('book-frame'),
        doc = frame.contentDocument;
    if (!doc || state.mode !== 'book') return;
    const e = doc.getElementById(anchor);
    if (e) {
        for (let p = e.parentElement; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
        e.scrollIntoView({ block: 'start' });
    } else if (anchor === 'start') {
        frame.contentWindow.scrollTo(0, 0);
    }
    const m = $('chapter-jump');
    m.value = [...m.options].some((o) => o.value === anchor) ? anchor : '';
}
function preceding(node, offset) {
    const doc = node.ownerDocument,
        point = doc.createRange();
    point.setStart(node, offset);
    point.collapse(true);
    let lo = 0,
        hi = frameMarkers.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1,
            r = doc.createRange();
        r.selectNode(frameMarkers[mid]);
        r.collapse(true);
        if (r.compareBoundaryPoints(0, point) <= 0) lo = mid + 1;
        else hi = mid;
    }
    return lo ? frameMarkers[lo - 1].id : 'start';
}
function visibleBlock() {
    const f = $('book-frame'),
        doc = f.contentDocument;
    if (!doc?.body || doc.body.dataset.course !== state.doc?.course?.key)
        return state.doc?.course?.key || 'start';
    const rect = doc.body.getBoundingClientRect();
    const el = doc.elementFromPoint(Math.max(10, rect.left + 40), 30);
    if (!el || el === doc.body || el === doc.documentElement) {
        const heads = [...doc.querySelectorAll('h1,h2,h3,h4,h5,h6')].filter(
            (e) => e.getClientRects().length && e.getBoundingClientRect().top <= 90
        );
        return heads.length ? preceding(heads.at(-1), 0) : 'start';
    }
    return preceding(el, 0);
}
function bookRequest() {
    const anchor = visibleBlock();
    return { origin: 'book', anchor, end_anchor: anchor, quote: '' };
}
function bookSelection() {
    const frame = $('book-frame'),
        doc = frame.contentDocument,
        s = doc?.getSelection();
    if (!s?.rangeCount || s.isCollapsed || state.mode !== 'book') return;
    const r = s.getRangeAt(0),
        q = s.toString().trim();
    if (!q) return;
    if (q.length > 8000) {
        say('请选取不超过 8000 字的段落。');
        return;
    }
    const anchor = preceding(r.startContainer, r.startOffset),
        end = preceding(r.endContainer, r.endOffset);
    state.selection = {
        origin: 'book',
        anchor,
        end_anchor: end,
        quote: q,
        section: state.doc.course.outline.find((h) => h.anchor === anchor)?.title || state.doc.short
    };
    const rect = r.getBoundingClientRect(),
        box = frame.getBoundingClientRect(),
        btn = $('ask-selection');
    btn.hidden = false;
    btn.style.left = Math.min(innerWidth - 110, Math.max(box.left + 5, box.left + rect.left)) + 'px';
    btn.style.top = Math.max(box.top, Math.min(innerHeight - 45, box.top + rect.top - 38)) + 'px';
}
function setupChapterFrame() {
    const frame = $('book-frame'),
        doc = frame.contentDocument,
        key = state.doc?.course?.key;
    if (!key || !doc?.body || frame.contentWindow.location.pathname !== '/course/' + key) return;
    doc.body.dataset.course = key;
    doc.addEventListener('keydown', readingKeys);
    frameMarkers = [...doc.querySelectorAll('a[id]:empty,h2#book-contents')];
    doc.addEventListener('pointerup', () => setTimeout(bookSelection, 0));
    doc.addEventListener('keyup', bookSelection);
    doc.addEventListener('selectionchange', () => {
        if (doc.getSelection()?.isCollapsed) $('ask-selection').hidden = true;
    });
    doc.addEventListener('click', (e) => {
        const a = e.target.closest?.('a[href]');
        if (!a) return;
        const href = a.getAttribute('href');
        e.preventDefault();
        if (href.startsWith('/#')) {
            if (e.ctrlKey || e.metaKey) window.open(href, '_blank', 'noopener');
            else followRoute(href);
        } else if (href.startsWith('/archive/') || safeUrl(href)) {
            window.open(href, '_blank', 'noopener');
        }
    });
    doc.addEventListener('wheel', (e) => {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        e.stopPropagation();
        wheelZoom('article', e.deltaY);
    }, { passive: false });
    applyBookZoom();
    frame.contentWindow.addEventListener(
        'scroll',
        () => {
            $('ask-selection').hidden = true;
            clearTimeout(positionTimer);
            positionTimer = setTimeout(() => {
                if (state.mode === 'book' && state.doc?.course?.key === key) {
                    const position = visibleBlock();
                    localStorage.setItem('paperdesk-locus-' + key, position);
                }
            }, 300);
        },
        { passive: true }
    );
    jumpToBlock(pendingAnchor || key);
    pendingAnchor = null;
}
function followRoute(url) {
    if (location.hash !== url.slice(1)) history.pushState(null, '', url);
    handleCourseRoute();
}
async function handleCourseRoute() {
    const [type, id, a] = location.hash.slice(1).split('/');
    if (type === 'read') {
        const p = state.books.find((x) => x.course?.key === id);
        if (!p) return false;
        const anchor = a ? decodeURIComponent(a) : id;
        if (state.doc?.id === p.id) openChapter(anchor);
        else await choose(p.id, { anchor });
        return true;
    }
    if (type === 'pdf') {
        const p = state.books.find((x) => x.id === id && x.count),
            page = Number(a);
        if (!p || !Number.isInteger(page) || page < 1 || page > p.count) return false;
        if (state.doc?.id === id) {
            if (state.mode === 'book') await mode('original');
            await go(page);
        } else await choose(id, { page });
        return true;
    }
    return false;
}
function courseOutline(area, d) {
    area.append(el('h3', '', d.short), el('p', 'status-note', d.course.kind + ' · 章内导航'));
    const list = el('div', 'course-outline');
    for (const h of d.course.outline.filter((h) => h.level <= 4)) {
        list.append(
            btn(h.title, '', () => {
                closeLibrary();
                openChapter(h.anchor);
            })
        );
    }
    area.append(list);
}
function evidenceLink(ref, label) {
    const a = el('a', '', label || ref.title);
    if (ref.kind === 'page') {
        a.href = '/#pdf/' + ref.doc + '/' + ref.page;
        a.target = '_blank';
        a.rel = 'noopener';
        if (ref.doc === state.id)
            a.onclick = (e) => {
                if (!e.ctrlKey && !e.metaKey && !e.shiftKey) {
                    e.preventDefault();
                    closeLibrary();
                    jumpToPage(ref.page);
                }
            };
    } else {
        a.href = ref.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
    }
    return a;
}
function courseEvidence(container, r) {
    if (!r.evidence?.length) return;
    const links = el('div', 'citations course-evidence');
    for (const ref of r.evidence) {
        const a = evidenceLink(ref, (ref.kind === 'code' ? '源码 · ' : '') + ref.title);
        a.dataset.cite = ref.id;
        links.append(a);
    }
    container.append(links);
}
document.addEventListener('DOMContentLoaded', () => {
    $('mode-book').onclick = () => openChapter();
    $('book-frame').addEventListener('load', setupChapterFrame);
    $('chapter-jump').onchange = (e) => {
        if (e.target.value)
            followRoute('/#read/' + state.doc.course.key + '/' + encodeURIComponent(e.target.value));
    };
    window.addEventListener('hashchange', handleCourseRoute);
});
