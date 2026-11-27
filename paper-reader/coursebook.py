"""The complete, published HTML is the source of truth for order, text and anchors."""

import hashlib
import html
import json
import mimetypes
import re
from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import assistant_engine as ai
import library as lib

FILE = "内核切分与计算加速-连续阅读.html"
IDS = re.compile(r'\bid="([^"]+)"')
MARKERS = re.compile(r'<a id="([\w.-]+)"></a>|<h2 id="(book-contents)">')
LINKS = re.compile(r'\bhref="([^"]*)"')
HEAD = re.compile(r"<h([1-6])(?: [^>]*)?>(.*?)</h\1>", re.S)
CODE = re.compile(r"^code/[0-9a-f]{16}\.html$")
SOURCE_EXT = {".py", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".c", ".hs", ".sh", ".mlir", ".rs"}
BOOK_ANSWER = ai.obj(answer=ai.S, evidence=ai.arr(ai.S), sources=ai.arr(ai.SOURCE), figure=ai.GRAPH)


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ("br", "p", "tr", "li", "pre", "td", "th", "div", "summary"):
            self.parts.append("\n")


def plaintext(markup):
    parser = Text()
    parser.feed(markup)
    return "".join(parser.parts).strip()


@dataclass
class Passage:
    anchor: str
    body: str
    links: list
    heading: str


def passage_window(body, phrase, limit):
    if len(body) <= limit:
        return body
    # Retain the actual match even when it is near the end of a long appendix.
    pos = body.find(phrase) if phrase else 0
    if pos < 0:
        words = re.findall(r"[\w\u3400-\u9fff]{3,}", phrase)
        pos = next((i for w in sorted(words, key=len, reverse=True) if (i := body.find(w)) >= 0), 0)
    start = max(0, min(pos - limit // 4, len(body) - limit))
    return body[start : start + limit]


class Chapter:
    def __init__(self, key, title, kind, ordinal, stage, slug, body):
        self.key, self.title, self.kind, self.order, self.stage, self.slug = key, title, kind, ordinal, stage, slug
        self.body = body
        self.blocks = []
        self.lookup = {}
        self.outline = []
        marks = list(MARKERS.finditer(body))
        bounds = [("start", 0)] + [(m[1] or m[2], m.start()) for m in marks]
        heading = title
        for i, (anchor, begin) in enumerate(bounds):
            piece = body[begin : bounds[i + 1][1] if i + 1 < len(bounds) else len(body)]
            match = HEAD.search(piece)
            if match:
                heading = plaintext(match[2])
                if heading:
                    self.outline.append({"anchor": anchor, "title": heading, "level": int(match[1])})
            self.lookup[anchor] = i
            self.blocks.append(
                Passage(anchor, plaintext(piece), [html.unescape(x) for x in LINKS.findall(piece)], heading)
            )

    def info(self):
        return dict(key=self.key, order=self.order, stage=self.stage, kind=self.kind)

    def selection(self, start, end, quote_text):
        if start not in self.lookup or end not in self.lookup:
            raise ValueError("这段选文的位置与本章不符，请重新划选。")
        a, b = self.lookup[start], self.lookup[end]
        if b < a:
            raise ValueError("选文范围无效。")
        body = "\n".join(v.body for v in self.blocks[a : b + 1])
        if quote_text and (not lib.compact(quote_text) or lib.compact(quote_text) not in lib.compact(body)):
            raise ValueError("选文与这段拆解正文不一致，请重新选择。")
        return a, b, body


class CourseBook:
    def __init__(self, root):
        self.root = root.resolve()
        self.chapters = {}
        self.anchors = {}
        self.assets = set()
        self.by_id = {}
        self.pdfs = {}
        path = root / FILE
        if not path.is_file():
            return
        src = path.read_text()
        self.css = re.search(r"<style>(.*?)</style>", src, re.S)[1]
        pieces = re.findall(r"<section>(.*?)</section>", src.split("<main>", 1)[1], re.S)
        stage = ""
        prefix = ""
        for part in pieces:
            ids = IDS.findall(part)
            head = HEAD.search(part)
            title = plaintext(head[2])
            key = ids[0]
            if key == "book-contents" or key.startswith("stage-"):
                prefix = part
                if key.startswith("stage-"):
                    stage = title
                continue
            order = int(key[-2:]) if key.startswith("entry-") else 0 if key == "general" else 42
            if key in ("general", "appendices"):
                kind = "总论" if key == "general" else "附录"
                slug = "course-" + key
            else:
                kind = "项目" if title.endswith("（项目）") else "论文"
                title = re.sub(r"^第 \d+ 章\s*|：计算加速|（论文）$|（项目）$", "", title)
                if kind == "论文":
                    slug = re.search(r'href="papers/([^/"#]+)\.pdf', part)[1]
                else:
                    slug = "project-" + title.split(" / ")[0].lower()
            full = prefix + part
            prefix = ""
            ch = Chapter(key, title, kind, order, stage if key.startswith("entry-") else "", slug, full)
            self.chapters[key] = ch
            self.anchors.update({anchor: key for anchor in IDS.findall(full)})
            for href in LINKS.findall(full):
                local = self.local(html.unescape(href))
                if local:
                    self.assets.add(local)
        if list(self.chapters) != ["general", *(f"entry-{i:02}" for i in range(1, 42)), "appendices"]:
            raise ValueError("合订本章节顺序或结构有误。")

    def local(self, href):
        u = urlsplit(href)
        if u.scheme or u.netloc or not u.path:
            return None
        p = Path(unquote(u.path))
        path = p if p.is_absolute() else self.root / p
        if ".." in path.parts or not path.is_relative_to(self.root):
            return None
        return path.relative_to(self.root).as_posix()

    def attach(self):
        with lib.db() as c:
            for key, ch in self.chapters.items():
                row = c.execute("select id,count from papers where slug=?", (ch.slug,)).fetchone()
                if ch.kind == "论文" and not row:
                    path = self.root / "papers" / (ch.slug + ".pdf")
                    if path.is_file():
                        pid = hashlib.sha256(path.read_bytes()).hexdigest()
                        row = c.execute("select id,count from papers where id=?", (pid,)).fetchone()
                if not row:
                    pid = hashlib.sha256(("paperdesk-curriculum:" + ch.slug).encode()).hexdigest()
                    c.execute(
                        "insert or ignore into papers(id,slug,short,title,count) values(?,?,?,?,0)",
                        (pid, ch.slug, ch.title, ch.title),
                    )
                else:
                    pid = row["id"]
                self.by_id[pid] = ch
                if ch.kind == "论文" and row and row["count"]:
                    self.pdfs[ch.slug] = pid

    def framed(self, ch):
        def link(m):
            href = html.unescape(m[1])
            u = urlsplit(href)
            a = unquote(u.fragment)
            local = self.local(href)
            if not u.path and not u.scheme and not u.netloc and a in self.anchors:
                target = "/#read/" + self.anchors[a] + "/" + quote(a)
            elif local in self.assets:
                slug = Path(local).stem
                if local == "papers/" + slug + ".pdf" and slug in self.pdfs:
                    page = int(a[5:]) if re.fullmatch(r"page=\d+", a) else 1
                    target = "/#pdf/" + self.pdfs[slug] + "/" + str(page)
                else:
                    target = "/archive/" + quote(local) + ("#" + quote(a, safe="=") if a else "")
            else:
                return m[0]
            return 'href="' + html.escape(target, quote=True) + '"'

        return (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
            + html.escape(ch.title)
            + "</title><style>"
            + self.css
            + "html{scroll-padding-top:.5em}body{margin:0 auto;padding:22px clamp(14px,3vw,36px) 45px;font-size:16px;background:#fff;color:#1f2329} h4,h5,h6{font-size:1em} img{display:block;margin:1em auto} summary{padding:.5em 0} :target{scroll-margin-top:12px} pre{tab-size:4} h1,h2,h3,h4,h5,h6{color:#1f2329} a{color:#048a45} pre,.equation{background:#f5f6f7} th,td{border-color:#ebedf0} th{background:#f7f8fa} .book-nav{border-bottom-color:#ebedf0} ::selection{background:#07c16040} </style></head><body>"
            + LINKS.sub(link, ch.body)
            + "</body></html>"
        )

    def archive_path(self, name):
        if name not in self.assets:
            raise KeyError("未收录的资料。")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise KeyError("资料不存在。")
        return path

    def archive(self, name):
        path = self.archive_path(name)
        if path.suffix in SOURCE_EXT:
            lines = path.read_text(errors="replace").splitlines()
            rows = "\n".join(
                f'<span id="L{i}"><a href="#L{i}">{i:5}</a>  {html.escape(v)}</span>' for i, v in enumerate(lines, 1)
            )
            return (
                "<!doctype html><meta charset=utf-8><style>body{margin:2em}pre span{display:block}pre span:target{background:#fff2bf}a{color:#65796b}</style><h1>"
                + html.escape(name)
                + "</h1><pre>"
                + rows
                + "</pre>"
            ).encode(), "text/html; charset=utf-8"
        if CODE.fullmatch(name):
            # Also permit the immutable raw file linked by this indexed line viewer.
            body = path.read_bytes()
            for href in LINKS.findall(body[:12000].decode(errors="ignore")):
                rel = (path.parent / unquote(urlsplit(html.unescape(href)).path)).resolve()
                if rel.is_relative_to(self.root) and rel.is_file():
                    self.assets.add(rel.relative_to(self.root).as_posix())
            return body, "text/html; charset=utf-8"
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), mime

    @lru_cache(maxsize=96)
    def code_excerpt(self, name, line):
        p = self.archive_path(name)
        if CODE.fullmatch(name):
            text = p.read_text()
            title = plaintext(re.search(r"<h1>(.*?)</h1>", text, re.S)[1])
            if not line:
                return title, ""
            rows = []
            for n in range(max(1, line - 2), line + 23):
                m = re.search(rf'<span id="L{n}">(.*?)</span>', text)
                if m:
                    rows.append(plaintext(m[1]))
            if f'id="L{line}"' not in text:
                raise ValueError("源码行不存在")
            return title, "\n".join(rows)[:4200]
        lines = p.read_text(errors="replace").splitlines()
        if line and not 1 <= line <= len(lines):
            raise ValueError("源码行不存在")
        pos = line or 1
        return name, "\n".join(lines[max(0, pos - 3) : pos + 22])[:4200]

    def evidence(self, ch, start, end, phrase, question):
        a, b, selected = ch.selection(start, end, phrase)
        terms = ai.terms(phrase + " " + question)

        def score(v):
            return len(terms & ai.terms((v.heading + " " + passage_window(v.body, phrase, 3000))))

        matched = sorted((v for v in ch.blocks if v.body), key=score, reverse=True)[:5]
        active = ch.blocks[a : b + 1]
        linked = []
        for v in active:
            for h in v.links:
                if h.startswith("#") and h[1:] in ch.lookup:
                    linked.append(ch.blocks[ch.lookup[h[1:]]])
        chosen = (
            active + linked + ch.blocks[max(0, a - 1) : a] + ch.blocks[b + 1 : b + 4] + [x for x in matched if score(x)]
        )
        if ch.key == "general":
            # The book puts the complete reference solver at the end of the introduction.
            chosen.extend(v for v in ch.blocks if "source/partition_dp_tutorial.py" in v.links)
        excerpts = []
        refs = {}
        related = []
        seen = set()
        for v in chosen:
            if v.anchor in seen:
                continue
            seen.add(v.anchor)
            if v.body and len(excerpts) < 15:
                excerpts.append(dict(section=v.heading, text=passage_window(v.body, phrase, 2400)))
            for href in v.links:
                name = self.local(href)
                frag = unquote(urlsplit(href).fragment)
                if not name:
                    if href.startswith("#") and frag in self.anchors:
                        related.append((self.anchors[frag], None))
                    continue
                if name.startswith("papers/") and Path(name).stem in self.pdfs:
                    related.append(
                        (
                            self.by_id[self.pdfs[Path(name).stem]].key,
                            int(frag[5:]) if re.fullmatch(r"page=\d+", frag) else None,
                        )
                    )
                if href in refs or len(refs) >= 12 or not (CODE.fullmatch(name) or Path(name).suffix in SOURCE_EXT):
                    continue
                line = int(frag[1:]) if re.fullmatch(r"L\d+", frag) else 0
                if name == "source/partition_dp_tutorial.py" and not line:
                    symbols = {
                        "固定顺序时，如何推出动态规划": "chain_plan",
                        "分支图上，怎样判断一组划分可执行": "dag_plan",
                    }
                    symbol = symbols.get(ch.blocks[a].heading)
                    if symbol:
                        line = next(
                            (
                                i
                                for i, src in enumerate(self.archive_path(name).read_text().splitlines(), 1)
                                if src.startswith("def " + symbol + "(")
                            ),
                            0,
                        )
                title, snippet = self.code_excerpt(name, line)
                refs[href] = dict(
                    kind="code",
                    title=title + (f":{line}" if line else ""),
                    url="/archive/" + quote(name) + ("#L" + str(line) if line else ""),
                    excerpt=snippet,
                )
        return (
            dict(
                chapter=ch.title,
                section=ch.blocks[a].heading,
                selection=phrase,
                surrounding_text=passage_window(selected, phrase, 9000),
                sections=excerpts,
            ),
            list(refs.values()),
            related,
        )

    def answer(self, ch, request, history):
        ctx, code, related = self.evidence(
            ch, request["anchor"], request["end_anchor"], request["quote"], request["question"]
        )
        candidates = {f"C{i}": v for i, v in enumerate(code, 1)}
        papers = [ch.key] if ch.kind == "论文" else []
        # Only attribute cross-paper evidence to a named paper linked by this chapter.
        related += [
            (self.anchors[h[1:]], None)
            for block in ch.blocks
            for h in block.links
            if ch.kind == "项目" and h.startswith("#") and h[1:] in self.anchors
        ]
        for key, _ in dict.fromkeys(related):
            if self.chapters[key].kind == "论文" and key not in papers:
                papers.append(key)
        keywords = ai.terms(
            request["question"]
            + " "
            + request["quote"]
            + " "
            + json.dumps(ctx["sections"][:2], ensure_ascii=False)[:2400]
        )
        for key in papers[: (1 if ch.kind == "论文" else 3)]:
            target = self.chapters[key]
            pid = self.pdfs.get(target.slug)
            if not pid:
                continue
            with lib.db() as c:
                row = c.execute("select * from papers where id=?", (pid,)).fetchone()
            pages = lib.all_text(row)
            rank = sorted(range(1, len(pages) + 1), key=lambda n: len(keywords & ai.terms(pages[n - 1])), reverse=True)
            explicit = re.match(r"evt-original-(\d+)", request["anchor"])
            priority = [int(explicit[1]) - 300] if explicit and target.slug == "evt-asplos2024" else []
            priority += [n for target_key, n in related if target_key == key and n]
            for n in dict.fromkeys(priority + rank[: (16 if key == ch.key else 4)]):
                if 1 <= n <= len(pages) and pages[n - 1].strip():
                    candidates[f"P{target.order}-{n}"] = dict(
                        kind="page",
                        doc=pid,
                        page=n,
                        title=target.title + " · PDF 第 " + str(n) + " 页",
                        excerpt=pages[n - 1][:6500],
                    )
        prompt = "这是合订本拆解正文提问。先解释选文，再联系一般原理、实现与适用条件。参考资料以编号标识，回答每项论断时就近写 [编号]，并在 evidence 中按出现顺序返回实际支持它的编号。论文页和源码均有依据时都标注；工程事实优先具体源码。不得把相关论文当作项目本身的论文，不要猜页码或行号。保留实测、论文报告与待验证的区别。"
        data, searches = ai.model(
            prompt,
            dict(question=request["question"], book=ctx, evidence=candidates, history=history),
            BOOK_ANSWER,
            request["search"],
            set(),
        )
        keys = list(dict.fromkeys(re.findall(r"\[((?:C\d+|P\d+-\d+))\]", data["answer"]) + data["evidence"]))
        data["evidence"] = [
            dict(id=k, **{f: v for f, v in candidates[k].items() if f != "excerpt"}) for k in keys if k in candidates
        ]
        data["pages"] = []
        data["searches"] = searches
        return data
