"""Supplementary reading collections without changing the published coursebook."""
import hashlib
import html
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

import library as lib
from coursebook import Chapter, CourseBook, IDS, LINKS

COLLECTIONS = Path(__file__).parent / "collections"


class ReadingRoom(CourseBook):
    def __init__(self, root, collections=COLLECTIONS):
        super().__init__(root)
        self.mounts = {}
        self.extra_papers = []
        self.extra_pdf_ids = {}
        self.extra_guides = {}
        if not hasattr(self, "css"):
            self.css = "body{max-width:1000px;font:16px/1.7 sans-serif}table{border-collapse:collapse}td,th{padding:.6em;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        for config in sorted(Path(collections).glob("*.json")):
            settings = json.loads(config.read_text())
            directory = (config.parent / settings["root"]).resolve()
            manifest = json.loads((directory / "collection.json").read_text())
            key = manifest["id"]
            if not re.fullmatch(r"[a-z0-9-]+", key):
                raise ValueError("专题标识无效。")
            prefix = "collections/" + key + "/"
            if prefix in self.mounts:
                raise ValueError("专题标识重复。")
            self.mounts[prefix] = directory
            for name in manifest["assets"]:
                self._contained(directory, name)
                self.assets.add(prefix + name)
            for paper in manifest["papers"]:
                self._contained(directory, paper["file"])
                self.extra_papers.append((prefix, directory, paper))
            for item in manifest.get("guides", []):
                if item["slug"] in self.extra_guides:
                    raise ValueError("专题导读重复。")
                path = self._contained(directory, item["file"])
                self.extra_guides[item["slug"]] = json.loads(path.read_text())
            for item in manifest["chapters"]:
                # Existing paper chapters and their reading positions remain canonical.
                existing = next((ch for ch in self.chapters.values() if ch.slug == item["slug"]), None)
                if existing and item.get("mode") != "append":
                    continue
                if not existing and item["key"] in self.chapters:
                    raise ValueError("专题章节标识重复。")
                path = self._contained(directory, item["file"])
                body = path.read_text()
                new_anchors = IDS.findall(body)
                if len(new_anchors) != len(set(new_anchors)) or any(anchor in self.anchors for anchor in new_anchors):
                    raise ValueError("专题正文锚点重复。")
                if existing:
                    notice = ''
                    if item.get('version_notice'):
                        notice = '<p class="version-notice">' + html.escape(item['version_notice'])
                        if new_anchors:
                            notice += ' <a href="#' + html.escape(new_anchors[0], quote=True) + '">查看最新源码核对</a>'
                        notice += '</p>\n'
                    ch = Chapter(existing.key, existing.title, existing.kind, existing.order, existing.stage,
                                 existing.slug, notice + existing.body + '\n<hr>\n' + body)
                else:
                    ch = Chapter(item["key"], item["title"], item["kind"], item["order"], manifest["title"], item["slug"], body)
                self.chapters[ch.key] = ch
                for anchor in new_anchors:
                    self.anchors[anchor] = ch.key

    def guide_for(self, pid, persisted):
        """Compose grounded guides without overwriting saved analysis or reading state."""
        chapter = self.by_id.get(pid)
        extra = self.extra_guides.get(chapter.slug) if chapter else None
        if not extra:
            return persisted
        if not persisted:
            return extra
        combined = {**persisted, **extra}
        saved_overview = persisted.get('overview')
        combined['overview'] = '\n\n'.join(filter(None, [extra.get('overview'),
            '已保存导读（保留供对照）：' + saved_overview if saved_overview else None]))
        for key in ['steps', 'design', 'implementation', 'reproduction', 'gaps', 'bibliography', 'sources']:
            combined[key] = [*extra.get(key, []), *persisted.get(key, [])]
        combined['origin'] = extra.get('origin', persisted.get('origin'))
        return combined

    @staticmethod
    def _contained(root, name):
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("专题资料路径无效：" + name)
        return path

    def attach(self):
        with lib.db() as c:
            for prefix, root, rec in self.extra_papers:
                path = self._contained(root, rec["file"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != rec["sha256"]:
                    raise ValueError("专题论文校验失败：" + rec["id"])
                row = c.execute("select id,count from papers where slug=? or id=? order by count desc limit 1", (rec["id"], digest)).fetchone()
                if row and row["count"]:
                    pid = row["id"]
                else:
                    count, _ = lib.inspect_pdf(path)
                    pid = row["id"] if row else digest
                    if row:
                        c.execute("update papers set title=?,short=?,venue=?,year=?,path=?,count=? where id=?", (rec["title"], rec["short"], rec["venue"], rec["year"], str(path), count, pid))
                    else:
                        c.execute("insert into papers(id,slug,title,short,venue,year,path,count) values(?,?,?,?,?,?,?,?)", (pid, rec["id"], rec["title"], rec["short"], rec["venue"], rec["year"], str(path), count))
                self.extra_pdf_ids[prefix + rec["file"]] = pid
        super().attach()

    def framed(self, ch):
        rendered = super().framed(ch)

        def pdf_link(match):
            url = urlsplit(html.unescape(match[1]))
            name = unquote(url.path.removeprefix("/archive/"))
            if url.path.startswith("/archive/") and name in self.extra_pdf_ids:
                page = int(url.fragment[5:]) if re.fullmatch(r"page=\d+", url.fragment) else 1
                return 'href="/#pdf/' + self.extra_pdf_ids[name] + "/" + str(page) + '"'
            return match[0]

        return LINKS.sub(pdf_link, rendered)

    def archive_path(self, name):
        if name not in self.assets:
            raise KeyError("未收录的资料。")
        for prefix, root in self.mounts.items():
            if name.startswith(prefix):
                try:
                    return self._contained(root, name[len(prefix):])
                except ValueError:
                    raise KeyError("资料不存在。") from None
        return super().archive_path(name)

    def archive(self, name):
        prefix = next((p for p in self.mounts if name.startswith(p)), None)
        if not prefix:
            return super().archive(name)
        path = self.archive_path(name)
        body = path.read_bytes()
        # Render downloaded source as escaped text, including HTML and SVG sources.
        # Only our generated source index pages are served as HTML documents.
        relative = name[len(prefix):]
        if relative.startswith("indexes/") and path.suffix == ".html":
            return body, "text/html; charset=utf-8"
        if relative.startswith("sources/") and len(body) <= 4 * 1024 * 1024 and b"\x00" not in body[:8192]:
            text = body.decode("utf-8", "replace")
            lines = "\n".join(f'<span id="L{i}"><a href="#L{i}">{i:5}</a>  {html.escape(line)}</span>' for i, line in enumerate(text.splitlines(), 1))
            label = relative.removeprefix("sources/")
            header = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + html.escape(label) + '</title><style>body{margin:24px;font-family:system-ui}h1{font-size:18px;overflow-wrap:anywhere}pre{font-size:13px;line-height:1.55}pre span{display:block}pre span:target{background:#fff2bf}a{color:#048a45}nav{margin-bottom:16px}</style>'
            return (header + '<nav><a href="/#read/whole-program-20261120">返回全程序编译与融合专题</a></nav><h1>' + html.escape(label) + '</h1><pre>' + lines + '</pre></html>').encode(), "text/html; charset=utf-8"
        return body, mimetypes.guess_type(path.name)[0] or "application/octet-stream"
