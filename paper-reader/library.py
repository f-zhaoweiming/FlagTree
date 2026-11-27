"""PDF library, page coordinates and the existing research notebook."""

import hashlib
import html
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from assistant_engine import good_url

BOOK = Path(os.getenv("PAPERDESK_BOOK", str(Path.home() / "Documents/learn/kernel-partitioning")))
DATA = Path(os.getenv("PAPERDESK_DATA", str(Path.home() / ".local/share/paperdesk"))).expanduser()
MAX_BYTES = 64 * 1024 * 1024


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


@contextmanager
def db():
    conn = sqlite3.connect(DATA / "library.sqlite3", timeout=20)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init():
    DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
    (DATA / "pdf").mkdir(exist_ok=True)
    (DATA / "cache").mkdir(exist_ok=True)
    with db() as c:
        c.execute("pragma journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
          id TEXT PRIMARY KEY, slug TEXT UNIQUE, title TEXT, short TEXT, venue TEXT, year TEXT,
          path TEXT, count INTEGER, texts TEXT, guide TEXT, created REAL DEFAULT (unixepoch()),
          ocr TEXT DEFAULT '[]', current_page INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id INTEGER PRIMARY KEY, paper TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
          request TEXT NOT NULL, result TEXT, error TEXT, progress TEXT, created REAL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS paper_jobs ON jobs(paper,id);
        """)
        if "ocr" not in {x[1] for x in c.execute("pragma table_info(papers)")}:
            c.execute("alter table papers add column ocr TEXT DEFAULT '[]'")
        c.execute("update jobs set state='queued', progress='恢复上次任务' where state='working'")


def utility(name, *args, timeout=60):
    try:
        return subprocess.run(
            ["/usr/bin/" + name, *map(str, args)], capture_output=True, timeout=timeout, check=True
        ).stdout
    except FileNotFoundError:
        raise ValueError("缺少 PDF 阅读组件，请安装 poppler-utils。") from None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError("无法解析这份 PDF，请检查文件是否完整或受密码保护。") from None


def inspect_pdf(path):
    info = utility("pdfinfo", path).decode("utf-8", "replace")
    m = re.search(r"^Pages:\s+(\d+)", info, re.M)
    if not m or not 1 <= int(m[1]) <= 800:
        raise ValueError("支持 1 至 800 页的 PDF。")
    title = re.search(r"^Title:\s*(.+)$", info, re.M)
    return int(m[1]), title[1].strip() if title else ""


def tidy(t):
    return "".join(ch for ch in t if ord(ch) >= 32 or ch in "\n\t").strip()


def all_text(row):
    if row["texts"] is not None:
        return json.loads(row["texts"])
    path = row["path"]
    pages = utility("pdftotext", "-cropbox", "-enc", "UTF-8", path, "-").decode("utf-8", "replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if len(pages) != row["count"]:
        pages = [
            utility("pdftotext", "-cropbox", "-f", n, "-l", n, path, "-").decode("utf-8", "replace")
            for n in range(1, row["count"] + 1)
        ]
    pages = [tidy(t) for t in pages]
    with db() as c:
        c.execute("update papers set texts=? where id=? and texts is null", (dump(pages), row["id"]))
    return pages


def page_data(row, n):
    src = row["path"]
    xml = utility("pdftotext", "-cropbox", "-f", n, "-l", n, "-bbox-layout", src, "-")
    node = ET.fromstring(re.sub(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", b"", xml)).find(".//{*}page")
    if node is None:
        raise ValueError("这页无法提取坐标。")
    lines = []
    for line in node.findall(".//{*}line"):
        text = " ".join(w.text or "" for w in line.findall("{*}word"))
        if text.strip():
            lines.append([*[float(line.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")], text])
    return {"width": float(node.attrib["width"]), "height": float(node.attrib["height"]), "lines": lines}


def page_sizes(row):
    """Return the rendered aspect ratio of every page without extracting text."""
    info = utility(
        "pdfinfo", "-box", "-f", 1, "-l", row["count"], row["path"]
    ).decode("utf-8", "replace")
    sizes = {}
    for match in re.finditer(r"^Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", info, re.M):
        sizes[int(match[1])] = (float(match[2]), float(match[3]))
    rotated = {
        int(match[1])
        for match in re.finditer(r"^Page\s+(\d+)\s+rot:\s+(\d+)", info, re.M)
        if int(match[2]) % 180
    }
    if not sizes:
        raise ValueError("这份 PDF 无法读取页面尺寸。")
    pages = []
    for n in range(1, row["count"] + 1):
        width, height = sizes.get(n, sizes[min(sizes)])
        if n in rotated:
            width, height = height, width
        pages.append(dict(width=width, height=height))
    return pages


def page_image(row, n):
    directory = DATA / "cache" / row["id"]
    directory.mkdir(exist_ok=True)
    path = directory / f"{n}.jpg"
    if not path.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            prefix = Path(temp) / "page"
            utility(
                "pdftoppm",
                "-cropbox",
                "-f",
                n,
                "-l",
                n,
                "-singlefile",
                "-scale-to",
                "1400",
                "-jpeg",
                "-jpegopt",
                "quality=89",
                row["path"],
                prefix,
            )
            os.replace(prefix.with_suffix(".jpg"), path)
    return path


def compact(text):
    return re.sub(r"[\s\u00ad\-]+", "", unicodedata.normalize("NFKC", text)).casefold()


def section(title, body, sources=()):
    return dict(title=title, body=body, pages=[], sources=list(sources))


def curated(rec, entry, atlas):
    key = entry["id"]
    s = (BOOK / entry["page"]).read_text()
    area = s.partition('id="reader-scope">')[2].partition("</section>")[0]
    tested = []
    gaps = []
    steps = []
    implementation = []
    for title, body in re.findall(r"<p><(?:b|strong)>([^<]+)</(?:b|strong)>(.*?)</p>", area, re.S):
        text = html.unescape(re.sub("<[^>]+>", "", body)).strip()
        dest = gaps if any(a in title for a in ("缺", "条件", "验收", "继续复现")) else tested
        dest.append(section(title, text))
    for item in atlas["steps"]:
        sources = []
        symbols = []
        for r in item["sources"]:
            name = f"{Path(r['file']).name}:{r['line']}"
            url = r.get("upstream", "")
            if good_url(url):
                sources.append(dict(title=name, url=url))
            source_path = BOOK / r["file"]
            excerpt = ""
            if source_path.is_file() and hashlib.sha256(source_path.read_bytes()).hexdigest() == r["sha256"]:
                code = source_path.read_text(errors="replace").splitlines()
                line = r["line"]
                excerpt = "\n" + "\n".join(code[max(0, line - 2) : line + 3])
            symbols.append(f"{r['origin']} · {r['file']}:{r['line']} · {r['anchor']}{excerpt}")
        steps.append(section(item["name"], item["state"] + "。" + item["implementation"], sources))
        implementation.append(section(item["name"], item["implementation"] + "\n" + "\n".join(symbols), sources))
    pub = rec.get("publication_evidence") or rec.get("publication_url") or rec.get("pdf_url")
    prereq = re.search(r"共用原理[^<]*((?:<a .*?</a>(?: · )?)+)", s)
    prereqs = re.findall(r">([^<>]+)</a>", prereq[1]) if prereq else []
    return dict(
        title=rec["title"],
        algorithm=atlas["algorithm"],
        overview=rec.get("core") or atlas["reason"],
        prerequisites=prereqs,
        steps=steps,
        figure=dict(
            title=atlas["algorithm"],
            nodes=[v["name"] for v in atlas["steps"]],
            edges=[dict(start=a, end=b, label=v) for a, b, v in atlas["edges"]],
        ),
        design=[section("设计理由", atlas["reason"])],
        implementation=implementation,
        reproduction=tested,
        gaps=gaps,
        sources=[dict(title="论文发表记录", url=pub)] if good_url(pub) else [],
        bibliography=[
            section(
                "发表记录",
                rec["title"]
                + " · "
                + rec["venue"]
                + " "
                + str(rec["year"])
                + (" · DOI " + rec["doi"] if rec.get("doi") else ""),
            ),
            section(
                "版本依据",
                (
                    "已核对会议排版副本，16 页，印刷页码 301–316。第 307 页页脚留白，正文衔接与相邻页已核对。"
                    if key == "evt-asplos2024"
                    else rec.get("version") or "书库已保存此版本 PDF。"
                ),
            ),
        ],
        origin="随书核对记录",
        searches=[],
        missing_pages=[],
    )


def seed():
    if not (BOOK / "article-algorithms.json").is_file():
        return

    def file(name):
        return json.loads((BOOK / name).read_text())

    catalog = {p["id"]: p for p in file("paper-records.json")["papers"] + file("project-paper-records.json")}
    atlas = file("article-algorithms.json")
    with db() as c:
        seen = {r[0] for r in c.execute("select slug from papers where slug is not null")}
        for item in file("encyclopedia-coverage.json")["entries"]:
            key = item["id"]
            path = BOOK / "papers" / (key + ".pdf")
            if item["kind"] != "paper" or key in seen or key not in catalog or not path.is_file():
                continue
            rec = catalog[key]
            n, _ = inspect_pdf(path)
            pid = hashlib.sha256(path.read_bytes()).hexdigest()
            guide = curated(rec, item, atlas[key])
            c.execute(
                "insert or ignore into papers(id,slug,title,short,venue,year,path,count,guide) values(?,?,?,?,?,?,?,?,?)",
                (pid, key, rec["title"], rec["name"], rec["venue"], str(rec["year"]), str(path), n, dump(guide)),
            )


def import_pdf(path, filename, search):
    pid = hashlib.sha256(path.read_bytes()).hexdigest()
    dest = DATA / "pdf" / (pid + ".pdf")
    if not path.read_bytes()[:1024].lstrip().startswith(b"%PDF-"):
        raise ValueError("请选择 PDF 文件。")
    count, meta = inspect_pdf(path)
    title = (meta or Path(filename).stem or "新论文")[:300]
    with db() as c:
        c.execute("begin immediate")
        if c.execute("select 1 from papers where id=?", (pid,)).fetchone():
            return pid, True
        os.replace(path, dest)
        c.execute(
            "insert into papers(id,title,short,venue,year,path,count) values(?,?,?,?,?,?,?)",
            (pid, title, title, "", "", str(dest), count),
        )
        c.execute(
            "insert into jobs(paper,kind,state,request,progress) values(?,'guide','queued',?,'等待读取原文')",
            (pid, dump({"search": search})),
        )
    return pid, False
