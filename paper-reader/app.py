#!/usr/bin/env python3
"""Run the local Paperdesk reading room."""

import argparse
import fcntl
import json
import logging
import mimetypes
import shutil
import re
import subprocess
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree.ElementTree import ParseError as ETError
from urllib.parse import unquote, urlsplit

import assistant_engine as ai
import library as lib
from reading_collection import ReadingRoom

STATIC = Path(__file__).parent / "static"
WAKE = threading.Event()
LOGIN = False
COURSE = None
SUMMARY = "id,slug,title,short,venue,year,count,current_page"


def public_job(row):
    if not row:
        return None
    d = dict(row)
    for k in ("request", "result"):
        if d.get(k):
            d[k] = json.loads(d[k])
    return d


def fetch_paper(pid):
    with lib.db() as c:
        r = c.execute("select * from papers where id=?", (pid,)).fetchone()
    if not r:
        raise KeyError("论文不存在。")
    return r


def view_paper(pid):
    with lib.db() as c:
        r = c.execute(f"select {SUMMARY},guide from papers where id=?", (pid,)).fetchone()
        if not r:
            raise KeyError("论文不存在。")
        doc = dict(r)
        chapter = COURSE.by_id.get(pid) if COURSE else None
        doc["course"] = dict(chapter.info(), outline=chapter.outline) if chapter else None
        doc["guide"] = json.loads(r["guide"]) if r["guide"] else None
        if COURSE:
            doc["guide"] = COURSE.guide_for(pid, doc["guide"])
        doc["analysis"] = public_job(
            c.execute("select * from jobs where paper=? and kind='guide' order by id desc limit 1", (pid,)).fetchone()
        )
        if doc["analysis"]:
            doc["analysis"].pop("result", None)
        doc["messages"] = [
            public_job(j) for j in c.execute("select * from jobs where paper=? and kind='ask' order by id", (pid,))
        ]
    return doc


def worker(kind):
    while True:
        with lib.db() as c:
            c.execute("begin immediate")
            job = c.execute(
                "select * from jobs where kind=? and state='queued' order by id limit 1", (kind,)
            ).fetchone()
            if job:
                c.execute("update jobs set state='working',progress='正在读取论文' where id=?", (job["id"],))
        if not job:
            WAKE.wait(1)
            WAKE.clear()
            continue
        jid = job["id"]
        req = json.loads(job["request"])

        def progress(text):
            with lib.db() as c:
                c.execute("update jobs set progress=? where id=?", (text, jid))

        try:
            r = fetch_paper(job["paper"])
            book_question = req.get("origin") == "book"
            pages = [] if book_question else lib.all_text(r)
            if kind == "guide":
                attempted = set(json.loads(r["ocr"] or "[]"))
                sparse = [i for i, t in enumerate(pages, 1) if len(t.strip()) < 40 and i not in attempted]
                for k in range(0, len(sparse), 3):
                    group = sparse[k : k + 3]
                    progress("正在识别原版页图 " + ", ".join(map(str, group)))
                    scans, _ = ai.model(
                        "把每张所附 PDF 页图逐字转录为原文，可选取复制。图注保留原文，无法辨认处标 [不可辨认]，空白页返回空串。按提供顺序配对物理页码。",
                        {"physical_page_numbers": group},
                        ai.TRANSCRIPT,
                        False,
                        set(group),
                        [lib.page_image(r, n) for n in group],
                    )
                    if sorted(x["number"] for x in scans["transcriptions"]) != group:
                        raise ValueError("页图识别的页码不一致，请重试。")
                    for rec in scans["transcriptions"]:
                        if rec["text"].strip():
                            pages[rec["number"] - 1] = rec["text"]
                    attempted.update(group)
                    with lib.db() as c:
                        c.execute(
                            "update papers set texts=?,ocr=? where id=?",
                            (lib.dump(pages), lib.dump(sorted(attempted)), r["id"]),
                        )
                result = ai.analyze(r["title"], pages, req["search"], progress)
                result["ocr_pages"] = sorted(attempted)
                with lib.db() as c:
                    c.execute(
                        "update papers set guide=?, title=?,short=case when slug is null then ? else short end where id=?",
                        (lib.dump(result), result["title"] or r["title"], result["title"] or r["short"], r["id"]),
                    )
                    c.execute("update jobs set state='done',progress='拆解完成' where id=?", (jid,))
            else:
                with lib.db() as c:
                    prev = c.execute(
                        "select request,result from jobs where paper=? and kind='ask' and state='done' and id<? order by id desc limit 3",
                        (r["id"], jid),
                    ).fetchall()
                history = [
                    {"q": json.loads(v["request"])["question"], "a": json.loads(v["result"])["answer"][:1800]}
                    for v in reversed(prev)
                ]
                progress("正在对照选文、拆解与相关来源" if req["search"] else "正在对照选文与拆解")
                if book_question:
                    if not COURSE or r["id"] not in COURSE.by_id:
                        raise ValueError("当前书库找不到原先选中的章节。")
                    result = COURSE.answer(COURSE.by_id[r["id"]], req, history)
                else:
                    guide = json.loads(r["guide"] or "null")
                    if COURSE:
                        guide = COURSE.guide_for(r["id"], guide)
                    result = ai.answer(
                        {"title": r["title"], "guide": guide},
                        pages,
                        req,
                        history,
                        lib.page_image(r, req["page"]),
                    )
                with lib.db() as c:
                    c.execute("update jobs set state='done',result=?,progress='' where id=?", (lib.dump(result), jid))
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "处理失败，可点击重试。"
            logging.exception("Reading task %s failed", jid)
            with lib.db() as c:
                c.execute("update jobs set state='failed',error=?,progress='' where id=?", (message[:450], jid))


class Handler(BaseHTTPRequestHandler):
    server_version = "Paperdesk/1"

    def log_message(self, fmt, *args):
        pass

    def send(self, status, body, content="application/json; charset=utf-8", headers=None, document=False):
        if not isinstance(body, bytes):
            body = lib.dump(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "private, max-age=86400" if content == "image/jpeg" else "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'self'; base-uri 'none'; sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox"
                if document
                else "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
            ),
        )
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def read_body(self, limit):
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            raise ValueError("请求长度无效。") from None
        if size < 0 or size > limit:
            raise ValueError("文件或请求过大。PDF 上限为 64 MiB。")
        b = self.rfile.read(size)
        if len(b) != size:
            raise ValueError("文件传输未完成。")
        return b

    def dispatch(self, post):
        host = self.headers.get("Host", "")
        p = self.server.server_port
        if host not in (f"127.0.0.1:{p}", f"localhost:{p}"):
            return self.send(403, {"error": "请使用本机阅读器地址。"})
        if post and (
            self.headers.get("X-Paperdesk") != "1" or self.headers.get("Origin", "http://" + host) != "http://" + host
        ):
            return self.send(403, {"error": "请在阅读器页面中操作。"})
        path = urlsplit(self.path).path
        if not post:
            if path in ("/", "/app.js", "/course.js", "/style.css"):
                file = (
                    STATIC
                    / (
                        {"/": "index.html", "/app.js": "app.js", "/course.js": "course.js", "/style.css": "style.css"}[
                            path
                        ]
                    )
                )
                return self.send(
                    200,
                    file.read_bytes(),
                    {
                        "/": "text/html; charset=utf-8",
                        "/app.js": "application/javascript; charset=utf-8",
                        "/course.js": "application/javascript; charset=utf-8",
                        "/style.css": "text/css; charset=utf-8",
                    }[path],
                )
            if path == "/api/library":
                with lib.db() as c:
                    rows = c.execute(
                        f"select {SUMMARY} from papers order by (slug is null) desc, case slug when 'evt-asplos2024' then 0 when 'flashattention-neurips2022' then 1 when 'ansor-osdi2020' then 2 else 3 end,created desc,short"
                    ).fetchall()
                entries = []
                for row in rows:
                    ch = COURSE.by_id.get(row["id"]) if COURSE else None
                    if ch or row["count"]:
                        entries.append(dict(row, course=ch.info() if ch else None))
                entries.sort(key=lambda doc: doc["course"]["order"] if doc["course"] else 100)
                return self.send(200, dict(app="paperdesk", connected=LOGIN, papers=entries))
            if path.startswith("/course/") and COURSE:
                key = path.removeprefix("/course/")
                if key not in COURSE.chapters:
                    raise KeyError("找不到该章节。")
                return self.send(
                    200, COURSE.framed(COURSE.chapters[key]).encode(), "text/html; charset=utf-8", document=True
                )
            if path.startswith("/archive/") and COURSE:
                name = unquote(path.removeprefix("/archive/"))
                asset = COURSE.archive_path(name)
                if asset.stat().st_size > 12 * 1024 * 1024:
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(asset.stat().st_size))
                    self.send_header("Content-Disposition", "attachment")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    with asset.open("rb") as f:
                        shutil.copyfileobj(f, self.wfile)
                    return
                body, content = COURSE.archive(name)
                return self.send(200, body, content, document=True)
        if post and path == "/api/import":
            b = self.read_body(lib.MAX_BYTES)
            with tempfile.TemporaryDirectory(dir=lib.DATA) as tmp:
                src = Path(tmp) / "document.pdf"
                src.write_bytes(b)
                pid, duplicate = lib.import_pdf(
                    src,
                    unquote(self.headers.get("X-Filename", "paper.pdf")),
                    self.headers.get("X-Research", "1") == "1",
                )
            WAKE.set()
            return self.send(200, dict(id=pid, duplicate=duplicate))
        m = re.fullmatch(r"/api/papers/([0-9a-f]{64})(?:/(.*))?", path)
        if not m:
            raise KeyError("找不到该项目。")
        pid, action = m.groups()
        if not post and not action:
            return self.send(200, view_paper(pid))
        r = fetch_paper(pid)
        if not post:
            if not r["count"]:
                raise KeyError("本章使用拆解正文，没有独立 PDF。")
            if action == "pages":
                return self.send(200, dict(count=r["count"], pages=lib.page_sizes(r)))
            if action == "pdf":
                return self.send(200, Path(r["path"]).read_bytes(), "application/pdf")
            page = re.fullmatch(r"pages/(\d+)(/image)?", action or "")
            if not page:
                raise KeyError("找不到该项目。")
            n = int(page[1])
            if n < 1 or n > r["count"]:
                raise KeyError("页码超出范围。")
            if page[2]:
                return self.send(200, lib.page_image(r, n).read_bytes(), "image/jpeg")
            data = lib.page_data(r, n)
            data["text"] = lib.all_text(r)[n - 1]
            data["ocr"] = n in json.loads(r["ocr"] or "[]")
            return self.send(200, data)
        data = json.loads(self.read_body(32000))
        if not isinstance(data, dict):
            raise ValueError("无效请求。")
        book_question = action == "ask" and data.get("origin") == "book"
        if action == "position" or (action == "ask" and not book_question):
            n = data.get("page")
            if type(n) is not int or not 1 <= n <= r["count"]:
                raise ValueError("页码超出范围。")
        if action == "position":
            with lib.db() as c:
                c.execute("update papers set current_page=? where id=?", (n, pid))
            return self.send(200, dict(ok=True))
        search = data.get("search", True)
        if not isinstance(search, bool):
            raise ValueError("无效的检索选项。")
        if action == "ask":
            question = data.get("question", "")
            quote = data.get("quote", "")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                raise ValueError("问题需包含 1–2000 个字符。")
            if not isinstance(quote, str) or len(quote) > 8000:
                raise ValueError("请缩短选文至 8000 字以内。")
            if book_question:
                chapter = COURSE.by_id.get(pid) if COURSE else None
                if not chapter:
                    raise ValueError("此文档没有合订本章节。")
                start = data.get("anchor", "start")
                end = data.get("end_anchor", start)
                if not isinstance(start, str) or not isinstance(end, str):
                    raise ValueError("选文位置无效。")
                i, _, _ = chapter.selection(start, end, quote)
                req = dict(
                    origin="book",
                    anchor=start,
                    end_anchor=end,
                    section=chapter.blocks[i].heading,
                    quote=quote,
                    question=question.strip(),
                    search=search,
                )
            elif quote:
                needle = lib.compact(quote)
                if not needle:
                    raise ValueError("请选择有文字的段落。")
                text = lib.all_text(r)[n - 1]
                if needle not in lib.compact(text):
                    overlay = " ".join(v[-1] for v in lib.page_data(r, n)["lines"])
                    if needle not in lib.compact(overlay):
                        raise ValueError("选文与本页原文不一致，请重新选择。")
            if not book_question:
                req = dict(page=n, quote=quote, question=question.strip(), search=search)
        elif action == "guide":
            if not r["count"]:
                raise ValueError("本章已有完整拆解正文。")
            req = dict(search=search)
        else:
            raise KeyError("找不到该操作。")
        with lib.db() as c:
            c.execute("begin immediate")
            running = c.execute(
                "select id from jobs where paper=? and kind=? and state in ('queued','working')", (pid, action)
            ).fetchone()
            if running:
                return self.send(409, dict(error="该论文已有同类任务在进行。", id=running[0]))
            cur = c.execute(
                "insert into jobs(paper,kind,state,request,progress) values(?,?,'queued',?,'等待助手')",
                (pid, action, lib.dump(req)),
            )
            jid = cur.lastrowid
        WAKE.set()
        return self.send(202, dict(id=jid))

    def run_route(self, post):
        try:
            self.dispatch(post)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except KeyError as e:
            self.send(404, {"error": e.args[0]})
        except (ValueError, UnicodeError, ETError) as e:
            self.send(400, {"error": str(e)[:400]})
        except Exception:
            logging.exception("Request failed")
            self.send(500, {"error": "本次操作失败，请重试。"})

    def do_GET(self):
        self.run_route(False)

    def do_POST(self):
        self.run_route(True)


def main():
    global LOGIN, COURSE
    parser = argparse.ArgumentParser(description="纸间 · 私人论文研读室")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        from urllib.request import urlopen

        try:
            with urlopen(url + "api/library", timeout=2) as resp:
                existing = json.load(resp).get("app") == "paperdesk"
        except Exception:
            existing = False
        if not existing:
            parser.exit(1, "端口已被占用，可用 --port 指定其他端口。\n")
        if args.open:
            webbrowser.open(url)
        print(url)
        return
    lib.DATA.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (lib.DATA / "reader.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(1, "这个书库已在另一个阅读器中打开。\n")
    lib.init()
    lib.seed()
    COURSE = ReadingRoom(lib.BOOK)
    COURSE.attach()
    if ai.cli():
        status = subprocess.run([ai.cli(), "login", "status"], capture_output=True, timeout=15)
        LOGIN = status.returncode == 0
    for kind in ("ask", "guide"):
        threading.Thread(target=worker, args=(kind,), daemon=True).start()
    if args.open:
        webbrowser.open(url)
    print("纸间论文研读室  " + url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
