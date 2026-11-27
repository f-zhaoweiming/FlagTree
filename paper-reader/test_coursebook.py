"""Checks against the actual offline publication when it is available."""

import re
import unittest
from collections import Counter

import library
from coursebook import Chapter, CourseBook, FILE


class SelectionTests(unittest.TestCase):
    def test_validate_location_and_selected_text_across_blocks(self):
        chapter = Chapter(
            "general",
            "总论",
            "总论",
            0,
            "",
            "test",
            '<p><a id="a"></a></p><h3>条件</h3><p>保留输入。</p><p><a id="b"></a></p><p>列出输出。</p>',
        )
        self.assertEqual(chapter.selection("a", "b", "保留输入。列出输出。")[:2], (1, 2))
        for args in [("a", "a", "列出输出。"), ("b", "a", ""), ("other", "b", "")]:
            with self.assertRaises(ValueError):
                chapter.selection(*args)


@unittest.skipUnless((library.BOOK / FILE).is_file(), "需要本机合订本")
class OfflineBookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book = CourseBook(library.BOOK)

    def test_preserves_every_original_section_and_its_order(self):
        complete = (library.BOOK / FILE).read_text().split("<main>", 1)[1].split("</main>", 1)[0]
        original = "".join(re.findall(r"<section>(.*?)</section>", complete, re.S))
        actual = "".join(ch.body for ch in self.book.chapters.values())
        self.assertEqual(actual, original)
        self.assertEqual(
            Counter(c.kind for c in self.book.chapters.values()), {"总论": 1, "论文": 35, "项目": 6, "附录": 1}
        )
        self.assertEqual(len(re.findall(r'id="entry-\d\d--algorithm-step-\d+"', actual)), 155)
        self.assertEqual(len(re.findall(r'id="evt-original-\d{3}"', actual)), 16)

    def test_step_offers_local_source_and_its_actual_definition(self):
        chapter = self.book.chapters["entry-19"]
        anchor = "entry-19--algorithm-step-1"
        ctx, refs, _ = self.book.evidence(chapter, anchor, anchor, "建立同组关系", "目标函数")
        match = next(r for r in refs if r["title"].endswith("pass_fusion.py:518"))
        self.assertEqual(ctx["section"], "1　建立分区变量")
        self.assertTrue(match["url"].endswith("#L518"))
        self.assertIn("def get_objective_fn(", match["excerpt"])

    def test_general_solver_and_cross_chapter_links(self):
        c = self.book.chapters["general"]
        anchor = "general--chain"
        _, refs, _ = self.book.evidence(c, anchor, anchor, "", "动态规划源码")
        self.assertTrue(any("def chain_plan(" in r["excerpt"] and "#L" in r["url"] for r in refs))
        self.assertIn('href="/#read/entry-01/entry-01"', self.book.framed(c))
        with self.assertRaises(KeyError):
            self.book.archive_path("../other.html")
