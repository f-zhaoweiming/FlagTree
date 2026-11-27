"""Regression checks for page identity during vision transcription and reading."""

import unittest

import assistant_engine as ai
from library import compact


class PageEvidenceTests(unittest.TestCase):
    def test_vision_transcript_keeps_page_identity(self):
        result = {"transcriptions": [{"number": 7, "text": "Epilogue Visitor Tree"}]}
        ai.validate(result, ai.TRANSCRIPT)
        ai.sanitize(result, {7})
        self.assertEqual(result["transcriptions"][0]["number"], 7)

    def test_disallowed_page_and_non_web_source_are_removed(self):
        result = {
            "pages": [3, 99, 3, 1],
            "sources": [
                {"title": "author", "url": "https://example.org/paper"},
                {"title": "local", "url": "file:///tmp/paper"},
            ],
        }
        ai.sanitize(result, {1, 3})
        self.assertEqual(result["pages"], [3, 1])
        self.assertEqual([s["title"] for s in result["sources"]], ["author"])

    def test_dense_pages_keep_all_text_and_their_original_numbers(self):
        groups = list(ai.chunks(["abcdefgh", "", "xyz"], budget=3))
        parts = [part for group in groups for part in group]
        self.assertEqual("".join(part["text"] for part in parts), "abcdefghxyz")
        self.assertEqual([p["page"] for p in parts], [1, 1, 1, 3])

    def test_selection_works_across_pdf_line_breaks_and_ligatures(self):
        self.assertEqual(compact("efﬁcient nor-\nmalization"), compact("efficient normalization"))


if __name__ == "__main__":
    unittest.main()
