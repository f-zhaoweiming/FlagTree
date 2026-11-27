"""Collection integration: preserve existing papers and restrict archived files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import library
from reading_collection import ReadingRoom


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.configs = self.root / "configs"
        self.configs.mkdir()
        (self.configs / "test.json").write_text(json.dumps({"root": "../bundle"}))
        (self.bundle / "chapter.html").write_text('<a id="sample"></a><h1>Sample</h1><p><a href="collections/sample/sources/example.html">Source</a></p>')
        (self.bundle / "sources").mkdir()
        (self.bundle / "sources/example.html").write_text('<script>alert("untrusted")</script>')
        self.manifest = {
            "id": "sample", "title": "Sample", "papers": [],
            "assets": ["sources/example.html"],
            "chapters": [{"key": "sample", "title": "Sample", "kind": "项目", "order": 50, "slug": "sample-project", "file": "chapter.html"}],
        }

    def load(self):
        (self.bundle / "collection.json").write_text(json.dumps(self.manifest))
        return ReadingRoom(self.root / "missing-original-book", self.configs)

    def test_source_navigation_and_escaped_html(self):
        room = self.load()
        self.assertIn('href="/archive/collections/sample/sources/example.html"', room.framed(room.chapters['sample']))
        body, mime = room.archive('collections/sample/sources/example.html')
        self.assertEqual(mime, 'text/html; charset=utf-8')
        self.assertNotIn(b'<script>', body)
        self.assertIn(b'&lt;script&gt;', body)

    def test_only_manifest_assets_are_accessible(self):
        (self.bundle / 'private.txt').write_text('not listed')
        room = self.load()
        for name in ['collections/sample/private.txt', 'collections/sample/../private.txt', '../private.txt']:
            with self.assertRaises(KeyError):
                room.archive_path(name)

    def test_manifest_cannot_escape_bundle(self):
        (self.root / 'outside.txt').write_text('outside')
        (self.bundle / 'outside-link.txt').symlink_to(self.root / 'outside.txt')
        for asset in ['../outside.txt', 'outside-link.txt']:
            self.manifest['assets'] = [asset]
            with self.assertRaises(ValueError):
                self.load()

    def test_existing_paper_and_reading_position_survive_attach(self):
        path = self.bundle / 'paper.pdf'
        path.write_bytes(b'%PDF-collection-test')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest['papers'] = [{"id": "existing-paper", "file": "paper.pdf", "sha256": digest}]
        self.manifest['assets'].append('paper.pdf')
        self.manifest['chapters'].append({"key": "paper", "title": "Paper", "kind": "论文", "order": 51, "slug": "existing-paper", "file": "paper.html"})
        (self.bundle / 'paper.html').write_text('<a id="paper"></a><h1>Paper</h1><a href="collections/sample/paper.pdf#page=2">PDF</a>')
        with patch.object(library, 'DATA', self.root / 'data'):
            library.init()
            with library.db() as conn:
                conn.execute('insert into papers(id,slug,title,short,path,count,current_page,guide) values(?,?,?,?,?,?,?,?)', (digest, 'existing-paper', 'Preserved title', 'Preserved short', str(path), 3, 2, '{"kept":true}'))
            room = self.load()
            room.attach()
            room.attach()
            with library.db() as conn:
                row = conn.execute('select * from papers where id=?', (digest,)).fetchone()
                self.assertEqual(row['current_page'], 2)
                self.assertEqual(row['title'], 'Preserved title')
                self.assertEqual(row['guide'], '{"kept":true}')
                self.assertEqual(conn.execute('select count(*) from papers').fetchone()[0], 2)
            self.assertIn('href="/#pdf/' + digest + '/2"', room.framed(room.chapters['paper']))

    def test_append_keeps_canonical_key_and_old_anchors(self):
        (self.bundle / 'audit.html').write_text('<a id="audit-version"></a><h2>Version audit</h2><p>Verified source.</p>')
        self.manifest['chapters'].append({
            'key': 'audit', 'title': 'Audit', 'kind': '项目', 'order': 90,
            'slug': 'sample-project', 'mode': 'append', 'file': 'audit.html',
            'version_notice': 'Latest code is primary.'})
        room = self.load()
        self.assertEqual(list(room.chapters), ['sample'])
        self.assertEqual(room.chapters['sample'].title, 'Sample')
        self.assertEqual(room.chapters['sample'].order, 50)
        self.assertEqual(room.anchors['audit-version'], 'sample')
        self.assertEqual(room.anchors['sample'], 'sample')
        self.assertIn('Verified source.', room.chapters['sample'].body)
        self.assertTrue(room.chapters['sample'].body.startswith('<p class="version-notice">'))
        self.assertIn('href="#audit-version"', room.chapters['sample'].body)

    def test_curated_guide_composes_without_changing_saved_guide(self):
        extra = {'overview': 'Pinned code audit', 'steps': [{'title': 'New step'}]}
        (self.bundle / 'guide.json').write_text(json.dumps(extra))
        self.manifest['guides'] = [{'slug': 'sample-project', 'file': 'guide.json'}]
        with patch.object(library, 'DATA', self.root / 'data'):
            library.init()
            room = self.load()
            room.attach()
            pid = next(iter(room.by_id))
            saved = {'overview': 'Previous guide', 'steps': [{'title': 'Old step'}]}
            with library.db() as conn:
                conn.execute('update papers set guide=?,current_page=7 where id=?', (json.dumps(saved), pid))
            first = room.guide_for(pid, saved)
            second = room.guide_for(pid, saved)
            self.assertEqual(first, second)
            self.assertEqual(len(first['steps']), 2)
            self.assertEqual(first['steps'][0]['title'], 'New step')
            self.assertEqual(len(saved['steps']), 1)
            with library.db() as conn:
                row = conn.execute('select guide,current_page from papers where id=?', (pid,)).fetchone()
                self.assertEqual(json.loads(row['guide']), saved)
                self.assertEqual(row['current_page'], 7)

    def test_append_rejects_duplicate_anchor(self):
        self.manifest['chapters'].append({**self.manifest['chapters'][0], 'mode': 'append'})
        with self.assertRaisesRegex(ValueError, '锚点重复'):
            self.load()


if __name__ == '__main__':
    unittest.main()
