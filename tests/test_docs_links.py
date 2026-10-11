import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_docs_assets.py"
SPEC = importlib.util.spec_from_file_location("docs_links", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocsLinksTests(unittest.TestCase):
    def test_pages_assets_and_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide").mkdir()
            (root / "figures").mkdir()
            (root / "figures/a.svg").write_text("<svg/>")
            (root / "index.html").write_text(
                '<a href="guide/#section">guide</a><img src="figures/a.svg">'
                '<a href="https://example.com/missing">external</a>'
            )
            (root / "guide/index.html").write_text(
                '<h2 id="section">Section</h2><a href="#section">self</a>'
                '<a href="/ascend-fft/">home</a>'
            )
            checked, failures = MODULE.check_site(root, "/ascend-fft/")
            self.assertEqual(checked, 4)
            self.assertEqual(failures, [])

    def test_raw_markdown_page_and_missing_anchor_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text(
                '<a href="guide.md">bad page</a><a href="#missing">bad anchor</a>'
            )
            checked, failures = MODULE.check_site(root)
            self.assertEqual(checked, 2)
            self.assertEqual(len(failures), 2)
            self.assertTrue(any("missing target" in item for item in failures))
            self.assertTrue(any("missing anchor" in item for item in failures))

    def test_non_index_page_resolves_sibling_and_encoded_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folder").mkdir()
            (root / "folder/page.html").write_text('<a href="other.html#hello%20world">other</a>')
            (root / "folder/other.html").write_text('<a name="hello world">anchor</a>')
            self.assertEqual(MODULE.check_site(root)[1], [])

    def test_root_relative_link_must_include_pages_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text('<a href="/guide/">wrong root</a>')
            (root / "guide").mkdir()
            (root / "guide/index.html").write_text("guide")
            self.assertIn("outside site prefix", MODULE.check_site(root, "/ascend-fft/")[1][0])


if __name__ == "__main__":
    unittest.main()
