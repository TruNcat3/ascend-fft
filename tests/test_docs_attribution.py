from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AttributionTests(unittest.TestCase):
    def test_author_identity_is_visible_from_both_entry_points(self):
        for relative in ("README.md", "docs/index.md", "docs/about.md"):
            content = (ROOT / relative).read_text()
            self.assertIn("Teng Wang", content)
            self.assertIn("High Efficient Intelligent Computing Lab", content)
            self.assertIn("wangt635@ustc.edu.cn", content)

    def test_institution_mark_has_source_and_license_boundary(self):
        self.assertTrue((ROOT / "docs/assets/ustc-logo.png").is_file())
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
        self.assertIn("official USTC website", notice)
        self.assertIn("not granted under this repository's Apache-2.0 license", notice)
        self.assertIn("does not imply", notice)

    def test_about_page_is_in_site_navigation(self):
        config = (ROOT / "mkdocs.yml").read_text()
        self.assertIn("项目与作者: about.md", config)
        self.assertIn("stylesheets/extra.css", config)


if __name__ == "__main__":
    unittest.main()
