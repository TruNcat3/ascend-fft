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

    def test_upstream_concept_links_use_the_live_default_branch(self):
        for relative in ("docs/index.md", "docs/design/architecture.md"):
            content = (ROOT / relative).read_text()
            self.assertIn("cuButterfly/blob/main/figures/cubutterfly_concept.svg", content)
            self.assertNotIn("cuButterfly/blob/master/figures/cubutterfly_concept.svg", content)

    def test_algorithm_core_and_hardware_lowering_are_not_conflated(self):
        motivation = (ROOT / "docs/design/motivation.md").read_text()
        architecture = (ROOT / "docs/design/architecture.md").read_text()
        self.assertIn("算法核心空间 P", motivation)
        self.assertIn("hardware lowering 是对每个格子的门禁", architecture)
        self.assertNotIn("计算核心空间 P = {radix-2, radix-4, vector, Cube", motivation)

    def test_hybrid_unfolding_example_preserves_claim_boundary(self):
        motivation = (ROOT / "docs/design/motivation.md").read_text()
        figure = ROOT / "docs/figures/hybrid_unfolding_example.svg"
        self.assertTrue(figure.is_file())
        self.assertIn("(Us, Ts, Ud, Td) = (2, 2, 4, 2)", motivation)
        self.assertIn("Qj = [x[j], x[j+4], x[j+8], x[j+12]]", motivation)
        self.assertIn("NOT LOWERED", motivation)
        figure_text = figure.read_text()
        self.assertIn("方法抽象来源：cuButterfly", figure_text)
        self.assertIn("教学实例，不是当前 N=16 部署配置", figure_text)
        renderer = (ROOT / "scripts/render_docs_diagrams.py").read_text()
        self.assertIn('"hybrid_unfolding_example"', renderer)

    def test_roofline_insight_is_quantified_and_keeps_evidence_boundary(self):
        motivation = (ROOT / "docs/design/motivation.md").read_text()
        figure = ROOT / "docs/figures/roofline_insight.svg"
        self.assertTrue(figure.is_file())
        self.assertIn("Q_GM,payload = compulsory input/output + materialized boundaries", motivation)
        self.assertIn("= 2 * B * N * q * G", motivation)
        self.assertIn("规模增大不等于强度必然上升", motivation)
        self.assertIn("I_GM` 近似正比于 `K/G", motivation)
        self.assertIn("C=Us*Ud", motivation)
        self.assertIn("符号化上界示意", figure.read_text())
        self.assertIn("不是 Ascend 实测 Roofline", figure.read_text())
        renderer = (ROOT / "scripts/render_docs_diagrams.py").read_text()
        self.assertIn('"roofline_insight"', renderer)

    def test_statistical_tolerance_is_not_called_bandwidth(self):
        for relative in (
            "README.md",
            "docs/index.md",
            "docs/benchmarks/results.md",
            "docs/benchmarks/methodology.md",
        ):
            content = (ROOT / relative).read_text()
            self.assertNotIn("±5% 带宽", content)


if __name__ == "__main__":
    unittest.main()
