import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "plot_results", ROOT / "scripts" / "plot_results.py")
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


class PlotResultsTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.md"
            path.write_text(content)
            return plots.parse_matrix(path)

    def test_compact_published_matrix_is_supported(self):
        rows = self.parse(
            "| N | Batch | Ascend-FFT (us) | CANN native (us) | Speedup |\n"
            "|---:|---:|---:|---:|---:|\n"
            "| 64 | 4 | 16.1 | 88.3 | 5.484x |\n")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["n"], rows[0]["b"]), (64, 4))
        self.assertAlmostEqual(rows[0]["ratio"], 5.484)
        self.assertNotEqual(rows[0]["eta"], rows[0]["eta"])

    def test_rich_matrix_preserves_model_fields(self):
        rows = self.parse(
            "| n | batch | ours | min | native | min | ratio | eta | dev | maxRel | ok |\n"
            "| 64 | 4 | 16.1 | 9.5 | 88.3 | 82.5 | **5.48x** | 17.8 | +10.6% | 7e-8 | PASS |\n")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["eta"], 17.8)
        self.assertAlmostEqual(rows[0]["dev"], 10.6)

    def test_application_shape_groups_are_disjoint_and_complete(self):
        groups = list(plots.APP_SHAPES.values())
        union = set().union(*groups)
        self.assertEqual(sum(map(len, groups)), 12)
        self.assertEqual(len(union), 12)


if __name__ == "__main__":
    unittest.main()
