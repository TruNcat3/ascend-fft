"""P0 baseline tooling: parsing, speedup math, deterministic docs table."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import bench_long_baseline as bl  # noqa: E402


def _full_native_output():
    lines = []
    for n in bl.NS:
        for b in bl.BS:
            lines.append(
                f"NATIVE n={n} b={b} native_us=100.0 native_mean_us=110.0 "
                f"maxRel=1.2e-07 PASS")
            lines.append(
                f"NATIVE_E2E n={n} b={b} e2e_us=200.0 e2e_mean_us=210.0 "
                f"first_us=900.0 maxRel=1.2e-07 PASS")
    return "\n".join(lines)


def _doc():
    shapes = []
    for n in bl.NS:
        for b in bl.BS:
            row = {"n": n, "b": b,
                   "native_device_min_us": 100.0,
                   "native_device_mean_us": 110.0,
                   "native_max_rel": 1.2e-7, "native_pass": True,
                   "native_e2e_min_us": 200.0, "native_e2e_mean_us": 210.0,
                   "native_first_us": 900.0, "native_e2e_max_rel": 1.2e-7,
                   "native_e2e_pass": True,
                   "ours_chain_median_us": 50.0,
                   "ours_e2e_median_us": 150.0, "ours_e2e_min_us": 140.0,
                   "ours_max_rel": 3e-7}
            row["speedup_device_native_over_ours"] = bl._ratio(
                row["native_device_min_us"], row["ours_chain_median_us"])
            row["speedup_e2e_native_over_ours"] = bl._ratio(
                row["native_e2e_min_us"], row["ours_e2e_median_us"])
            shapes.append(row)
    return {"schema": "long-fft-baseline/1", "threshold": 1e-4,
            "manifest": {"git_sha": "0" * 40, "torch": "2.1",
                         "torch_npu": "2.1", "cann": "9.0.0",
                         "ours_git_sha": "1" * 40,
                         "ours_binary_sha256": "ab" * 32},
            "grid": {"ns": list(bl.NS), "bs": list(bl.BS)},
            "shapes": shapes}


class ParseNativeTest(unittest.TestCase):
    def test_full_grid_parsed(self):
        shapes = bl.parse_native_output(_full_native_output())
        self.assertEqual(len(shapes), len(bl.NS) * len(bl.BS))
        s = shapes[(65536, 47)]
        self.assertEqual(s["native_device_min_us"], 100.0)
        self.assertEqual(s["native_e2e_min_us"], 200.0)
        self.assertTrue(s["native_pass"])

    def test_incomplete_grid_rejected(self):
        partial = ("NATIVE n=8192 b=1 native_us=100.0 native_mean_us=110.0 "
                   "maxRel=1.2e-07 PASS\n"
                   "NATIVE_E2E n=8192 b=1 e2e_us=200.0 e2e_mean_us=210.0 "
                   "first_us=900.0 maxRel=1.2e-07 PASS\n")
        with self.assertRaises(ValueError):
            bl.parse_native_output(partial)

    def test_e2e_line_missing_rejected(self):
        only_dev = "\n".join(
            l for l in _full_native_output().splitlines()
            if l.startswith("NATIVE "))
        with self.assertRaises(ValueError):
            bl.parse_native_output(only_dev)


class RenderTest(unittest.TestCase):
    def test_render_deterministic(self):
        self.assertEqual(bl.render_markdown(_doc()),
                         bl.render_markdown(_doc()))

    def test_speedup_math(self):
        self.assertEqual(bl._ratio(100.0, 50.0), 2.0)
        self.assertIsNone(bl._ratio(None, 50.0))
        self.assertIsNone(bl._ratio(100.0, 0))

    def test_table_contains_speedups_and_status(self):
        md = bl.render_markdown(_doc())
        self.assertIn("2.00x", md)
        self.assertIn("PASS", md)
        self.assertIn("| 65536 | 47 |", md)


class ArchiveConsistencyTest(unittest.TestCase):
    @unittest.skipUnless(bl.OUT_JSON.is_file(),
                         "baseline archive not collected yet")
    def test_committed_md_matches_archive(self):
        doc = json.loads(bl.OUT_JSON.read_text(encoding="utf-8"))
        problems, _ = bl.check()
        self.assertEqual(problems, [])
        self.assertEqual(len(doc["shapes"]), 12)
        self.assertEqual(doc["manifest"]["git_dirty"], False)


if __name__ == "__main__":
    unittest.main()
