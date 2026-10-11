"""bench_convergence runner tests (PR #6 targeted convergence)."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BenchConvergenceTests(unittest.TestCase):
    def test_manifest_loads(self):
        doc = json.loads((ROOT / "config" / "convergence_targets.json").read_text())
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(len(doc["targets"]), 14)
        ops = {t["operator"] for t in doc["targets"]}
        self.assertEqual(ops, {"c2c", "r2c", "c2r"})

    def test_reps_policy(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_convergence as bc
        self.assertEqual(bc.reps_for("c2c", 1024, 4096), 20)
        self.assertEqual(bc.reps_for("r2c", 64, 1), 50)     # nB=64 <= 4096
        self.assertEqual(bc.reps_for("r2c", 128, 64), 30)   # nB=8192 <= 2^20
        self.assertEqual(bc.reps_for("r2c", 8192, 1024), 10)  # nB > 2^20

    def test_source_speedups(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_convergence as bc
        doc = bc.load_targets()
        sp = bc.load_source_speedups(doc)
        self.assertIn(("c2c", 1024, 4096), sp)
        self.assertLess(sp[("c2c", 1024, 4096)], 1.0)
        self.assertIn(("r2c", 128, 4096), sp)
        self.assertLess(sp[("r2c", 128, 4096)], 0.5)


if __name__ == "__main__":
    unittest.main()
