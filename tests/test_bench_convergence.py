"""bench_convergence runner tests (PR #6 targeted convergence, R0)."""
import json
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

    def test_candidate_closure(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_convergence as bc
        cands = bc.parse_candidates(["foldD=16", "foldD=8,planeK=8"])
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0]["overrides"], {"foldD": "16"})
        self.assertEqual(cands[1]["overrides"], {"foldD": "8", "planeK": "8"})
        self.assertEqual(cands[0]["id"], "foldD16")

    def test_resolve_mapping(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_convergence as bc
        # r2c n=128 (inner=64), B=4096: cap=16, ceil(4096/16)=256>=96 -> D=16
        m = bc.resolve_mapping("r2c", 128, 4096, {})
        self.assertEqual(m["fold_d"], 16)
        self.assertEqual(m["plane_k"], 8)
        self.assertEqual(m["inner_n"], 64)
        # r2c n=1024 (inner=512), B=4096: cap=4 -> D=4
        m = bc.resolve_mapping("r2c", 1024, 4096, {})
        self.assertEqual(m["fold_d"], 4)
        # override wins
        m = bc.resolve_mapping("r2c", 128, 4096, {"foldD": "4"})
        self.assertEqual(m["fold_d"], 4)

    def test_binary_sha(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_convergence as bc
        sha = bc.binary_sha256(ROOT / "config" / "convergence_targets.json")
        self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main()
