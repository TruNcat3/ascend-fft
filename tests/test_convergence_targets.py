"""Optimization tasks must match the immutable current NPU snapshot exactly."""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_convergence_targets as checker


class ConvergenceTargetsTest(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT / "config/convergence_targets.json").read_text())

    def test_current_manifest_passes_and_includes_near_unity_points(self):
        targets = checker.validate(self.document)
        self.assertEqual(len(targets), 14)
        self.assertEqual([sum(key[0] == op for key in targets)
                          for op in ("c2c", "r2c", "c2r")], [1, 12, 1])
        self.assertIn(("r2c", 128, 64), targets)
        self.assertIn(("r2c", 1024, 1024), targets)

    def test_omitted_target_fails(self):
        self.document["targets"].pop()
        with self.assertRaisesRegex(ValueError, "missing="):
            checker.validate(self.document)

    def test_extra_target_fails(self):
        extra = copy.deepcopy(self.document["targets"][0])
        extra["n"] = 64
        self.document["targets"].append(extra)
        with self.assertRaisesRegex(ValueError, "extra="):
            checker.validate(self.document)

    def test_cpu_baseline_is_rejected(self):
        self.document["baselines"]["c2c"]["runner"] = "torch-cpu"
        with self.assertRaisesRegex(ValueError, "baseline drift"):
            checker.validate(self.document)

    def test_different_snapshot_is_rejected(self):
        self.document["source_snapshot"] = "ascend910_9382-cann9.0.0-v2"
        with self.assertRaisesRegex(ValueError, "snapshot drift"):
            checker.validate(self.document)

    def test_changed_source_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "results/published" / checker.SNAPSHOT
            shutil.copytree(ROOT / "results/published" / checker.SNAPSHOT, destination)
            with (destination / "matrix.csv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "artifact drift: matrix.csv"):
                checker.validate(self.document, root)

    def test_report_parity_cannot_weaken_objective(self):
        self.document["objective"]["pointwise_minimum_speedup"] = 0.95
        with self.assertRaisesRegex(ValueError, "objective/report band drift"):
            checker.validate(self.document)

    def test_duplicate_target_is_rejected(self):
        self.document["targets"].append(copy.deepcopy(self.document["targets"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate manifest target"):
            checker.validate(self.document)


if __name__ == "__main__":
    unittest.main()
