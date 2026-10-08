import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def expand_shape_spec(spec):
    if "fixed_total_points" in spec:
        return [(n, total // n, total)
                for total in spec["fixed_total_points"] for n in spec["ns"]]
    return [(n, batch, n * batch)
            for n in spec.get("ns", []) for batch in spec.get("batches", [])]


def expand_experiment(item):
    specs = item.get("applications") or [item["shape_spec"]]
    cases = []
    for spec in specs:
        workload = spec.get("id", "")
        for n, batch, total in expand_shape_spec(spec):
            cases.append((workload, n, batch, total))
    return cases


class LongFftExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            (ROOT / "config" / "long_fft_experiments.json").read_text())

    def test_schema_and_stable_experiment_ids(self):
        self.assertEqual(self.document["schema_version"], 1)
        experiments = self.document["experiments"]
        self.assertEqual([item["id"] for item in experiments],
                         [f"E{i:02d}" for i in range(1, 9)])
        self.assertEqual([item["slug"] for item in experiments],
                         ["capacity", "latency", "equal-work", "saturation",
                          "ablation", "numerical", "applications", "profile"])
        required = {"id", "slug", "title", "runnable", "hypothesis",
                    "shape_spec", "controlled_variables", "metrics",
                    "baselines", "required_artifacts", "acceptance"}
        for item in experiments:
            self.assertTrue(required.issubset(item))
            for field in required - {"runnable", "shape_spec"}:
                self.assertTrue(item[field], (item["id"], field))

    def test_all_experiments_are_explicitly_not_runnable(self):
        self.assertEqual(self.document["status"], "planned-not-runnable")
        for item in self.document["experiments"]:
            self.assertIs(item["runnable"], False)

    def test_shapes_are_valid_and_unique_within_each_experiment(self):
        for item in self.document["experiments"]:
            specs = item.get("applications") or [item["shape_spec"]]
            for spec in specs:
                self.assertEqual(len(spec["ns"]), len(set(spec["ns"])))
            for _, n, _, _ in expand_experiment(item):
                self.assertGreaterEqual(n, 2048)
                self.assertLessEqual(n, 1048576)
                self.assertEqual(n & (n - 1), 0)
            shapes = expand_experiment(item)
            keys = [(workload, n, batch) for workload, n, batch, _ in shapes]
            self.assertEqual(len(keys), len(set(keys)), item["id"])
            self.assertTrue(all(batch > 0 for _, _, batch, _ in shapes))

    def test_equal_work_products_and_complete_length_sweep(self):
        item = next(item for item in self.document["experiments"]
                    if item["slug"] == "equal-work")
        spec = item["shape_spec"]
        self.assertEqual(spec["fixed_total_points"], [2**20, 2**24, 2**26])
        self.assertEqual(spec["ns"], [4096] + self.document["target"]["ns"])
        self.assertEqual(len(expand_shape_spec(spec)), 27)
        for n, batch, total in expand_shape_spec(spec):
            self.assertEqual(total % n, 0)
            self.assertEqual(n * batch, total)

    def test_primary_length_and_low_batch_coverage(self):
        self.assertEqual(self.document["target"]["primary_n"], 65536)
        for item in self.document["experiments"]:
            self.assertTrue(any(n == 65536 for _, n, _, _ in expand_experiment(item)), item["id"])
        latency = next(item for item in self.document["experiments"]
                       if item["slug"] == "latency")
        self.assertEqual(latency["shape_spec"]["batches"], [1, 4, 16])
        self.assertEqual(latency["shape_spec"]["ns"],
                         [2**power for power in range(13, 21)])

    def test_saturation_and_numerical_breakpoints_are_predeclared(self):
        saturation = next(item for item in self.document["experiments"]
                          if item["slug"] == "saturation")
        self.assertEqual(saturation["shape_spec"]["batches"],
                         [1, 4, 16, 32, 47, 48, 49, 64, 95, 96, 97, 128,
                          143, 144, 145, 191, 192, 193, 256, 512, 1024])
        self.assertEqual(len(expand_experiment(saturation)), 63)
        numerical = next(item for item in self.document["experiments"]
                         if item["slug"] == "numerical")
        self.assertEqual(numerical["shape_spec"]["batches"], [1, 3, 16])
        self.assertEqual(numerical["random_seeds"], [0, 1, 2])

    def test_application_cases_have_explicit_workload_identity(self):
        applications = next(item for item in self.document["experiments"]
                            if item["slug"] == "applications")
        self.assertEqual(applications["shape_spec"], {"derive_from": "applications"})
        cases = expand_experiment(applications)
        self.assertEqual(len(cases), 24)
        self.assertTrue(all(workload for workload, _, _, _ in cases))

    def test_ablation_pairs_have_explicit_controls(self):
        item = next(item for item in self.document["experiments"]
                    if item["slug"] == "ablation")
        pairs = item["pairs"]
        self.assertEqual(len(pairs), len({pair["id"] for pair in pairs}))
        for pair in pairs:
            self.assertNotEqual(pair["control"], pair["treatment"])
            self.assertTrue(pair["hold_fixed"])


if __name__ == "__main__":
    unittest.main()
