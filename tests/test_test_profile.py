import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_test_profile", ROOT / "scripts" / "run_test_profile.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TestProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = runner.load_config(ROOT / "config" / "test_matrix.json")

    def test_supported_profiles_have_stable_ids(self):
        for name, profile in self.document["profiles"].items():
            self.assertEqual(profile["id"], name)

    def test_smoke_expands_hardware_and_real_boundaries(self):
        cases = runner.expand_cases(self.document["profiles"]["smoke"])
        keys = {(row["direction"], row["n"], row["batch"], row["input"])
                for row in cases}
        self.assertIn(("c2c", 1024, 48, "sin"), keys)
        self.assertIn(("c2c", 1024, 49, "sin"), keys)
        self.assertIn(("r2c", 8192, 49, "sin"), keys)
        self.assertIn(("c2r", 64, 1, "half-spectrum"), keys)
        self.assertIn(("c2c", 4096, 49, "single-tone"), keys)

    def test_fixed_total_points_are_exact_and_deduplicated(self):
        profile = self.document["profiles"]["publication"]
        points = runner.c2c_points(profile)
        self.assertEqual(len(points), len(set(points)))
        self.assertIn((4096, 256), points)
        self.assertIn((4096, 4096), points)
        self.assertIn((64, 16384), points)

    def test_long_fft_profile_stays_non_runnable(self):
        self.assertFalse(self.document["profiles"]["future-long-fft"]["runnable"])
        self.assertIsNone(runner.dedicated_runner(self.document["profiles"]["future-long-fft"]))

    def test_stress_profile_dispatches_to_budgeted_runner(self):
        profile = self.document["profiles"]["stress"]
        self.assertTrue(profile["runnable"])
        runner_path = runner.dedicated_runner(profile)
        self.assertIsNotNone(runner_path)
        self.assertTrue(Path(runner_path).is_file())
        self.assertIsNone(runner.dedicated_runner(self.document["profiles"]["smoke"]))

    def test_forwarded_argv_drops_profile_and_keeps_flags(self):
        self.assertEqual(runner.forwarded_argv("stress", ["stress", "--dry-run"]),
                         ["--dry-run"])
        self.assertEqual(runner.forwarded_argv("stress", ["--config", "c.json"]),
                         ["--config", "c.json"])

    def test_dedicated_runner_rejects_generic_repetition_flags(self):
        with self.assertRaises(SystemExit):
            runner.main(["stress", "--reps", "3"])

    def test_seed_is_only_exported_when_nonzero(self):
        case = runner._case("numeric", "c2c", 64, 1, "random-seeded", 42)
        env, command = runner.command_for(case, 2)
        self.assertEqual(env["AB_SEED"], "42")
        self.assertEqual(command[-3:], ["64", "1", "2"])

    def test_all_declared_numeric_modes_are_implemented(self):
        source = (ROOT / "src" / "host" / "fft_check.cpp").read_text()
        patterns = {pattern for profile in self.document["profiles"].values()
                    for pattern in profile.get("numeric_patterns", [])}
        for pattern in patterns:
            self.assertIn(f'"{pattern}"', source)


if __name__ == "__main__":
    unittest.main()
