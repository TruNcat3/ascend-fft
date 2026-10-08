import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_stress", ROOT / "scripts" / "run_stress.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

MEMINFO_FIXTURE = "MemTotal:  1 kB\nMemAvailable:   1332907400 kB\n"
HBM_FIXTURE = (
    "| 0     Ascend910           | OK            | 166.1                41 "
    "                      0    / 0                0    / 0                |\n"
    "| 0     0                   | 0000:9D:00.0  | 0                    0"
    "    / 0                3122 / 65536            |\n"
    "| 1     Ascend910           | OK            | -                    40"
    "                      0    / 0                0    / 0                |\n"
    "| 1     1                   | 0000:9F:00.0  | 0                    0"
    "    / 0                2887 / 65536            |\n"
)


class TestStressRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = runner.load_config(ROOT / "config" / "test_matrix.json")
        cls.profile = cls.document["profiles"]["stress"]
        cls.big_host = 1024 * runner.GIB
        cls.big_device_free = 60 * runner.GIB
        cls.big_device_total = 64 * runner.GIB

    def test_stress_profile_declares_budgeted_runner(self):
        self.assertTrue(self.profile["runnable"])
        self.assertTrue((ROOT / self.profile["runner"]).is_file())

    def test_derive_batch_matches_input_buffer_tiers(self):
        self.assertEqual(runner.derive_batch(1024, runner.MIB, 8), 128)
        self.assertEqual(runner.derive_batch(8192, runner.MIB, 4), 32)
        self.assertEqual(runner.derive_batch(64, 32, 8), 0)
        self.assertEqual(runner.derive_batch(0, runner.MIB, 8), 0)

    def test_budget_runs_when_headroom_covers_estimate(self):
        budget = runner.budget_case(runner.GIB, self.big_host,
                                    self.big_device_free, self.big_device_total)
        self.assertEqual(budget["decision"], "run")
        self.assertEqual(budget["host_needed"],
                         int(runner.GIB * runner.HOST_FACTOR) + runner.HOST_FIXED)
        self.assertEqual(budget["device_needed"],
                         int(runner.GIB * runner.DEVICE_FACTOR) + runner.DEVICE_FIXED)

    def test_budget_refuses_when_device_headroom_would_break(self):
        budget = runner.budget_case(runner.GIB, self.big_host, 5 * runner.GIB,
                                    self.big_device_total)
        self.assertEqual(budget["decision"], "skip")
        self.assertIn("device budget", budget["reason"])

    def test_budget_refuses_when_host_headroom_would_break(self):
        budget = runner.budget_case(256 * runner.MIB, 512 * runner.MIB,
                                    self.big_device_free, self.big_device_total)
        self.assertEqual(budget["decision"], "skip")
        self.assertIn("host budget", budget["reason"])

    def test_budget_refuses_offset_limit_shapes(self):
        budget = runner.budget_case(5 * runner.GIB, self.big_host,
                                    self.big_device_free, self.big_device_total)
        self.assertEqual(budget["decision"], "skip")
        self.assertIn("offset-limit", budget["reason"])

    def test_plan_covers_all_tiers_and_soaks_canonical_shape(self):
        rows = runner.plan(self.profile, self.big_host, self.big_device_free,
                           self.big_device_total)
        tiers = {row["tier_bytes"] for row in rows}
        self.assertEqual(tiers, set(self.profile["input_buffer_bytes"]))
        self.assertTrue(all(row["decision"] == "run" for row in rows))
        soaks = [row for row in rows if row["stage"] == "soak"]
        self.assertEqual(len(soaks), len(self.profile["input_buffer_bytes"]))
        for row in soaks:
            self.assertEqual(row["n"], 1024)
            self.assertEqual(row["reps"], self.profile["repeat_execution"])
        c2r = [row for row in rows if row["direction"] == "c2r"]
        self.assertTrue(c2r)
        self.assertTrue(all(row["input"] == "half-spectrum" for row in c2r))
        self.assertTrue(all(row["input"] == "random-seeded"
                            for row in rows if row["direction"] in ("c2c", "r2c")))

    def test_plan_skips_tiers_that_cannot_hold_one_row(self):
        tiny = dict(self.profile, input_buffer_bytes=[32])
        rows = runner.plan(tiny, self.big_host, self.big_device_free,
                           self.big_device_total)
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["decision"], "skip")
            self.assertIn("cannot hold", row["reason"])

    def test_case_timeout_grows_with_reps(self):
        self.assertGreaterEqual(runner.case_timeout_s(10000), 300)
        self.assertGreater(runner.case_timeout_s(10000), runner.case_timeout_s(3))
        self.assertLessEqual(runner.case_timeout_s(10 ** 9), 1800)

    def test_meminfo_parser_reads_memavailable(self):
        with tempfile.NamedTemporaryFile("w", suffix=".meminfo",
                                         delete=False) as handle:
            handle.write(MEMINFO_FIXTURE)
            path = handle.name
        try:
            self.assertEqual(runner.read_host_available(path), 1332907400 * 1024)
        finally:
            Path(path).unlink()

    def test_meminfo_parser_requires_memavailable(self):
        with tempfile.NamedTemporaryFile("w", suffix=".meminfo",
                                         delete=False) as handle:
            handle.write("MemTotal: 1 kB\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                runner.read_host_available(path)
        finally:
            Path(path).unlink()

    def test_hbm_parser_reads_chip_row(self):
        used, total = runner.parse_hbm_table(HBM_FIXTURE, chip=0)
        self.assertEqual(used, 3122 * runner.MIB)
        self.assertEqual(total, 65536 * runner.MIB)
        used, total = runner.parse_hbm_table(HBM_FIXTURE, chip=1)
        self.assertEqual(used, 2887 * runner.MIB)

    def test_hbm_parser_rejects_tables_without_chip(self):
        with self.assertRaises(ValueError):
            runner.parse_hbm_table("| 0     Ascend910 | OK |\n", chip=0)

    def test_rss_growth_needs_steady_state_samples(self):
        self.assertIsNone(runner.rss_growth_bytes([(0.0, 100), (0.5, 100)]))
        samples = [(index * 0.5, 1000) for index in range(4)]
        samples += [(index * 0.5, 1000 + index * 10) for index in range(4, 40)]
        growth = runner.rss_growth_bytes(samples)
        self.assertGreater(growth, 0)
        self.assertGreaterEqual(runner.rss_limit_bytes(200 * runner.MIB),
                                96 * runner.MIB)

    def test_dry_run_budgets_without_touching_hardware(self):
        buffer = io.StringIO()
        with mock.patch.object(runner, "read_host_available",
                               return_value=8 * runner.GIB), \
             mock.patch.object(runner, "read_hbm",
                               return_value=(63 * runner.GIB, 64 * runner.GIB)), \
             redirect_stdout(buffer):
            code = runner.main(["--dry-run"])
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("plan rows=", output)
        self.assertIn("device budget", output)


if __name__ == "__main__":
    unittest.main()
