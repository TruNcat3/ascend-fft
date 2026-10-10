import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("matrix_test", ROOT / "scripts" / "matrix_test.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def result(text, code=0):
    return subprocess.CompletedProcess("mock", code, text, "")


def self_trial(error="1e-6", status="PASS", mean=12, minimum=10, code=0):
    return result(f"kfft_fwd: {mean} us/call (min {minimum} us)\nmaxRel={error}\n{status}\n", code)


def native_trial(error="1e-6", status="PASS", code=0):
    return result(f"NATIVE n=64 b=1 native_us=20 native_mean_us=25 maxRel={error} {status}\n", code)


class MatrixGateTests(unittest.TestCase):
    def run_matrix(self, records, native=True, rounds=2):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "matrix.md"
            arguments = ["--ns", "64", "--bs", "1", "--rounds", str(rounds),
                         "--no-eta", "--out", str(destination)]
            if not native:
                arguments.append("--no-native")
            else:
                native_records, self_records = records[:rounds], records[rounds:]
                interleaved = []
                for trial, order in enumerate(matrix.runner_orders(rounds, True, 0)):
                    for runner_name in order:
                        interleaved.append(native_records[trial] if runner_name == "native"
                                           else self_records[trial])
                records = interleaved
            with patch.object(matrix, "sh", side_effect=records) as runner:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = matrix.main(arguments)
            summary = json.loads((destination.parent / "summary.json").read_text())
            protocol = json.loads((destination.parent / "protocol.json").read_text())
            with (destination.parent / "matrix.csv").open() as handle:
                row = next(csv.DictReader(handle))
            with (destination.parent / "trials.csv").open() as handle:
                trials = list(csv.DictReader(handle))
            self.assertEqual(runner.call_count, len(records))
            self.assertEqual(protocol["c2c_matrix"]["rounds"], rounds)
            self.assertEqual(protocol["c2c_matrix"]["raw_trials"], "trials.csv")
            return code, summary, row, trials

    def test_all_trials_pass_and_worst_error_is_retained(self):
        code, summary, row, trials = self.run_matrix([
            native_trial("8e-6"), native_trial("2e-6"),
            self_trial("9e-6", mean=14), self_trial("1e-6", mean=12)])
        self.assertEqual(code, 0)
        self.assertEqual(float(row["max_rel"]), 9e-6)
        self.assertEqual(float(row["ours_us"]), 12)
        self.assertEqual(summary["native_max_rel"][0]["max_rel"], 8e-6)
        self.assertEqual(summary["failed_trials"], 0)
        self.assertEqual(summary["total_trials"], 4)
        self.assertEqual(len(summary["runner_orders"]), 2)
        self.assertEqual(len(trials), 4)

    def test_later_self_pass_cannot_hide_failure(self):
        code, summary, row, _ = self.run_matrix([
            self_trial(status="FAIL"), self_trial()], native=False)
        self.assertEqual(code, 1)
        self.assertEqual(row["correct"], "False")
        self.assertEqual(summary["failed_trials"], 1)

    def test_self_nonzero_exit_cannot_pass(self):
        code, summary, _, _ = self.run_matrix([self_trial(code=1), self_trial()], native=False)
        self.assertEqual(code, 1)
        self.assertEqual(summary["failures"][0]["returncode"], 1)

    def test_missing_self_output_fails_and_summary_has_no_nan(self):
        code, summary, _, _ = self.run_matrix([result(""), self_trial()], native=False)
        self.assertEqual(code, 1)
        self.assertIsNone(summary["failures"][0]["max_rel"])

    def test_self_error_threshold_is_enforced_without_trusting_marker(self):
        code, _, _, _ = self.run_matrix([self_trial("2e-4"), self_trial()], native=False)
        self.assertEqual(code, 1)

    def test_missing_self_minimum_is_rejected(self):
        code, _, _, _ = self.run_matrix([
            result("kfft_fwd: 12 us/call\nmaxRel=1e-6\nPASS\n"), self_trial()], native=False)
        self.assertEqual(code, 1)

    def test_native_failure_cannot_be_hidden_by_other_round(self):
        code, summary, row, _ = self.run_matrix([
            native_trial(status="FAIL"), native_trial(), self_trial(), self_trial()])
        self.assertEqual(code, 1)
        self.assertEqual(row["correct"], "False")
        self.assertEqual(summary["failures"][0]["runner"], "native")

    def test_native_point_coverage_is_checked_each_round(self):
        code, summary, _, _ = self.run_matrix([
            result("NPU unavailable\n"), native_trial(), self_trial(), self_trial()])
        self.assertEqual(code, 1)
        self.assertIn("found 0", summary["failures"][0]["failure"])

    def test_duplicate_native_records_fail_coverage(self):
        duplicate = native_trial().stdout * 2
        code, summary, _, _ = self.run_matrix([
            result(duplicate), native_trial(), self_trial(), self_trial()])
        self.assertEqual(code, 1)
        self.assertIn("found 2", summary["failures"][0]["failure"])

    def test_native_nonzero_exit_and_large_error_are_rejected(self):
        for failure in (native_trial(code=1), native_trial(error="2e-4")):
            with self.subTest(failure=failure):
                code, _, _, _ = self.run_matrix([
                    failure, native_trial(), self_trial(), self_trial()])
                self.assertEqual(code, 1)

    def test_no_native_gate_accepts_complete_self_trials(self):
        code, summary, _, _ = self.run_matrix([self_trial(), self_trial()], native=False)
        self.assertEqual(code, 0)
        self.assertFalse(summary["native_enabled"])
        self.assertEqual(summary["total_trials"], 2)

    def test_timeout_is_a_failed_command_result(self):
        with patch.object(matrix.subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 3)):
            timed_out = matrix.sh("mock", timeout=3)
        self.assertEqual(timed_out.returncode, 124)
        self.assertIn("timeout", timed_out.stderr)


if __name__ == "__main__":
    unittest.main()
