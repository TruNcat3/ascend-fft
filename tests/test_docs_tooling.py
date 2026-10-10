import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_results", ROOT / "scripts/publish_results.py")
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)

MATRIX = """# Matrix
| n | batch | ours | min | native | min | ratio | eta | dev | maxRel | correctness |
| 64 | 1 | 10.0 | 9.0 | 20.0 | 19.0 | **2.00×** | 9.0 | -10% | 1e-6 | PASS |
"""


class PublicationTests(unittest.TestCase):
    def make_run(self, directory):
        directory.mkdir()
        (directory / "matrix.md").write_text(MATRIX)
        (directory / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "commit": "test", "soc": "Ascend910_9382",
            "publication_status": "legacy-unverified",
            "provenance_limitations": ["test fixture"],
            "experiments": [{"name": "matrix", "status": "complete"}]}))

    def test_publication_is_deterministic_and_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            self.make_run(run)
            artifacts, generated = publish.build_artifacts(run)
            self.assertEqual(artifacts, publish.build_artifacts(run)[0])
            self.assertIn("2.000x", generated["comparison.md"].decode())
            published = root / "published"
            self.assertTrue(publish.synchronize(published, artifacts, True))
            self.assertFalse(published.exists())
            publish.synchronize(published, artifacts, False)
            self.assertFalse(publish.synchronize(published, artifacts, True))
            self.assertEqual(artifacts, publish.build_artifacts(published)[0])
            (published / "matrix.csv").write_text("stale")
            self.assertEqual(len(publish.synchronize(published, artifacts, True)), 1)

    def test_failed_or_missing_correctness_rejected(self):
        with self.assertRaises(ValueError):
            publish.parse_matrix(MATRIX.replace("PASS", "FAIL"))
        with self.assertRaises(ValueError):
            publish.validate_json("e2e.json", {"rows": [{"n": 64}]})
        with self.assertRaises(ValueError):
            publish.validate_json("e2e.json", {"rows": [{"ok": False}]})

    def test_sixway_detail_is_preserved_in_generated_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.make_run(run)
            (run / "sixway.md").write_text("# Six-way\n\nmatched detail\n")
            _, generated = publish.build_artifacts(run)
            comparison = generated["comparison.md"].decode()
            self.assertIn("Published Benchmark Summary", comparison)
            self.assertIn("matched detail", comparison)

    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.make_run(run)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["experiments"][0]["status"] = "running"
            (run / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                publish.build_artifacts(run)

    def test_current_publication_requires_clean_attributable_provenance(self):
        import hashlib
        profile = ROOT / "config/ascend910_93_profile.json"
        valid = {
            "schema_version": 3, "commit": "a" * 40, "dirty": False,
            "soc": "Ascend910_9382", "hardware_id": "device-0-redacted",
            "compiler_target": "Ascend910_9382",
            "cann_version": "9.0.0", "versions": {"ccec": "ccec 9.0.0"},
            "selected_profile": {"path": "config/ascend910_93_profile.json",
                                 "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                                 "soc": "Ascend910_9382"},
            "build_artifact_sha256": {"build/fft_check": "c" * 64},
        }
        self.assertEqual(publish.provenance_problems(valid), [])
        for field in ("soc", "hardware_id", "cann_version"):
            broken = dict(valid)
            broken[field] = "unknown"
            self.assertTrue(publish.provenance_problems(broken), field)
        dirty = dict(valid)
        dirty["dirty"] = True
        self.assertIn("source worktree was dirty", publish.provenance_problems(dirty))

    def test_schema_v2_requires_explicit_legacy_disclosure(self):
        manifest = {"schema_version": 2, "dirty": True}
        self.assertTrue(publish.provenance_problems(manifest))
        manifest.update({"publication_status": "legacy-unverified",
                         "provenance_limitations": ["source patch was not archived"]})
        self.assertEqual(publish.provenance_problems(manifest), [])

    def test_unknown_schema_and_invalid_hashes_are_rejected(self):
        self.assertTrue(publish.provenance_problems({"schema_version": 999}))
        manifest = {
            "schema_version": 3, "commit": "a" * 40, "dirty": False,
            "soc": "soc", "hardware_id": "device", "cann_version": "cann",
            "compiler_target": "soc",
            "versions": {"ccec": "compiler"},
            "selected_profile": {"path": "no-such", "sha256": "not-a-hash"},
            "build_artifact_sha256": {"build/x": "also-invalid"},
        }
        self.assertGreaterEqual(len(publish.provenance_problems(manifest)), 2)

    def test_schema_v3_requires_zero_experiment_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.make_run(run)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest.update({
                "schema_version": 3, "commit": "a" * 40, "dirty": False,
                "soc": "Ascend910_9382", "compiler_target": "Ascend910_9382",
                "hardware_id": "device", "cann_version": "9.0.0",
                "versions": {"ccec": "ccec 9.0.0"},
                "selected_profile": {
                    "path": "config/ascend910_93_profile.json",
                    "sha256": __import__("hashlib").sha256(
                        (ROOT / "config/ascend910_93_profile.json").read_bytes()).hexdigest(),
                    "soc": "Ascend910_9382"},
                "build_artifact_sha256": {"build/fft_check": "c" * 64},
            })
            manifest["experiments"][0]["exit_code"] = 1
            (run / "manifest.json").write_text(json.dumps(manifest))
            for name, data in {
                "summary.json": {"correct_points": 1, "total_points": 1,
                                 "rounds": 1, "native_enabled": False},
                "protocol.json": {"matrix": {"rounds": 1}},
            }.items():
                (run / name).write_text(json.dumps(data))
            (run / "trials.csv").write_bytes(
                b"runner,trial,n,batch,mean_us,min_us,max_rel,correct\n"
                b"self,1,64,1,10,9,1e-6,True\n")
            with self.assertRaisesRegex(ValueError, "zero exit_code"):
                publish.build_artifacts(run)

    def test_schema_v3_archives_and_hashes_declared_figure_outputs(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.make_run(run)
            profile = ROOT / "config/ascend910_93_profile.json"
            manifest = json.loads((run / "manifest.json").read_text())
            manifest.update({
                "schema_version": 3, "commit": "a" * 40, "dirty": False,
                "soc": "Ascend910_9382", "compiler_target": "Ascend910_9382",
                "hardware_id": "device", "cann_version": "9.0.0",
                "versions": {"ccec": "ccec 9.0.0"},
                "selected_profile": {
                    "path": "config/ascend910_93_profile.json",
                    "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                    "soc": "Ascend910_9382"},
                "build_artifact_sha256": {"build/fft_check": "c" * 64},
            })
            manifest["experiments"][0]["exit_code"] = 0
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "summary.json").write_text(json.dumps(
                {"correct_points": 1, "total_points": 1,
                 "rounds": 1, "native_enabled": False}))
            (run / "protocol.json").write_text(json.dumps({"matrix": {"rounds": 1}}))
            (run / "figures.json").write_text(json.dumps(
                {"overview_performance.png": "matrix.md"}))
            (run / "trials.csv").write_bytes(
                b"runner,trial,n,batch,mean_us,min_us,max_rel,correct\n"
                b"self,1,64,1,10,9,1e-6,True\n")
            artifacts, _ = publish.build_artifacts(run)
            figure_name = "figures/overview_performance.png"
            self.assertIn(figure_name, artifacts)
            published_manifest = json.loads(artifacts["manifest.json"])
            self.assertEqual(published_manifest["figure_output_sha256"][figure_name],
                             hashlib.sha256(artifacts[figure_name]).hexdigest())

    def test_raw_trial_csv_rejects_failed_or_nonfinite_rows(self):
        header = b"runner,trial,n,batch,mean_us,min_us,max_rel,correct\n"
        publish.validate_trials_csv(header + b"self,1,64,1,10,9,1e-6,True\n")
        for row in (b"self,1,64,1,nan,9,1e-6,True\n",
                    b"self,1,64,1,10,9,1e-6,False\n"):
            with self.assertRaises(ValueError):
                publish.validate_trials_csv(header + row)

    def test_raw_trial_coverage_rejects_missing_and_duplicate_rows(self):
        header = b"runner,trial,n,batch,mean_us,min_us,max_rel,correct\n"
        row = b"self,1,64,1,10,9,1e-6,True\n"
        summary = {"rounds": 1, "native_enabled": False}
        matrix = [{"n": 64, "batch": 1}]
        publish.validate_trial_coverage(header + row, summary, matrix)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            publish.validate_trial_coverage(header + row + row, summary, matrix)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            publish.validate_trial_coverage(
                header + b"self,1,128,1,10,9,1e-6,True\n", summary, matrix)

    def test_unavailable_optional_baseline_is_not_a_correctness_failure(self):
        publish.validate_json("e2e.json", {"rows": [{"ok": True, "bare_dev": float("nan"), "bare_ok": False}]})
        self.assertEqual(json.loads(publish.canonical({"missing": float("nan")})), {"missing": None})
        with self.assertRaises(ValueError):
            publish.validate_json("e2e.json", {"rows": [{"ok": True, "bare_dev": 2.0, "bare_ok": False}]})

    def test_record_preserves_multiple_experiments(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            publish.record_run(directory, "matrix", "command", None)
            publish.record_run(directory, "matrix", "command", 0)
            publish.record_run(directory, "e2e", "next", 1)
            entries = json.loads((directory / "manifest.json").read_text())["experiments"]
            self.assertEqual([entry["status"] for entry in entries], ["complete", "failed"])

    def test_explicit_output_remains_available(self):
        for name in ("gen_compare_doc.py", "gen_stdlib_doc.py", "matrix_test.py"):
            text = (ROOT / "scripts" / name).read_text()
            self.assertIn('ap.add_argument("--out"', text)
        text = (ROOT / "scripts/repro.sh").read_text()
        self.assertNotIn("--out docs/matrix_test", text)
        self.assertIn("results/runs/", text)


if __name__ == "__main__":
    unittest.main()
