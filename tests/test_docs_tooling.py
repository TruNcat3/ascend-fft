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
