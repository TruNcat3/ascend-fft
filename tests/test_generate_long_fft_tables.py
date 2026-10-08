import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("long_fft_tables", ROOT / "scripts" / "generate_long_fft_tables.py")
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


def manifest(runnable=False):
    return {"experiments": [{
        "id": "E01", "slug": "capacity", "title": "Capacity", "runnable": runnable,
        "hypothesis": "Test the cross-UB boundary.", "shape_spec": {"ns": [8192, 65536], "batches": [1, 16]},
        "metrics": ["kernel_ms", {"name": "max_error", "description": "Maximum observed error."}],
    }]}


class LongFftTableTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        first = tables.generate(manifest())
        self.assertEqual(first, tables.generate(manifest()))
        self.assertIn("| E01-N65536-B16 | 65536 | 16 | 1048576 |  | C2C | fp32 | planned |", first)

    def test_schema_has_traceable_raw_trial_fields(self):
        result = tables.generate(manifest())
        self.assertIn("experiment_id,case_id,N,batch,total_points,transform,precision,direction,normalization,placement,layout,timing_scope", result)
        self.assertIn("application_id,input_mode,seed", result)
        self.assertIn("implementation,configuration_id,selected_config,warmup,repeat,trial,status,kernel_ms,max_error", result)
        self.assertIn("No result rows are emitted", result)
        self.assertIn("|  |  |  |  |  |  |", result)

    def test_runnable_is_not_a_measured_result(self):
        result = tables.generate(manifest(True))
        self.assertIn("**Status:** not measured", result)
        self.assertNotIn("**Status:** passed", result)
        self.assertNotIn("1.000", result)

    def test_check_does_not_write_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, output = Path(temporary) / "config.json", Path(temporary) / "out.md"
            config.write_text(json.dumps(manifest()), encoding="utf-8")
            self.assertEqual(tables.main(["--config", str(config), "--out", str(output), "--check"]), 1)
            self.assertFalse(output.exists())
            self.assertEqual(tables.main(["--config", str(config), "--out", str(output)]), 0)
            self.assertEqual(tables.main(["--config", str(config), "--out", str(output), "--check"]), 0)
            output.write_text("outdated", encoding="utf-8")
            self.assertEqual(tables.main(["--config", str(config), "--out", str(output), "--check"]), 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "outdated")

    def test_duplicate_result_columns_rejected(self):
        document = manifest()
        document["experiments"][0]["metrics"] = ["trial"]
        with self.assertRaisesRegex(ValueError, "duplicate common"):
            tables.generate(document)

    def test_equal_work_has_exact_total_and_stable_ids(self):
        experiment = manifest()["experiments"][0]
        experiment["shape_spec"] = {"ns": [8192, 65536], "fixed_total_points": [1048576]}
        shapes = tables.expand_shapes(experiment)
        self.assertEqual([shape["batch"] for shape in shapes], [128, 16])
        self.assertEqual(shapes[1]["id"], "E01-N65536-B16-P1048576")
        self.assertTrue(all(shape["n"] * shape["batch"] == shape["total_points"] for shape in shapes))
        experiment["shape_spec"]["fixed_total_points"] = [1048577]
        with self.assertRaisesRegex(ValueError, "divisible"):
            tables.expand_shapes(experiment)

    def test_application_proxies_preserve_workload_identity(self):
        experiment = manifest()["experiments"][0]
        experiment["shape_spec"] = {"derive_from": "applications"}
        experiment["applications"] = [{"id": "convolution-proxy", "ns": [65536], "batches": [4]}]
        shapes = tables.expand_shapes(experiment)
        self.assertEqual(shapes[0]["workload"], "convolution-proxy")
        self.assertEqual(len(shapes), 1)

    def test_derived_shapes_require_applications(self):
        document = manifest()
        document["experiments"][0]["shape_spec"] = {"derive_from": "applications"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty applications"):
                tables.load_manifest(path)

    def test_repository_manifest_and_generated_artifact_are_consistent(self):
        document = tables.load_manifest(tables.DEFAULT_CONFIG)
        self.assertEqual(len(document["experiments"]), 8)
        self.assertEqual(tables.DEFAULT_OUT.read_text(encoding="utf-8"), tables.generate(document))


if __name__ == "__main__":
    unittest.main()
