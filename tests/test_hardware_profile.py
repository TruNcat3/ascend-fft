import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_hardware_profile", ROOT / "scripts" / "check_hardware_profile.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class HardwareProfileTests(unittest.TestCase):
    def test_name_normalization_is_not_substring_matching(self):
        profile = {"soc": "Ascend910_9382"}
        self.assertTrue(checker.compatible(profile, "Ascend910-9382"))
        self.assertFalse(checker.compatible(profile, "Ascend910_9399"))
        self.assertFalse(checker.compatible(profile, "unknown"))

    def test_explicit_alias_is_supported(self):
        profile = {"soc": "Ascend910_9382", "compatible_device_names": ["vendor-name"]}
        self.assertTrue(checker.compatible(profile, "Vendor Name"))

    def test_cli_fails_closed_for_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps({"profile_id": "test", "soc": "expected"}))
            common = [str(path), "--target-soc", "expected"]
            self.assertEqual(checker.main(common + ["--device-name", "expected"]), 0)
            self.assertEqual(checker.main(common + ["--device-name", "other"]), 1)

    def test_cli_fails_closed_for_compiler_target_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps({"profile_id": "test", "soc": "expected"}))
            args = [str(path), "--device-name", "expected", "--target-soc", "other"]
            self.assertEqual(checker.main(args), 1)


if __name__ == "__main__":
    unittest.main()
