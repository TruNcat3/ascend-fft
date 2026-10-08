"""Architecture descriptor legality tests (addendum step 2).

Hardware-independent: compiles tests/test_descriptors.cpp with plain g++ (no
CANN), runs its machine-readable case table, and checks
  - every unsupported tuple carries an explicit reason,
  - the builtin H profile stays in sync with config/ascend910_93_profile.json,
  - the current G1 chain reports itself honestly (host-assisted, not on-chip),
  - batch only adds data-dimension work (identical lowering structure),
  - fft_check queries the lowering contract before any allocation or launch.
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CASES = {
    "default_8192": (True, None),
    "default_65536_b3": (True, None),
    "batch_4096_independent": (True, None),
    "cell_resident_ok": (True, None),
    "precision_fp16": (False, "precision mismatch"),
    "direction_r2c_unmapped": (False, "direction"),
    "bad_power_of_two": (False, "power of two"),
    "stage_outside_unit_range": (False, "outside unit"),
    "ud_exceeds_aiv": (False, "AIV count"),
    "us_exceeds_aiv": (False, "AIV count"),
    "partition_mismatch": (False, "stage_partition inconsistent"),
    "role_out_of_range": (False, "role allocation"),
    "stage_product_mismatch": (False, "does not match transform length"),
    "layout_real_on_c2c": (False, "not supported by unit"),
    "ub_overflow": (False, "UB overflow"),
    "block_resident_needs_multi_role": (False, "multi-role unit"),
    "onchip_without_block_residence": (False, "on-chip boundary"),
    "gm_guard_overflow": (False, "40 GiB guard"),
}


def _parse(stdout):
    h, unit, cases, structs = {}, None, {}, {}
    for line in stdout.splitlines():
        parts = line.split(",")
        if parts[0] == "H" and len(parts) == 3:
            h[parts[1]] = parts[2]
        elif parts[0] == "U":
            unit = parts[1:]
        elif parts[0] == "CASE":
            cases[parts[1]] = (parts[2], ",".join(parts[3:]))
        elif parts[0] == "STRUCT":
            name = parts[1]
            kv = parts[2:]
            structs[name] = dict(zip(kv[0::2], kv[1::2]))
    return h, unit, cases, structs


class DescriptorLegalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gxx = shutil.which("g++") or shutil.which("c++")
        if not gxx:
            raise unittest.SkipTest("no C++ compiler; descriptor self-test skipped")
        cls._tmp = tempfile.TemporaryDirectory()
        binary = Path(cls._tmp.name) / "test_descriptors"
        subprocess.run(
            [gxx, "-std=c++17", "-Wall", "-I", str(ROOT / "include"),
             str(ROOT / "tests" / "test_descriptors.cpp"), "-o", str(binary)],
            check=True, capture_output=True, timeout=120,
        )
        proc = subprocess.run([str(binary)], check=True, capture_output=True,
                              text=True, timeout=60)
        cls.h, cls.unit, cls.cases, cls.structs = _parse(proc.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_case_table_complete(self):
        self.assertEqual(set(self.cases), set(EXPECTED_CASES),
                         "self-test case table changed; update EXPECTED_CASES")

    def test_supported_and_rejected_with_explicit_reasons(self):
        for name, (want_supported, want_reason) in EXPECTED_CASES.items():
            status, reason = self.cases[name]
            with self.subTest(case=name):
                self.assertEqual(status, "SUPPORTED" if want_supported else "UNSUPPORTED")
                if want_supported:
                    self.assertEqual(reason, "", f"{name} should carry no reason")
                else:
                    self.assertTrue(reason.strip(),
                                    f"{name} rejected without an explicit reason")
                    self.assertIn(want_reason, reason)

    def test_hardware_profile_matches_config(self):
        profile = json.loads(
            (ROOT / "config" / "ascend910_93_profile.json").read_text())
        for key in ("profile_id", "soc", "aicore_num", "vector_core_num",
                    "l2_bytes", "global_mem_total_bytes", "ub_bytes_per_core",
                    "vec_lane_fp32"):
            with self.subTest(key=key):
                self.assertEqual(self.h[key], str(profile[key]),
                                 f"builtin H profile drifted from config for {key}")

    def test_local_fft_unit_envelope(self):
        core, min_log, max_log, ub, scope, multi = self.unit
        self.assertEqual(core, "kfft_fwd_local_fft")
        self.assertEqual(min_log, "6")
        self.assertEqual(max_log, "12")
        self.assertEqual(ub, "196608")
        self.assertEqual(scope, "aiv_intra_core")
        self.assertEqual(multi, "0")

    def test_current_chain_reports_honestly(self):
        # host-assisted G1 chain: internal GM boundary, not an on-chip subgraph
        struct = self.structs["default_8192"]
        self.assertEqual(struct["groups"], "2")
        self.assertEqual(struct["launches"], "2")
        self.assertEqual(struct["gm_boundaries"], "1")
        self.assertEqual(struct["host_assisted"], "1")
        self.assertEqual(struct["on_chip"], "0")

    def test_batch_is_data_dimension_not_architectural(self):
        self.assertEqual(self.structs["default_8192"],
                         self.structs["batch_4096_independent"],
                         "batch must not change the lowering structure")

    def test_fft_check_queries_before_allocation_and_launch(self):
        src = (ROOT / "src" / "host" / "fft_check.cpp").read_text()
        self.assertIn('#include "butterfly/descriptors.hpp"', src)
        query = src.index("query_lowering(")
        self.assertLess(query, src.index("aclrtMalloc"),
                        "lowering must be queried before device allocation")
        self.assertLess(query, src.index("issuePass(d"),
                        "lowering must be queried before any launch")


if __name__ == "__main__":
    unittest.main()
