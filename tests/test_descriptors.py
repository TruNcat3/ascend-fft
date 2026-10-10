"""Architecture descriptor legality tests (addendum step 2, PR #2 stage 2).

Hardware-independent: compiles tests/test_descriptors.cpp with plain g++ (no
CANN), runs its machine-readable case table, and checks
  - every unsupported tuple carries an explicit reason,
  - abstract feasibility and executable lowering are separate verdicts
    (unimplemented Us/Ts/Ud/Td/pipeline_buffers/role/residence combos come
    back as `abstract-feasible-but-not-lowered`, never as supported),
  - the launch manifest matches the runtime: 6 device-boundary launches in
    transpose_in|row_fft|twiddle|transpose_boundary|row_fft|transpose_out
    order (PR-B: 5 launches with boundary_impl=Fused, twiddle merged into
    transpose_boundary, boundary_edge unchanged), 2 host-boundary row_fft
    launches, 1 materialized GM boundary each, and modeled payload GM bytes =
    launches x 2 x n x batch x 8 (payload model only: no twiddle/index/coeff
    traffic, not a profiler measurement),
  - the builtin H profile stays in sync with config/ascend910_93_profile.json,
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

NOT_LOWERED = "abstract-feasible-but-not-lowered"

EXPECTED_CASES = {
    "default_8192": (True, None),
    "default_65536_b3": (True, None),
    "batch_4096_independent": (True, None),
    "default_device_gm": (True, None),
    "fused_device_gm": (True, None),
    "cell_resident_not_lowered": (False, NOT_LOWERED),
    "us2_not_lowered": (False, NOT_LOWERED),
    "td99_not_lowered": (False, NOT_LOWERED),
    "pipeline99_not_lowered": (False, NOT_LOWERED),
    "ub_below_transpose_peak": (False, "UB overflow"),
    "ub_host_ok_at_80000": (True, None),
    "ub_exact_transpose_peak": (True, None),
    "ub_rowfft_checked_after_transpose": (False, "UB overflow"),
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
    "fold_d_override_rejected": (False, "UB overflow"),
    "plane_k_32_len256_accepted": (True, None),
    "plane_k_12_rejected": (False, "illegal row-FFT plan"),
    "plane_k_24_rejected": (False, "illegal row-FFT plan"),
    "plane_k_32_len64_rejected": (False, "illegal row-FFT plan"),
    # R2-A transpose tile candidates (AB_LT_TILE=HxW)
    "lt_tile_64x64_supported": (True, None),
    "lt_tile_256x16_supported": (True, None),
    "lt_tile_64x32_rejected": (False, "illegal transpose tile"),
    "lt_tile_64x16_rejected": (False, "illegal transpose tile"),
    "lt_tile_garbage_rejected": (False, "illegal transpose tile"),
    "lt_tile_128x64_rejected": (False, "UB overflow"),
    # R2-A Round 2 stripe K candidates (AB_LT_STRIPE_K)
    "lt_stripe_k_256_supported": (True, None),
    "lt_stripe_k_12_rejected": (False, "illegal transpose stripe"),
    "lt_tile_64x32_k256_supported": (True, None),
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

    def test_host_chain_manifest(self):
        # host-assisted G1 chain: two row_fft launches, one host GM boundary
        struct = self.structs["default_8192"]
        self.assertEqual(struct["launches"], "2")
        self.assertEqual(struct["kinds"], "row_fft|row_fft")
        self.assertEqual(struct["gm_boundaries"], "1")
        self.assertEqual(struct["host_assisted"], "1")
        self.assertEqual(struct["on_chip"], "0")
        self.assertEqual(struct["abstract"], "1")

    def test_device_chain_manifest_is_six_launches(self):
        # addendum §3 device chain: the full 6-launch list, no host hop
        struct = self.structs["default_device_gm"]
        self.assertEqual(struct["launches"], "6")
        self.assertEqual(struct["kinds"],
                         "transpose_in|row_fft|twiddle|transpose_boundary|"
                         "row_fft|transpose_out")
        self.assertEqual(struct["gm_boundaries"], "1")
        self.assertEqual(struct["host_assisted"], "0")
        self.assertEqual(struct["on_chip"], "0")
        self.assertEqual(struct["abstract"], "1")
        # Modeled payload GM accounting (PR-B, R1.0 rename): 6 launches x
        # 2 x n x batch x 8 bytes, 8192x1 tensor -> 6 * 2 * 65536 = 786432.
        self.assertEqual(struct["modeled_payload_gm"], "786432")

    def test_fused_device_chain_manifest_is_five_launches(self):
        # PR-B R1: boundary_impl=Fused drops the twiddle record; the merged
        # boundary transpose keeps the single GM edge, UB peak unchanged
        # (bTw statically reserved in kfft_lt_tr), GM bytes lose one
        # launch's full read+write.
        struct = self.structs["fused_device_gm"]
        self.assertEqual(struct["launches"], "5")
        self.assertEqual(struct["kinds"],
                         "transpose_in|row_fft|transpose_boundary|"
                         "row_fft|transpose_out")
        self.assertEqual(struct["gm_boundaries"], "1")
        self.assertEqual(struct["host_assisted"], "0")
        self.assertEqual(struct["on_chip"], "0")
        self.assertEqual(struct["abstract"], "1")
        self.assertEqual(struct["ub_peak"], "131072")
        self.assertEqual(struct["modeled_payload_gm"], "655360")
        self.assertEqual(int(struct["modeled_payload_gm"]) + 2 * 8192 * 8,
                         int(self.structs["default_device_gm"]["modeled_payload_gm"]))

    def test_plan_ub_is_serial_peak_not_sum(self):
        # device peak = transpose 3*128*32*8 = 98304 plus the statically
        # reserved twiddle tile 128*32*8 = 32768 -> AB_FUSED_UB_BYTES
        # 131072 (PR-B; was 98304 before bTw). Host peak = row FFT
        # AB_ROW_FFT_UB_BYTES with the launch's rows-aware D:
        # len128/rows64 -> D=1,K=8 -> 6208 (R0.1, was 6080 under the stale
        # 46.5n+128 macro).
        self.assertEqual(self.structs["default_device_gm"]["ub_peak"],
                         "131072")
        self.assertEqual(self.structs["default_host_memory"]["ub_peak"], "6208")
        self.assertEqual(self.structs["ub_exact_transpose_peak"]["ub_peak"],
                         "131072")

    def test_ub_below_transpose_peak_rejects_device_not_host(self):
        # the discriminating boundary: same 80000 B UB, host chain fits,
        # device chain must not (kfft_lt_tr peak 131072)
        status, reason = self.cases["ub_below_transpose_peak"]
        self.assertEqual(status, "UNSUPPORTED")
        self.assertIn("UB overflow", reason)
        self.assertIn("kfft_lt_tr", reason)
        self.assertIn("131072", reason)
        self.assertEqual(self.cases["ub_host_ok_at_80000"][0], "SUPPORTED")

    def test_rowfft_checked_after_transpose_threshold(self):
        # ub == 131072: transpose passes, then kfft_fwd (191616 B) rejects
        status, reason = self.cases["ub_rowfft_checked_after_transpose"]
        self.assertEqual(status, "UNSUPPORTED")
        self.assertIn("UB overflow", reason)
        self.assertIn("kfft_fwd", reason)

    def test_not_lowered_is_abstract_not_executable(self):
        status, reason = self.cases["cell_resident_not_lowered"]
        self.assertEqual(status, "UNSUPPORTED")
        self.assertTrue(reason.startswith(NOT_LOWERED))
        struct = self.structs["cell_resident_not_lowered"]
        self.assertEqual(struct["abstract"], "1",
                         "residency must pass level-1 abstract feasibility")
        self.assertEqual(struct["launches"], "0",
                         "an unlowered tuple must not report launches")

    def test_batch_is_data_dimension_not_architectural(self):
        # Since R0.1 the row-FFT resource is rows-aware (fold D depends on the
        # launch's row count), so ub_peak may grow with batch: 6208 (b=1) vs
        # 17536 (b=4096, D=4), and modeled payload GM scales with the tensor.
        # The ARCHITECTURE -- launches, kinds, GM boundaries, feasibility --
        # must stay batch-independent.
        a = dict(self.structs["default_8192"])
        b = dict(self.structs["batch_4096_independent"])
        a.pop("ub_peak")
        b.pop("ub_peak")
        gm_a, gm_b = a.pop("modeled_payload_gm"), b.pop("modeled_payload_gm")
        self.assertEqual(a, b,
                         "batch must not change the lowering structure")
        self.assertEqual(self.structs["default_8192"]["ub_peak"], "6208")
        self.assertEqual(self.structs["batch_4096_independent"]["ub_peak"],
                         "17536")
        self.assertGreater(
            int(self.structs["batch_4096_independent"]["ub_peak"]),
            int(self.structs["default_8192"]["ub_peak"]),
            "larger batch -> larger fold D -> larger exact UB (documented)")
        self.assertEqual(int(gm_b), 4096 * int(gm_a),
                         "modeled payload GM bytes must scale linearly")

    def test_fft_check_queries_before_allocation_and_launch(self):
        src = (ROOT / "src" / "host" / "fft_check.cpp").read_text()
        self.assertIn('#include "butterfly/descriptors.hpp"', src)
        query = src.index("query_lowering(")
        self.assertLess(query, src.index("aclrtMalloc"),
                        "lowering must be queried before device allocation")
        self.assertLess(query, src.index("issuePass(d"),
                        "lowering must be queried before any launch")

    def test_lt_tile_candidates_reshape_ub_peak(self):
        # R2-A: AB_LT_TILE selects a compiled entry; the descriptor gates the
        # exact four-tile peak of the selected shape (single-source macro).
        self.assertEqual(
            self.structs["lt_tile_64x64_supported_struct"]["ub_peak"],
            str(4 * 8 * 64 * 64))
        self.assertEqual(
            self.structs["lt_tile_256x16_supported_struct"]["ub_peak"],
            str(4 * 8 * 256 * 16))
        # unsetenv must restore the default 128x32 peak (no env leakage).
        self.assertEqual(self.structs["fused_after_unset"]["ub_peak"],
                         str(4 * 8 * 128 * 32))

    def test_lt_tile_illegal_rejected_loudly(self):
        # K=512 trim: 10K=5120 must fit 2HW, so 64x32 (2HW=4096) and 64x16
        # (2HW=2048) are stripe-illegal -- loud rejections, no silent reshape.
        for name in ("lt_tile_64x32_rejected", "lt_tile_64x16_rejected"):
            status, reason = self.cases[name]
            self.assertEqual(status, "UNSUPPORTED")
            self.assertIn("illegal transpose tile", reason)
            self.assertIn("10K", reason)
        status, reason = self.cases["lt_tile_garbage_rejected"]
        self.assertEqual(status, "UNSUPPORTED")
        self.assertIn("illegal transpose tile", reason)
        # 128x64 is stripe-legal but 4*8*128*64 = 262144 > the 192 KiB budget
        status, reason = self.cases["lt_tile_128x64_rejected"]
        self.assertEqual(status, "UNSUPPORTED")
        self.assertIn("UB overflow", reason)
        self.assertIn("262144", reason)

    def test_lt_kernel_entries_cover_the_legal_ub_fitting_set(self):
        # R2-A single source: every (H,W) that is stripe-legal at the default
        # K=512 AND fits the fused UB budget has a compiled entry, and no
        # other entry exists (the host would fail the symbol lookup).  The
        # kernel expands src/ascendc/fft_long_lt_tr.inc once per
        # (AB_LT_TR_SUFFIX, AB_LT_TR_H, AB_LT_TR_W) triple.
        import re
        src = (ROOT / "src" / "ascendc" / "fft_long.cpp").read_text()
        entries = set()
        for m in re.finditer(
                r"#define AB_LT_TR_SUFFIX\s*(\S*)\s*\n"
                r"#define AB_LT_TR_H (\d+)\s*\n"
                r"#define AB_LT_TR_W (\d+)", src):
            entries.add((int(m.group(2)), int(m.group(3))))

        def legal(h, w, k=512):
            if h not in (64, 128, 256) or w not in (16, 32, 64):
                return False
            if w % 4:
                return False
            if k > h * w or 10 * k > 2 * h * w:
                return False
            return 4 * 8 * h * w <= 196608      # fused UB budget

        expect = {(h, w) for h in (64, 128, 256) for w in (16, 32, 64)
                  if legal(h, w)}
        self.assertEqual(entries, expect,
                         "compiled kfft_lt_tr entries must match the legal "
                         "UB-fitting candidate set at K=512")
        # Round 2: stripe-K variants exist for the DEFAULT tile only.
        k_entries = set()
        for m in re.finditer(
                r"#define AB_LT_TR_SUFFIX\s*(\S+)\s*\n"
                r"#define AB_LT_TR_H (\d+)\s*\n"
                r"#define AB_LT_TR_W (\d+)\s*\n"
                r"#define AB_LT_TR_K (\d+)", src):
            k_entries.add((int(m.group(1).lstrip("_k")) if m.group(1).startswith("_k")
                           else None, int(m.group(2)), int(m.group(3)),
                           int(m.group(4))))
        self.assertEqual(
            {(h, w, k) for _, h, w, k in k_entries if _ is not None},
            {(128, 32, 256), (128, 32, 128)},
            "stripe-K variants must exist for the default 128x32 tile")


if __name__ == "__main__":
    unittest.main()
