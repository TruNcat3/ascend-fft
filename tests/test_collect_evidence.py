"""PR #2 stage 4: hard acceptance rules for the long-FFT evidence collector.

Hardware-free: exercises the collector's pure verify_* / parse / manifest
helpers with synthetic outputs so the acceptance vocabulary (rc==0, exactly
three ordered seq entries, no STALE, finite errors under threshold, complete
unique grid, boundary counters, 5 raw trials, clean-tree manifest) cannot
silently regress.
"""

import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_long_fft_evidence as coll  # noqa: E402

SEQ_LINE = ("seq: 3 inputs re-executed under one plan (no rebuild), "
            "stale-check on -> PASS")


HOST_SCOPES = {"plan_setup": 1.0, "first_use": 2.0, "e2e_mean": 100.0,
               "e2e_min": 95.0, "h2d": 10.0, "device_chain": 40.0,
               "d2h": 11.0, "reps": 5}
# six device segments telescoping into device_chain=40.0 (same launch)
DEVICE_SEGMENTS = {"transpose_in": 10.0, "fft1": 8.0, "twiddle": 6.0,
                   "transpose_boundary": 7.0, "fft2": 6.0,
                   "transpose_out": 3.0}
# PR-B R1 fused chain: five segments (twiddle 6.0 + boundary 7.0 merged into
# transpose_boundary=13.0), same device_chain=40.0
FUSED_SEGMENTS = {"transpose_in": 10.0, "fft1": 8.0,
                  "transpose_boundary": 13.0, "fft2": 6.0,
                  "transpose_out": 3.0}


def good_point():
    return {
        "n": 8192, "b": 1, "rc": 0, "max_rel": 1e-7, "max_abs": 3e-6,
        "pass": True,
        "seq": [{"input": s, "max_rel": 1e-7, "pass": True, "stale": False}
                for s in coll.EXPECTED_SEQ],
        "seq_summary": SEQ_LINE,
        "e2e_transfers": "E2E transfers: in=1 out=1 boundary=2 per_execution",
        "boundary_ok": True,
        "trials": {
            "count": 5,
            "raw": [{"e2e_us": v, "e2e_min_us": v, "rc": 0,
                     "max_rel": 1e-7, "max_abs": 3e-6, "pass": True,
                     "scopes": dict(HOST_SCOPES), "segments": None}
                    for v in (100.0, 101.0, 99.0, 102.0, 98.0)],
            "stats": coll.compute_stats([100.0, 101.0, 99.0, 102.0, 98.0]),
        },
    }


def to_device_trials(p):
    """Host-shaped trial samples -> realistic device-boundary samples."""
    for s in p["trials"]["raw"]:
        s["segments"] = dict(DEVICE_SEGMENTS)
        s["boundary_impl"] = "separate"
    return p


def to_fused_trials(p):
    """-> device-boundary samples under AB_LONG_BOUNDARY_IMPL=fused."""
    p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                         "per_execution"
    p["boundary_ok"] = True
    for s in p["trials"]["raw"]:
        s["segments"] = dict(FUSED_SEGMENTS)
        s["boundary_impl"] = "fused"
    return p


def problems_of(mutate, expect_boundary="boundary=2", trials=True):
    p = good_point()
    mutate(p)
    return coll.verify_point(p, expect_boundary, trials=trials)


class VerifyPointTest(unittest.TestCase):
    def test_good_point_accepted(self):
        self.assertEqual(
            coll.verify_point(good_point(), "boundary=2"), [])

    def test_rc_nonzero_rejected(self):
        got = problems_of(lambda p: p.update(rc=1))
        self.assertTrue(any("rc=1" in x for x in got))

    def test_missing_pass_rejected(self):
        got = problems_of(lambda p: p.update({"pass": False}))
        self.assertTrue(any("PASS line missing" in x for x in got))

    def test_max_rel_over_threshold_rejected(self):
        got = problems_of(lambda p: p.update(max_rel=2e-4))
        self.assertTrue(any("max_rel 0.0002" in x for x in got))

    def test_max_rel_nonfinite_rejected(self):
        got = problems_of(lambda p: p.update(max_rel=float("nan")))
        self.assertTrue(any("not finite" in x for x in got))

    def test_seq_length_enforced(self):
        got = problems_of(lambda p: p["seq"].pop())
        self.assertTrue(any("exactly 3" in x for x in got))

    def test_seq_order_enforced(self):
        def swap(p):
            p["seq"][0], p["seq"][1] = p["seq"][1], p["seq"][0]
        got = problems_of(swap)
        self.assertTrue(any("seq order" in x for x in got))

    def test_stale_output_rejected(self):
        def stale(p):
            p["seq"][1]["stale"] = True
        got = problems_of(stale)
        self.assertTrue(any("STALE-OUTPUT" in x for x in got))

    def test_seq_error_over_threshold_rejected(self):
        def bad(p):
            p["seq"][2]["max_rel"] = 5e-4
        got = problems_of(bad)
        self.assertTrue(any("seq[2]" in x and "0.0005" in x for x in got))

    def test_missing_seq_summary_rejected(self):
        got = problems_of(lambda p: p.update(seq_summary=""))
        self.assertTrue(any("summary line missing" in x for x in got))

    def test_failed_seq_summary_rejected(self):
        got = problems_of(
            lambda p: p.update(seq_summary=SEQ_LINE.replace("PASS", "FAIL")))
        self.assertTrue(any("not successful" in x for x in got))

    def test_boundary_counter_mismatch_rejected(self):
        got = problems_of(lambda p: p.update(
            e2e_transfers="E2E transfers: in=1 out=1 boundary=0 per_execution",
            boundary_ok=False))
        self.assertTrue(any("want boundary=2" in x for x in got))
        self.assertTrue(any("boundary_ok" in x for x in got))

    def test_device_mode_accepts_boundary_zero(self):
        p = to_device_trials(good_point())
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
        self.assertEqual(coll.verify_point(p, "boundary=0"), [])

    def test_device_missing_segments_rejected(self):
        p = good_point()
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
        got = coll.verify_point(p, "boundary=0")
        self.assertTrue(any("missing six segments" in x for x in got))
        self.assertEqual(len([x for x in got if "missing six" in x]), 5)

    def test_device_non_telescoping_segments_rejected(self):
        p = to_device_trials(good_point())
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
        p["trials"]["raw"][2]["segments"]["twiddle"] = 999.0
        got = coll.verify_point(p, "boundary=0")
        self.assertTrue(any("trial[2]" in x and "sum(segments)" in x
                            for x in got))

    def test_host_trial_with_segments_rejected(self):
        p = good_point()
        p["trials"]["raw"][0]["segments"] = dict(DEVICE_SEGMENTS)
        got = coll.verify_point(p, "boundary=2")
        self.assertTrue(any("trial[0]" in x and "segments=NA" in x
                            for x in got))

    # ---- PR-B impl contract (5-seg fused vs 6-seg separate) ----
    def test_fused_point_accepted(self):
        p = to_fused_trials(good_point())
        self.assertEqual(
            coll.verify_point(p, "boundary=0", impl="fused"), [])

    def test_fused_rejects_twiddle_segment(self):
        p = to_fused_trials(good_point())
        p["trials"]["raw"][0]["segments"]["twiddle"] = 6.0
        got = coll.verify_point(p, "boundary=0", impl="fused")
        self.assertTrue(any("twiddle" in x and "fused" in x for x in got))

    def test_fused_missing_five_segments(self):
        p = to_fused_trials(good_point())
        p["trials"]["raw"][0]["segments"] = None
        got = coll.verify_point(p, "boundary=0", impl="fused")
        self.assertTrue(any("missing five segments" in x for x in got))

    def test_separate_contract_rejects_five_segments(self):
        p = to_fused_trials(good_point())
        got = coll.verify_point(p, "boundary=0", impl="separate")
        self.assertTrue(any("segment twiddle missing" in x for x in got))

    def test_boundary_impl_mismatch_rejected(self):
        p = to_fused_trials(good_point())
        got = coll.verify_point(p, "boundary=0", impl="separate")
        self.assertTrue(any("boundary_impl 'fused' != 'separate'" in x
                            for x in got))
        p = to_device_trials(good_point())
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
        got = coll.verify_point(p, "boundary=0", impl="fused")
        self.assertTrue(any("boundary_impl 'separate' != 'fused'" in x
                            for x in got))

    def test_host_trial_with_boundary_impl_rejected(self):
        p = good_point()
        p["trials"]["raw"][0]["boundary_impl"] = "separate"
        got = coll.verify_point(p, "boundary=2")
        self.assertTrue(any("must not report boundary_impl" in x
                            for x in got))

    def test_legacy_trial_without_boundary_impl_key_accepted(self):
        # pre-PR-B archives carry no boundary_impl key at all: attest only
        # what the contract had at the time.
        p = to_device_trials(good_point())
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
        for s in p["trials"]["raw"]:
            del s["boundary_impl"]
        self.assertEqual(coll.verify_point(p, "boundary=0"), [])

    def test_trials_count_enforced(self):
        def shrink(p):
            p["trials"]["count"] = 4
            p["trials"]["stats"]["samples"] = p["trials"]["stats"]["samples"][:4]
        got = problems_of(shrink)
        self.assertTrue(any("trials count 4" in x for x in got))

    def test_trials_raw_sample_invalid_rejected(self):
        def poison(p):
            p["trials"]["stats"]["samples"][2] = float("nan")
        got = problems_of(poison)
        self.assertTrue(any("trial[2]" in x for x in got))

    def test_missing_transfers_rejected(self):
        got = problems_of(lambda p: p.update(e2e_transfers=""))
        self.assertTrue(any("transfers line missing" in x for x in got))


class ControlAndE2eTest(unittest.TestCase):
    def test_control_accepted_without_boundary(self):
        ctrl = good_point()
        del ctrl["trials"]
        del ctrl["boundary_ok"]
        self.assertEqual(coll.verify_control(ctrl), [])

    def test_control_still_requires_seq_contract(self):
        ctrl = good_point()
        ctrl["seq"] = ctrl["seq"][:2]
        del ctrl["trials"]
        self.assertTrue(coll.verify_control(ctrl))

    def test_e2e_good(self):
        e2e = {"rc": 0, "transfers": "E2E transfers: in=1 out=1 boundary=2 "
                                      "per_execution",
               "line": "E2E n=8192 batch=1 reps=3 ..."}
        self.assertEqual(coll.verify_e2e(e2e, "boundary=2"), [])

    def test_e2e_wrong_boundary(self):
        e2e = {"rc": 0, "transfers": "E2E transfers: in=1 out=1 boundary=0 "
                                      "per_execution",
               "line": "E2E n=8192 batch=1 reps=3 ..."}
        got = coll.verify_e2e(e2e, "boundary=2")
        self.assertTrue(any("want boundary=2" in x for x in got))

    def test_e2e_rc_failure(self):
        got = coll.verify_e2e({"rc": 7, "transfers": "", "line": ""},
                              "boundary=0")
        self.assertTrue(any("rc=7" in x for x in got))


class GridTest(unittest.TestCase):
    def points(self):
        return [{"n": n, "b": b} for n in coll.NS for b in coll.BS]

    def test_full_grid_accepted(self):
        self.assertEqual(coll.verify_grid(self.points()), [])

    def test_missing_shape_rejected(self):
        pts = self.points()[:-1]
        got = coll.verify_grid(pts)
        self.assertTrue(any("missing" in x for x in got))

    def test_duplicate_shape_rejected(self):
        pts = self.points()
        pts.append(dict(pts[0]))
        got = coll.verify_grid(pts)
        self.assertTrue(any("duplicate" in x for x in got))


class StatsAndParseTest(unittest.TestCase):
    def test_compute_stats(self):
        s = coll.compute_stats([100.0, 101.0, 99.0, 102.0, 98.0])
        self.assertEqual(s["min"], 98.0)
        self.assertEqual(s["median"], 100.0)
        self.assertEqual(s["mean"], 100.0)
        self.assertAlmostEqual(s["cv"], math.sqrt(2.0) / 100.0, places=6)
        self.assertEqual(len(s["samples"]), 5)

    def test_compute_stats_drops_nonfinite(self):
        s = coll.compute_stats([100.0, float("nan")])
        self.assertEqual(len(s["samples"]), 2)
        self.assertTrue(math.isnan(s["samples"][1]))
        self.assertEqual(s["mean"], 100.0)

    def test_parse_point_captures_stale(self):
        out = ("seq[0]=/a.bin maxRel=1.0e-07 PASS\n"
               "seq[1]=/b.bin maxRel=2.0e-07 PASS STALE-OUTPUT\n"
               "seq[2]=/a.bin maxRel=1.0e-07 PASS\n" + SEQ_LINE + "\n"
               "E2E transfers: in=1 out=1 boundary=2 per_execution\n"
               "n=8192 batch=1 blocks=1  maxAbs=1.0e-06 maxRel=1.0e-07 "
               "(worst idx 3)\nPASS\n")
        p = coll.parse_point(out)
        self.assertEqual(len(p["seq"]), 3)
        self.assertFalse(p["seq"][0]["stale"])
        self.assertTrue(p["seq"][1]["stale"])
        self.assertTrue(p["pass"])
        self.assertIn("boundary=2", p["e2e_transfers"])
        got = coll.verify_point(p, "boundary=2", trials=False)
        self.assertTrue(any("STALE-OUTPUT" in x for x in got))

    def test_parse_trial_scopes(self):
        out = ("E2E n=8192 batch=1 reps=5 e2e_us=100.0 e2e_min_us=95.0 "
               "first_us=1.0 boot_us=2.0 plan_us=3.0 warmup_us=4.0 "
               "mode=async host=chain input=random-seeded\n"
               "scopes: plan_setup=1.0 us first_use=2.0 us "
               "host_end_to_end mean=100.0 min=95.0 us h2d=10.0 us "
               "device_chain=40.0 us d2h=11.0 us reps=5\n"
               "segments: transpose_in=10.0 us fft1=8.0 us twiddle=6.0 us "
               "transpose_boundary=7.0 us fft2=6.0 us transpose_out=3.0 us\n")
        s = coll.parse_trial(out)
        self.assertEqual(s["e2e_us"], 100.0)
        self.assertEqual(s["scopes"]["device_chain"], 40.0)
        self.assertAlmostEqual(sum(s["segments"].values()), 40.0)

    def test_parse_trial_segments_na(self):
        out = ("scopes: plan_setup=1.0 us first_use=2.0 us "
               "host_end_to_end mean=100.0 min=95.0 us h2d=10.0 us "
               "device_chain=40.0 us d2h=11.0 us reps=5\n"
               "segments: NA\n")
        s = coll.parse_trial(out)
        self.assertIsNone(s["segments"])
        self.assertIsNone(s["boundary_impl"])

    def test_parse_trial_boundary_impl(self):
        base = ("scopes: plan_setup=1.0 us first_use=2.0 us "
                "host_end_to_end mean=100.0 min=95.0 us h2d=10.0 us "
                "device_chain=40.0 us d2h=11.0 us reps=5\n")
        s = coll.parse_trial(base + "boundary_impl: fused\n"
                             "segments: transpose_in=10.0 us fft1=8.0 us "
                             "transpose_boundary=13.0 us fft2=6.0 us "
                             "transpose_out=3.0 us\n")
        self.assertEqual(s["boundary_impl"], "fused")
        self.assertAlmostEqual(sum(s["segments"].values()), 40.0)
        s = coll.parse_trial(base + "boundary_impl: separate\n")
        self.assertEqual(s["boundary_impl"], "separate")

    def test_parse_trial_malformed_boundary_impl_rejected(self):
        with self.assertRaises(ValueError):
            coll.parse_trial("boundary_impl: maybe\n")


class ModeContractTest(unittest.TestCase):
    """PR-B mode table: device-fused maps to the same boundary=0 chain with
    the fused impl and its own archive directory."""

    def test_mode_tables(self):
        self.assertEqual(coll.BOUNDARY_BY_MODE["device-fused"], "boundary=0")
        self.assertEqual(coll.OUT_BY_MODE["device-fused"],
                         "long-fft-device-boundary-fused")
        self.assertEqual(coll.OUT_BY_MODE["device"],
                         "long-fft-device-boundary")
        self.assertNotEqual(coll.OUT_BY_MODE["device-fused"],
                            coll.OUT_BY_MODE["device"])
        self.assertEqual(coll.IMPL_BY_MODE,
                         {"host": "separate", "device": "separate",
                          "device-fused": "fused"})

    def test_env_tags(self):
        self.assertEqual(coll.env_tags("host"),
                         ["AB_LONG_BOUNDARY_IMPL=separate"])
        self.assertEqual(coll.env_tags("device"),
                         ["AB_BOUNDARY=device",
                          "AB_LONG_BOUNDARY_IMPL=separate"])
        self.assertEqual(coll.env_tags("device-fused"),
                         ["AB_BOUNDARY=device",
                          "AB_LONG_BOUNDARY_IMPL=fused"])

    def test_mode_env_pins_impl_against_ambient(self):
        old = os.environ.get("AB_LONG_BOUNDARY_IMPL")
        os.environ["AB_LONG_BOUNDARY_IMPL"] = "fused"
        try:
            self.assertEqual(
                coll.mode_env("host")["AB_LONG_BOUNDARY_IMPL"], "separate")
            self.assertEqual(
                coll.mode_env("device")["AB_LONG_BOUNDARY_IMPL"], "separate")
            self.assertEqual(
                coll.mode_env("device")["AB_BOUNDARY"], "device")
            self.assertEqual(
                coll.mode_env("device-fused")["AB_LONG_BOUNDARY_IMPL"],
                "fused")
            self.assertEqual(
                coll.mode_env("device-fused")["AB_BOUNDARY"], "device")
            self.assertNotIn("AB_BOUNDARY", coll.mode_env("host"))
        finally:
            if old is None:
                os.environ.pop("AB_LONG_BOUNDARY_IMPL", None)
            else:
                os.environ["AB_LONG_BOUNDARY_IMPL"] = old

    def test_control_env_pins_default_impl(self):
        # the short control never runs the long chain, and fft_check
        # rejects fused+short; device-fused must still get a runnable
        # control with the canonical default pinned.
        for mode in ("host", "device", "device-fused"):
            env = coll.control_env(mode)
            self.assertEqual(env["AB_LONG_BOUNDARY_IMPL"], "separate", mode)
            self.assertEqual(env["AB_E2E"], "1", mode)
            self.assertEqual(env["AB_INPUT_SEQ"],
                             "impulse,random-seeded,impulse", mode)
        self.assertNotIn("AB_BOUNDARY", coll.control_env("host"))


class ManifestTest(unittest.TestCase):
    def test_git_state(self):
        sha, dirty = coll.git_state()
        self.assertRegex(sha, r"^[0-9a-f]{40}$")
        self.assertIsInstance(dirty, bool)

    def test_patch_digest_shape(self):
        self.assertRegex(coll.patch_digest(), r"^[0-9a-f]{64}$")

    def test_manifest_binary_hash(self):
        binary = ROOT / "build" / "fft_check"
        if not binary.is_file():
            self.skipTest("build/fft_check not built in this environment")
        man = coll.build_manifest("host", allow_dirty=False)
        self.assertRegex(man["binary_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(man["binary"], "build/fft_check")
        self.assertIn("soc", man)
        self.assertIn("cann_version", man)
        self.assertIn("driver_version", man)
        self.assertIn("env", man)
        self.assertCountEqual(
            man["kernel_objects"],
            ["fft_radix2.o", "fft_long.o", "fft_real.o"])
        for v in man["kernel_objects"].values():
            self.assertRegex(v, r"^[0-9a-f]{64}$")
        self.assertTrue(man["config_sha256"])
        for v in man["config_sha256"].values():
            self.assertRegex(v, r"^[0-9a-f]{64}$")
        self.assertIn("driver_source", man)
        if man["driver_source"] is None:
            self.assertEqual(man["driver_version"], "unknown")
        self.assertEqual(coll.manifest_incomplete(man), [])


class ErrorInjectionTest(unittest.TestCase):
    """R0.2: injected corruptions must each be rejected explicitly."""

    def test_empty_raw_rejected(self):
        def empty(p):
            p["trials"]["raw"] = []
            p["trials"]["count"] = 0
            p["trials"]["stats"] = coll.compute_stats([])
        got = problems_of(empty)
        self.assertTrue(any("trials.raw has 0" in x for x in got))

    def test_trial_rc_nonzero_rejected(self):
        got = problems_of(lambda p: p["trials"]["raw"][1].update(rc=1))
        self.assertTrue(any("trial[1] rc=1" in x for x in got))

    def test_tampered_stats_median_rejected(self):
        def tamper(p):
            p["trials"]["stats"]["median"] = 1.0
        got = problems_of(tamper)
        self.assertTrue(any("stats median" in x and "!= recomputed" in x
                            for x in got))

    def test_tampered_stats_samples_rejected(self):
        def tamper(p):
            p["trials"]["stats"]["samples"][0] = 123.0
        got = problems_of(tamper)
        self.assertTrue(any("samples != raw" in x for x in got))

    def test_trial_missing_pass_rejected(self):
        got = problems_of(
            lambda p: p["trials"]["raw"][0].update({"pass": False}))
        self.assertTrue(any("trial[0] PASS line missing" in x for x in got))

    def test_trial_max_rel_over_threshold_rejected(self):
        got = problems_of(lambda p: p["trials"]["raw"][2].update(max_rel=2e-4))
        self.assertTrue(any("trial[2] max_rel 0.0002" in x for x in got))

    def test_trial_scopes_missing_rejected(self):
        got = problems_of(lambda p: p["trials"]["raw"][0].update(scopes=None))
        self.assertTrue(any("trial[0] scopes missing" in x for x in got))

    def test_kernel_object_change_changes_manifest(self):
        if not (ROOT / "build" / "fft_check").is_file():
            self.skipTest("build/fft_check not built in this environment")
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.o"
            b = Path(td) / "b.o"
            a.write_bytes(b"kernel-a")
            b.write_bytes(b"kernel-b")
            digests = []
            for obj in (a, b):
                os.environ["AB_FFT_O"] = str(obj)
                try:
                    man = coll.build_manifest("host", allow_dirty=False)
                finally:
                    os.environ.pop("AB_FFT_O", None)
                digests.append(man["kernel_objects"]["fft_radix2.o"])
            self.assertNotEqual(digests[0], digests[1])
            self.assertEqual(digests[1],
                             hashlib.sha256(b"kernel-b").hexdigest())


class DriverDetectionTest(unittest.TestCase):
    def test_driver_info_env_override(self):
        os.environ["AB_DRIVER_VERSION"] = "9.9.9-test"
        try:
            self.assertEqual(
                coll.driver_info(),
                ("9.9.9-test", "env:AB_DRIVER_VERSION"))
            self.assertEqual(coll.detect_driver(), "9.9.9-test")
        finally:
            os.environ.pop("AB_DRIVER_VERSION", None)

    def test_unresolved_driver_is_incomplete_not_unknown(self):
        man = {"driver_source": None, "driver_version": "unknown",
               "kernel_objects": {"fft_radix2.o": "0" * 64,
                                  "fft_long.o": "0" * 64,
                                  "fft_real.o": "0" * 64},
               "config_sha256": {"x.json": "0" * 64}}
        got = coll.manifest_incomplete(man)
        self.assertTrue(any("driver version unrecognized" in x for x in got))

    def test_missing_hash_is_incomplete(self):
        man = {"driver_source": "npu-smi", "driver_version": "1",
               "kernel_objects": {"fft_radix2.o": "0" * 64},
               "config_sha256": {}}
        got = coll.manifest_incomplete(man)
        self.assertTrue(any("fft_long.o" in x for x in got))
        self.assertTrue(any("config_sha256 empty" in x for x in got))


class VerifyDocumentTest(unittest.TestCase):
    ARCHIVES = [ROOT / "results" / "evidence" / d / "acceptance.json"
                for d in ("long-fft-acceptance", "long-fft-device-boundary")]
    FUSED = (ROOT / "results" / "evidence" /
             "long-fft-device-boundary-fused" / "acceptance.json")
    HAVE = all(p.is_file() for p in ARCHIVES)

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_committed_archives_accepted(self):
        for path in self.ARCHIVES + ([self.FUSED]
                                     if self.FUSED.is_file() else []):
            self.assertEqual(coll.verify_document(path), [], str(path))

    @classmethod
    def _load(cls, i=0):
        return json.loads(cls.ARCHIVES[i].read_text(encoding="utf-8"))

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_tampered_stats_rejected(self):
        doc = self._load()
        doc["points"][0]["trials"]["stats"]["median"] = 1.0
        got = coll.verify_document(doc)
        self.assertTrue(any("stats median" in x for x in got))

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_tampered_raw_rc_rejected(self):
        doc = self._load()
        doc["points"][0]["trials"]["raw"][0]["rc"] = 1
        got = coll.verify_document(doc)
        self.assertTrue(any("trial[0] rc=1" in x for x in got))

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_missing_kernel_hash_rejected(self):
        doc = self._load()
        doc["manifest"].pop("kernel_objects", None)
        got = coll.verify_document(doc)
        self.assertTrue(any("kernel_objects" in x for x in got))

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_missing_top_level_key_rejected(self):
        doc = self._load()
        doc.pop("points")
        got = coll.verify_document(doc)
        self.assertTrue(any("missing top-level key 'points'" in x
                            for x in got))

    @unittest.skipUnless(HAVE, "acceptance archives not present")
    def test_impl_mismatch_rejected(self):
        # device archive claims fused impl: boundary/impl cross-check fails
        doc = self._load(1)
        doc["impl"] = "fused"
        got = coll.verify_document(doc)
        self.assertTrue(any("impl 'fused' != 'separate'" in x for x in got))

    @unittest.skipUnless(HAVE and FUSED.is_file(),
                         "fused acceptance archive not present")
    def test_fused_archive_attested(self):
        self.assertEqual(coll.verify_document(self.FUSED), [],
                         str(self.FUSED))
        doc = json.loads(self.FUSED.read_text(encoding="utf-8"))
        self.assertEqual(doc["boundary"], "device-fused")
        self.assertEqual(doc["impl"], "fused")
        for p in doc["points"]:
            for s in p["trials"]["raw"]:
                self.assertEqual(s.get("boundary_impl"), "fused")
                self.assertNotIn("twiddle", s["segments"])


if __name__ == "__main__":
    unittest.main()
