"""PR #2 stage 4: hard acceptance rules for the long-FFT evidence collector.

Hardware-free: exercises the collector's pure verify_* / parse / manifest
helpers with synthetic outputs so the acceptance vocabulary (rc==0, exactly
three ordered seq entries, no STALE, finite errors under threshold, complete
unique grid, boundary counters, 5 raw trials, clean-tree manifest) cannot
silently regress.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_long_fft_evidence as coll  # noqa: E402

SEQ_LINE = ("seq: 3 inputs re-executed under one plan (no rebuild), "
            "stale-check on -> PASS")


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
            "raw": [{"e2e_us": v, "e2e_min_us": v} for v in
                    (100.0, 101.0, 99.0, 102.0, 98.0)],
            "stats": coll.compute_stats([100.0, 101.0, 99.0, 102.0, 98.0]),
        },
    }


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
        p = good_point()
        p["e2e_transfers"] = "E2E transfers: in=1 out=1 boundary=0 " \
                             "per_execution"
        p["boundary_ok"] = True
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
               "device_chain=40.0 us d2h=11.0 us reps=5\n")
        s = coll.parse_trial(out)
        self.assertEqual(s["e2e_us"], 100.0)
        self.assertEqual(s["scopes"]["device_chain"], 40.0)


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


if __name__ == "__main__":
    unittest.main()
