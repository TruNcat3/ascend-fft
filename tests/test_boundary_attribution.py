"""PR-B R1: attribution statistics, plan gate and archive contract.

Hardware-free: exercises the pure helpers of
scripts/collect_boundary_attribution.py — paired medians, the 4/5-not-slower
+ >=5% improvement candidate rule, the >3% regression fallback, the
alternating pair order, the descriptor-derived constants (kernel counts, GM
bytes, UB peak) and verify_attribution's tamper rejection on a synthetic
archive.
"""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_boundary_attribution as aff  # noqa: E402

SEP_SEGS = (0.25, 0.20, 0.15, 0.20, 0.10, 0.10)   # sums to 1.0
FUS_SEGS = (0.25, 0.20, 0.35, 0.10, 0.10)         # sums to 1.0


def _segs(fracs, chain):
    names = ("transpose_in", "fft1", "twiddle", "transpose_boundary",
             "fft2", "transpose_out") if len(fracs) == 6 else (
        "transpose_in", "fft1", "transpose_boundary", "fft2",
        "transpose_out")
    return {k: chain * f for k, f in zip(names, fracs)}


def _run(chain, e2e, impl):
    fracs = SEP_SEGS if impl == "separate" else FUS_SEGS
    return {"rc": 0, "pass": True, "boundary_impl": impl,
            "chain": chain, "e2e": e2e, "segments": _segs(fracs, chain),
            "blocks": 48, "max_rel": 1e-7}


def _pairs(delta_pct=-10.0, trials=5):
    pairs = []
    for i in range(trials):
        sep_c = 100.0 + i
        fus_c = sep_c * (1.0 + delta_pct / 100.0)
        sep_e = 1000.0 + i
        fus_e = sep_e * (1.0 + delta_pct / 200.0)
        pairs.append({
            "trial": i,
            "order": "-".join(aff.order_for_trial(i)),
            "separate": _run(sep_c, sep_e, "separate"),
            "fused": _run(fus_c, fus_e, "fused"),
            "delta_chain_pct": (fus_c - sep_c) / sep_c * 100.0,
            "delta_e2e_pct": (fus_e - sep_e) / sep_e * 100.0,
        })
    return pairs


def _manifest():
    return {"git_sha": "0" * 40, "git_dirty": False,
            "binary": "build/fft_check", "binary_sha256": "a" * 64,
            "kernel_objects": {"fft_radix2.o": "b" * 64,
                               "fft_long.o": "b" * 64,
                               "fft_real.o": "b" * 64},
            "config_sha256": {"x.json": "c" * 64},
            "build": {"script": "s", "recipe": "r"},
            "soc": "Ascend910_9382", "cann_version": "9.0.0",
            "driver_version": "1", "driver_source": "env:AB_DRIVER_VERSION",
            "python": "3.11", "env": {}, "boundary": "device",
            "allow_dirty": False}


def _snap():
    return {"utc": "2026-10-10T00:00:00+00:00",
            "loadavg": {"1m": 0.1, "5m": 0.1, "15m": 0.1},
            "uptime_s": 1.0,
            "npu_smi": {"npus": [], "chips": [],
                        "frequency": "unavailable"}}


def _point(n, b, delta_pct=-10.0):
    pairs = _pairs(delta_pct)
    stats = aff.pair_stats(pairs)
    return {"n": n, "b": b, "kernel_counts": dict(aff.KERNEL_COUNT),
            "ub_peak_bytes": aff.ub_peak_bytes(),
            "separate_modeled_payload_gm_rw_bytes":
                aff.modeled_payload_gm_rw("separate", n, b),
            "fused_modeled_payload_gm_rw_bytes":
                aff.modeled_payload_gm_rw("fused", n, b),
            "launch_blocks": {"separate": 48, "fused": 48},
            "pairs": pairs, "stats": stats,
            "env_monitor": {"before": _snap(), "after": _snap()},
            "verdict": aff.point_verdict(stats)}


def _doc(delta_pct=-10.0, selected=None):
    selected = tuple(selected or aff.POINTS)
    points = [_point(n, b, delta_pct) for n, b in selected]
    verdicts = {f"{p['n']}x{p['b']}": p["verdict"] for p in points}
    doc = {
        "generated_utc": "2026-10-09T00:00:00+00:00",
        "manifest": _manifest(),
        "command": "python3 scripts/collect_boundary_attribution.py",
        "binary": "build/fft_check (AB_INPUT_SEQ named input modes)",
        "boundary": "device",
        "impls": list(aff.IMPLS),
        "grid": [list(p) for p in selected],
        "trials_per_point": 5,
        "reps": aff.REPS,
        "threshold": aff.THRESHOLD,
        "kernel_counts": dict(aff.KERNEL_COUNT),
        "ub_peak_bytes": aff.ub_peak_bytes(),
        "env_monitor": {"before": _snap(), "after": _snap()},
        "points": points,
        "gate": aff.evaluate_gate(verdicts),
        "problems": [],
        "incomplete": [],
        "status": "pass",
    }
    if selected != aff.POINTS:
        doc["session"] = ("cross-session reproducibility subset "
                          "(PR-B review R1.1)")
    return doc


class ConstantsTest(unittest.TestCase):
    def test_ub_peak_bytes(self):
        # AB_FUSED_UB_BYTES = 4 * 128 * 32 * 8 = 131072 (bIn+bOut+bIdx+bTw)
        self.assertEqual(aff.ub_peak_bytes(), 131072)

    def test_modeled_payload_gm_rw_matches_descriptor_contract(self):
        # 8192x1 tensor = 65536 B; 6 launches separate / 5 fused
        # (payload model only: excludes twiddle/index/coeff traffic)
        self.assertEqual(aff.modeled_payload_gm_rw("separate", 8192, 1),
                         786432)
        self.assertEqual(aff.modeled_payload_gm_rw("fused", 8192, 1),
                         655360)
        self.assertEqual(aff.modeled_payload_gm_rw("fused", 65536, 47),
                         5 * 2 * 65536 * 47 * 8)

    def test_kernel_counts(self):
        self.assertEqual(aff.KERNEL_COUNT, {"separate": 6, "fused": 5})

    def test_order_alternates_by_parity(self):
        self.assertEqual(aff.order_for_trial(0), ("separate", "fused"))
        self.assertEqual(aff.order_for_trial(1), ("fused", "separate"))
        self.assertEqual(aff.order_for_trial(2), ("separate", "fused"))

    def test_blocks_of_parses_result_line(self):
        out = "n=8192 batch=1 blocks=48  maxAbs=1.0e-06 maxRel=1.0e-07 " \
              "(worst idx 3)\n"
        self.assertEqual(aff.blocks_of(out), 48)
        self.assertIsNone(aff.blocks_of("no result line"))


class StatsAndGateTest(unittest.TestCase):
    def test_pair_stats(self):
        stats = aff.pair_stats(_pairs(-10.0))
        self.assertAlmostEqual(stats["median_delta_chain_pct"], -10.0)
        self.assertEqual(stats["not_slower_pairs"], 5)
        self.assertAlmostEqual(stats["worst_delta_chain_pct"], -10.0)
        self.assertAlmostEqual(stats["chain_fused_median"],
                               stats["chain_separate_median"] * 0.9,
                               places=6)

    def test_point_verdict_candidate(self):
        stats = aff.pair_stats(_pairs(-10.0))
        self.assertEqual(aff.point_verdict(stats), "candidate")

    def test_point_verdict_exact_threshold_is_candidate(self):
        stats = aff.pair_stats(_pairs(-5.0))
        self.assertEqual(aff.point_verdict(stats), "candidate")

    def test_point_verdict_hold_when_improvement_short(self):
        stats = aff.pair_stats(_pairs(-3.0))
        self.assertEqual(aff.point_verdict(stats), "hold")

    def test_point_verdict_hold_when_pairs_too_few(self):
        # median improves a lot but only 3/5 pairs not slower (two tiny
        # wins, two tiny losses) -> fails the 4/5 pairing rule
        pairs = _pairs(-10.0)
        pairs[0]["delta_chain_pct"] = 1.0
        pairs[1]["delta_chain_pct"] = 1.0
        stats = aff.pair_stats(pairs)
        self.assertEqual(stats["not_slower_pairs"], 3)
        self.assertEqual(aff.point_verdict(stats), "hold")

    def test_point_verdict_regress(self):
        stats = aff.pair_stats(_pairs(+4.0))
        self.assertEqual(aff.point_verdict(stats), "regress")

    def test_min_not_slower(self):
        self.assertEqual(aff.min_not_slower(5), 4)
        self.assertEqual(aff.min_not_slower(6), 5)

    def test_gate_all_candidates_promotes(self):
        v = {f"{n}x{b}": "candidate" for n, b in aff.POINTS}
        gate = aff.evaluate_gate(v)
        self.assertTrue(gate["promoted"])
        self.assertFalse(gate["fallback_separate"])
        self.assertEqual(gate["candidates"], sorted(gate["candidates"]))
        self.assertEqual(len(gate["candidates"]), 4)

    def test_gate_regression_forces_fallback(self):
        v = {f"{n}x{b}": "candidate" for n, b in aff.POINTS}
        v["32768x47"] = "regress"
        gate = aff.evaluate_gate(v)
        self.assertTrue(gate["fallback_separate"])
        self.assertFalse(gate["promoted"])
        self.assertEqual(gate["regressing_points"], ["32768x47"])

    def test_gate_hold_is_not_promoted(self):
        v = {f"{n}x{b}": "candidate" for n, b in aff.POINTS}
        v["8192x1"] = "hold"
        gate = aff.evaluate_gate(v)
        self.assertFalse(gate["promoted"])
        self.assertFalse(gate["fallback_separate"])


class RunProblemsTest(unittest.TestCase):
    def sample(self):
        segs = _segs(SEP_SEGS, 100.0)
        return {"rc": 0, "pass": True, "max_rel": 1e-7,
                "scopes": {"plan_setup": 1.0, "first_use": 2.0,
                           "e2e_mean": 100.0, "e2e_min": 95.0,
                           "h2d": 10.0, "device_chain": 100.0,
                           "d2h": 11.0, "reps": 5},
                "segments": segs, "boundary_impl": "separate",
                "blocks": 48}

    def test_good_run_accepted(self):
        self.assertEqual(aff.run_problems(self.sample(), "separate"), [])

    def test_missing_boundary_impl_rejected(self):
        s = self.sample()
        s["boundary_impl"] = None
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("boundary_impl" in x for x in got))

    def test_five_segments_under_separate_rejected(self):
        s = self.sample()
        s["segments"] = _segs(FUS_SEGS, 100.0)
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("twiddle missing" in x for x in got))

    def test_missing_blocks_rejected(self):
        s = self.sample()
        s["blocks"] = None
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("blocks=" in x for x in got))

    def test_nonzero_rc_rejected(self):
        s = self.sample()
        s["rc"] = 507035
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("rc=507035" in x for x in got))

    def test_missing_pass_rejected(self):
        s = self.sample()
        s["pass"] = False
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("PASS line missing" in x for x in got))

    def test_wrong_impl_rejected(self):
        s = self.sample()
        s["boundary_impl"] = "fused"     # run claims the other impl
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("boundary_impl" in x for x in got))

    def test_missing_scopes_rejected(self):
        s = self.sample()
        s["scopes"] = None
        got = aff.run_problems(s, "separate")
        self.assertTrue(any("scopes missing" in x for x in got))


class MakePairTest(unittest.TestCase):
    def _samples(self):
        segs_s = _segs(SEP_SEGS, 100.0)
        segs_f = _segs(FUS_SEGS, 90.0)
        base = {"rc": 0, "pass": True, "blocks": 48, "max_rel": 1e-7}
        return {
            "separate": dict(base, boundary_impl="separate", segments=segs_s,
                             scopes={"device_chain": 100.0}, e2e_us=1000.0),
            "fused": dict(base, boundary_impl="fused", segments=segs_f,
                          scopes={"device_chain": 90.0}, e2e_us=900.0),
        }

    def test_pair_retains_attestation(self):
        pair = aff.make_pair(0, ("separate", "fused"), self._samples())
        self.assertEqual(pair["separate"]["rc"], 0)
        self.assertIs(pair["separate"]["pass"], True)
        self.assertEqual(pair["separate"]["boundary_impl"], "separate")
        self.assertEqual(pair["fused"]["boundary_impl"], "fused")
        self.assertEqual(pair["trial"], 0)
        self.assertEqual(pair["order"], "separate-fused")
        self.assertAlmostEqual(pair["delta_chain_pct"], -10.0)


class VerifyAttributionTest(unittest.TestCase):
    def test_good_document_accepted(self):
        self.assertEqual(aff.verify_attribution(_doc()), [])

    def test_verify_reads_path(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(_doc(), fh)
            path = fh.name
        try:
            self.assertEqual(aff.verify_attribution(path), [])
        finally:
            Path(path).unlink()

    def test_tampered_stats_rejected(self):
        doc = _doc()
        doc["points"][0]["stats"]["median_delta_chain_pct"] = -99.0
        got = aff.verify_attribution(doc)
        self.assertTrue(any("median_delta_chain_pct" in x for x in got))

    def test_tampered_verdict_rejected(self):
        doc = _doc()
        doc["points"][1]["verdict"] = "hold"
        got = aff.verify_attribution(doc)
        self.assertTrue(any("verdict" in x for x in got))

    def test_tampered_gate_rejected(self):
        doc = _doc()
        doc["gate"]["promoted"] = False
        got = aff.verify_attribution(doc)
        self.assertTrue(any(x.startswith("gate ") for x in got))

    def test_wrong_segment_contract_rejected(self):
        doc = _doc()
        # fused run claims the six-span separate segments
        doc["points"][0]["pairs"][0]["fused"]["segments"] = \
            doc["points"][0]["pairs"][0]["separate"]["segments"]
        got = aff.verify_attribution(doc)
        self.assertTrue(any("fused: segments" in x for x in got))

    def test_tampered_delta_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][2]["delta_chain_pct"] = -1.0
        got = aff.verify_attribution(doc)
        self.assertTrue(any("delta_chain_pct" in x for x in got))

    def test_tampered_pair_order_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][1]["order"] = "separate-fused"
        got = aff.verify_attribution(doc)
        self.assertTrue(any("order" in x for x in got))

    def test_missing_point_rejected(self):
        doc = _doc()
        doc["points"] = doc["points"][:-1]
        doc["grid"] = doc["grid"][:-1]
        got = aff.verify_attribution(doc)
        self.assertTrue(any("grid" in x for x in got))

    def test_wrong_modeled_payload_gm_rejected(self):
        doc = _doc()
        doc["points"][2]["fused_modeled_payload_gm_rw_bytes"] = 1
        got = aff.verify_attribution(doc)
        self.assertTrue(any("modeled_payload_gm_rw_bytes" in x for x in got))

    def test_wrong_kernel_counts_rejected(self):
        doc = _doc()
        doc["kernel_counts"] = {"separate": 6, "fused": 6}
        got = aff.verify_attribution(doc)
        self.assertTrue(any("kernel_counts" in x for x in got))

    def test_wrong_ub_peak_rejected(self):
        doc = _doc()
        doc["ub_peak_bytes"] = 98304
        got = aff.verify_attribution(doc)
        self.assertTrue(any("ub_peak_bytes" in x for x in got))

    def test_status_mismatch_rejected(self):
        doc = _doc()
        doc["status"] = "incomplete"
        got = aff.verify_attribution(doc)
        self.assertTrue(any("status" in x for x in got))

    def test_stored_problems_mismatch_rejected(self):
        doc = _doc()
        doc["problems"] = ["invented"]
        got = aff.verify_attribution(doc)
        self.assertTrue(any("stored problems list" in x for x in got))

    def test_missing_top_level_key_rejected(self):
        doc = _doc()
        doc.pop("gate")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("missing top-level key 'gate'" in x
                            for x in got))

    def test_doc_not_mutated_by_verify(self):
        doc = _doc()
        snapshot = copy.deepcopy(doc)
        aff.verify_attribution(doc)
        self.assertEqual(doc, snapshot)

    # ---- R1.0 failure injection: every corrupt fixture must be rejected --

    def test_pair_nonzero_rc_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][2]["fused"]["rc"] = 1
        got = aff.verify_attribution(doc)
        self.assertTrue(any("fused: rc 1 != 0" in x for x in got))

    def test_pair_missing_pass_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][2]["fused"]["pass"] = False
        got = aff.verify_attribution(doc)
        self.assertTrue(any("pass False != True" in x for x in got))

    def test_pair_wrong_boundary_impl_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][2]["fused"]["boundary_impl"] = "separate"
        got = aff.verify_attribution(doc)
        self.assertTrue(any("boundary_impl" in x for x in got))

    def test_duplicate_point_rejected(self):
        doc = _doc()
        doc["points"].append(copy.deepcopy(doc["points"][0]))
        got = aff.verify_attribution(doc)
        self.assertTrue(any("duplicate point" in x for x in got))
        self.assertTrue(any("5 points != 4" in x for x in got))

    def test_missing_point_rejected_even_with_matching_grid(self):
        doc = _doc()
        doc["points"] = doc["points"][:-1]           # grid not tampered
        got = aff.verify_attribution(doc)
        self.assertTrue(any("3 points != 4" in x for x in got))
        self.assertTrue(any("missing from grid" in x for x in got))

    def test_trial_index_mismatch_rejected(self):
        doc = _doc()
        doc["points"][0]["pairs"][3]["trial"] = 0    # duplicate index
        got = aff.verify_attribution(doc)
        self.assertTrue(any("trial 0 != index 3" in x for x in got))

    def test_nan_chain_rejected(self):
        doc = _doc()
        doc["points"][1]["pairs"][1]["separate"]["chain"] = float("nan")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("not finite" in x or "nan" in x for x in got))

    def test_inf_e2e_rejected(self):
        doc = _doc()
        doc["points"][1]["pairs"][1]["fused"]["e2e"] = float("inf")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("e2e" in x for x in got))

    def test_negative_chain_rejected(self):
        doc = _doc()
        doc["points"][2]["pairs"][0]["fused"]["chain"] = -5.0
        got = aff.verify_attribution(doc)
        self.assertTrue(any("chain -5.0" in x for x in got))

    def test_nan_segment_rejected(self):
        doc = _doc()
        doc["points"][2]["pairs"][0]["fused"]["segments"]["fft1"] = \
            float("nan")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("segment fft1" in x for x in got))

    def test_nan_delta_rejected(self):
        doc = _doc()
        doc["points"][3]["pairs"][4]["delta_chain_pct"] = float("nan")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("delta_chain_pct" in x for x in got))

    def test_zero_e2e_rejected(self):
        doc = _doc()
        doc["points"][3]["pairs"][2]["separate"]["e2e"] = 0.0
        got = aff.verify_attribution(doc)
        self.assertTrue(any("e2e 0.0" in x for x in got))

    def test_missing_env_monitor_rejected(self):
        doc = _doc()
        doc.pop("env_monitor")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("missing top-level key 'env_monitor'" in x
                            for x in got))

    def test_point_missing_env_monitor_rejected(self):
        doc = _doc()
        doc["points"][1].pop("env_monitor")
        got = aff.verify_attribution(doc)
        self.assertTrue(any("env_monitor.before" in x for x in got))

    def test_session_subset_document_accepted(self):
        doc = _doc(selected=((65536, 47),))
        self.assertIn("session", doc)
        self.assertEqual(aff.verify_attribution(doc), [])

    def test_session_grid_outside_points_rejected(self):
        doc = _doc(selected=((65536, 47),))
        doc["grid"] = [[32768, 47], [9999, 9]]
        got = aff.verify_attribution(doc)
        self.assertTrue(any("session grid" in x for x in got))

    def test_main_archive_must_cover_full_grid(self):
        doc = _doc(selected=((8192, 1), (16384, 47)))
        doc.pop("session")           # pretend it is the gate archive
        got = aff.verify_attribution(doc)
        self.assertTrue(any("grid mismatch" in x for x in got))
        self.assertTrue(any("2 points != 4" in x for x in got))


class ParsePointsTest(unittest.TestCase):
    def test_parses_and_validates(self):
        self.assertEqual(aff.parse_points("8192x1,65536x47"),
                         ((8192, 1), (65536, 47)))
        with self.assertRaises(ValueError):
            aff.parse_points("9999x9")
        with self.assertRaises(ValueError):
            aff.parse_points("8192x1,8192x1")
        with self.assertRaises(ValueError):
            aff.parse_points("")


if __name__ == "__main__":
    unittest.main()
