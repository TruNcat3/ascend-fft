#!/usr/bin/env python3
"""PR-B R1 attribution: interleaved separate/fused device-chain trials.

Four priority points x >=5 independent paired trials.  Within each pair the
two impls run back-to-back (fft_check under AB_BOUNDARY=device +
AB_LONG_BOUNDARY_IMPL=separate|fused, reps=5) with the order alternating by
pair parity, so slow machine drift hits both impls equally.  Every run is
hard-accepted before it enters a pair: rc==0, PASS, maxRel under threshold,
scopes valid for long_device, segments matching the impl (six spans separate
/ five fused, telescoping into device_chain) and the boundary_impl
self-report; a run that fails any of this poisons its pair into `problems`.

Archive: results/evidence/long-fft-boundary-attribution/attribution.json
(+ outputs.txt transcripts).  Per point the archive reports the kernel
counts (6/5), the modeled payload GM bytes (launches x 2 x n x batch x 8,
the descriptor `modeled_payload_gm_rw_bytes` contract -- a payload MODEL
that excludes twiddle/index/coefficient auxiliary transactions and is not
a profiler measurement), per-impl segment medians, the UB peak
(AB_FUSED_UB_BYTES, identical for both impls because bTw is statically
reserved) and the launch grid blocks (not measured active AIVs; that is a
profiler question).  Every run is hard-accepted by run_problems()
BEFORE it may enter a pair (R1.0: called unconditionally -- a non-zero
rc, missing PASS/scopes/segments, a wrong boundary_impl or any contract
violation rejects the run and poisons its pair; rejected runs never
contribute timing samples).  Pairs retain per-run rc/pass/boundary_impl
and verify_attribution independently re-derives every verdict from the
stored raws (no trust in collector summaries).  Gate from the plan:

  point candidate  = >=4/5 pairs not slower AND median chain improvement >=5%
  fallback         = any point regressing >3% keeps the separate fallback

  python3 scripts/collect_boundary_attribution.py
  python3 scripts/collect_boundary_attribution.py --verify PATH

tests/test_boundary_attribution.py exercises the pure stats/gate/verify
helpers without hardware.
"""

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_long_fft_evidence as coll  # noqa: E402  (shared run/manifest)
import scopes  # noqa: E402

POINTS = ((8192, 1), (16384, 47), (32768, 47), (65536, 47))
TRIALS = 5
REPS = 5
THRESHOLD = 1e-4
IMPROVE_PCT = 5.0     # median chain improvement required for candidate
REGRESS_PCT = 3.0     # median regression beyond this keeps separate
NOT_SLOWER_FRACTION = 0.8   # >=80% of pairs must not be slower (4 of 5)
IMPLS = ("separate", "fused")
KERNEL_COUNT = {"separate": 6, "fused": 5}
MODE_BY_IMPL = {"separate": "device", "fused": "device-fused"}
OUT_DIR = ROOT / "results" / "evidence" / "long-fft-boundary-attribution"
UB_HEADER = ROOT / "include" / "butterfly" / "long_fft_ub.h"


def ub_peak_bytes(path=UB_HEADER):
    """AB_FUSED_UB_BYTES = 4 * LT_H * LT_W * 8 (bIn+bOut+bIdx+bTw)."""
    text = Path(path).read_text(encoding="utf-8")
    h = int(re.search(r"#define AB_LT_H (\d+)u", text).group(1))
    w = int(re.search(r"#define AB_LT_W (\d+)u", text).group(1))
    return 4 * h * w * 8


def modeled_payload_gm_rw(impl, n, batch):
    """Descriptor modeled_payload_gm_rw_bytes contract (R1.0 rename):
    every launch is modeled as streaming one full complex tensor in and
    out of GM (2 * n * batch * 8).  Payload MODEL only -- excludes
    twiddle/index/coefficient auxiliary GM transactions; not a profiler
    measurement."""
    return KERNEL_COUNT[impl] * 2 * n * batch * 8


def order_for_trial(i):
    """Pair order alternates by parity: drift cancels across the pairs."""
    return ("separate", "fused") if i % 2 == 0 else ("fused", "separate")


def blocks_of(out):
    m = re.search(r"n=\d+ batch=\d+ blocks=(\d+)", out)
    return int(m.group(1)) if m else None


def run_impl(n, b, impl):
    """One fft_check invocation -> (rc, parsed sample with blocks)."""
    cmd = ["./build/fft_check", str(n), str(b), str(REPS)]
    rc, out = coll.run(cmd, env=coll.mode_env(MODE_BY_IMPL[impl],
                                               AB_E2E=str(REPS)))
    sample = coll.parse_trial(out)
    sample["rc"] = rc
    sample["blocks"] = blocks_of(out)
    return sample, out


def run_problems(sample, impl, threshold=THRESHOLD):
    """Hard acceptance for one run before it may enter a pair."""
    problems = []
    tag = f"impl={impl}"
    if sample.get("rc") != 0:
        problems.append(f"{tag} rc={sample.get('rc')} (want 0)")
    if not sample.get("pass"):
        problems.append(f"{tag} PASS line missing")
    mr = sample.get("max_rel")
    if mr is None:
        problems.append(f"{tag} max_rel missing")
    elif not math.isfinite(mr) or mr > threshold:
        problems.append(f"{tag} max_rel {mr} > threshold {threshold}")
    fields = sample.get("scopes")
    if not fields:
        problems.append(f"{tag} scopes missing")
    else:
        problems += [f"{tag} {x}" for x in
                     scopes.validate_scopes(fields, "long_device")]
    seg = sample.get("segments")
    if not fields or seg is None:
        problems.append(f"{tag} segments missing")
    else:
        problems += [f"{tag} {x}" for x in
                     scopes.validate_segments(seg, fields, "long_device",
                                              impl=impl)]
    bi = sample.get("boundary_impl")
    if bi != impl:
        problems.append(f"{tag} boundary_impl {bi!r} != {impl!r}")
    if sample.get("blocks") is None:
        problems.append(f"{tag} blocks= line missing")
    return problems


def make_pair(trial, order, samples):
    pair = {"trial": trial, "order": "-".join(order)}
    for impl in IMPLS:
        s = samples[impl]
        # R1.0: retain the per-run attestation so the verifier can
        # independently re-check it without trusting the collector.
        pair[impl] = {
            "rc": s["rc"],
            "pass": bool(s.get("pass")),
            "boundary_impl": s.get("boundary_impl"),
            "chain": s["scopes"]["device_chain"],
            "e2e": s["e2e_us"],
            "segments": s["segments"],
            "blocks": s["blocks"],
            "max_rel": s["max_rel"],
        }
    sep, fus = pair["separate"], pair["fused"]
    pair["delta_chain_pct"] = (fus["chain"] - sep["chain"]) / sep["chain"] \
        * 100.0
    pair["delta_e2e_pct"] = (fus["e2e"] - sep["e2e"]) / sep["e2e"] * 100.0
    return pair


def pair_stats(pairs):
    """Per-point paired statistics (pure; unit-tested)."""
    dc = [p["delta_chain_pct"] for p in pairs]
    de = [p["delta_e2e_pct"] for p in pairs]
    sep_c = [p["separate"]["chain"] for p in pairs]
    fus_c = [p["fused"]["chain"] for p in pairs]
    sep_e = [p["separate"]["e2e"] for p in pairs]
    fus_e = [p["fused"]["e2e"] for p in pairs]
    return {
        "median_delta_chain_pct": statistics.median(dc),
        "median_delta_e2e_pct": statistics.median(de),
        "worst_delta_chain_pct": max(dc),
        "not_slower_pairs": sum(1 for d in dc if d <= 0.0),
        "chain_separate_median": statistics.median(sep_c),
        "chain_fused_median": statistics.median(fus_c),
        "e2e_separate_median": statistics.median(sep_e),
        "e2e_fused_median": statistics.median(fus_e),
    }


def min_not_slower(trials=TRIALS):
    """Plan gate: >=4 of 5 pairs not slower (>=80%, rounds up)."""
    return math.ceil(NOT_SLOWER_FRACTION * trials)


def point_verdict(stats, trials=TRIALS):
    """candidate | hold | regress for one point (pure; unit-tested)."""
    if stats["median_delta_chain_pct"] > REGRESS_PCT:
        return "regress"
    if (stats["not_slower_pairs"] >= min_not_slower(trials)
            and stats["median_delta_chain_pct"] <= -IMPROVE_PCT):
        return "candidate"
    return "hold"


def evaluate_gate(verdicts):
    """verdicts: {point_key: verdict} -> plan-level gate (pure)."""
    keys = sorted(verdicts)
    regress = [k for k in keys if verdicts[k] == "regress"]
    candidates = [k for k in keys if verdicts[k] == "candidate"]
    return {
        "min_not_slower_pairs": min_not_slower(),
        "improve_pct_required": IMPROVE_PCT,
        "regress_pct_fallback": REGRESS_PCT,
        "candidates": candidates,
        "regressing_points": regress,
        "fallback_separate": bool(regress),
        "promoted": (len(candidates) == len(keys) and not regress),
    }


# ---------- archive gate (pure; unit-tested) ------------------------------
def _snap_problem(snap, where):
    if not isinstance(snap, dict) or not isinstance(snap.get("utc"), str):
        return [f"{where}: env_monitor snapshot missing utc"]
    probs = []
    if "npu_smi" not in snap:
        probs.append(f"{where}: env_monitor snapshot missing npu_smi")
    if "loadavg" not in snap:
        probs.append(f"{where}: env_monitor snapshot missing loadavg")
    return probs


def parse_points(spec):
    """'8192x1,65536x47' -> ((8192,1),(65536,47)); validates membership."""
    points = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n_s, b_s = tok.lower().split("x")
        point = (int(n_s), int(b_s))
        if point not in POINTS:
            raise ValueError(f"point {tok!r} not in the attribution grid "
                             f"{POINTS!r}")
        if point in points:
            raise ValueError(f"duplicate point {tok!r}")
        points.append(point)
    if not points:
        raise ValueError("empty point selection")
    return tuple(points)


def verify_attribution(doc, threshold=None):
    """Re-run the whole contract on a stored attribution.json.

    Returns [] only for a fully attested archive: stats and verdicts must
    recompute identically from the raw pairs, every run must carry the
    impl-matched segments + rc/pass/boundary_impl + correctness, the
    structural constants (kernel counts, modeled payload GM bytes, UB
    peak, launch blocks, env_monitor snapshots) must match the descriptor
    contract, and the point set must be a complete grid.  The main gate
    archive must cover the full 4-point POINTS grid; a `session` archive
    (R1.1 cross-session reproducibility run) may declare a unique subset.
    """
    if isinstance(doc, (str, Path)):
        doc = json.loads(Path(doc).read_text(encoding="utf-8"))
    need = ("generated_utc", "manifest", "binary", "boundary", "impls",
            "grid", "trials_per_point", "reps", "threshold",
            "kernel_counts", "ub_peak_bytes", "env_monitor", "points",
            "gate", "problems", "incomplete", "status")
    missing = [k for k in need if k not in doc]
    if missing:
        return [f"missing top-level key {k!r}" for k in missing]
    problems = []
    if doc["boundary"] != "device":
        problems.append(f"boundary {doc['boundary']!r} != 'device'")
    if tuple(doc["impls"]) != IMPLS:
        problems.append(f"impls {doc['impls']!r} != {list(IMPLS)!r}")
    for where, snap in (("env_monitor.before",
                         (doc.get("env_monitor") or {}).get("before")),
                        ("env_monitor.after",
                         (doc.get("env_monitor") or {}).get("after"))):
        problems += _snap_problem(snap, where)
    thr = doc.get("threshold", THRESHOLD) if threshold is None else threshold
    trials = doc["trials_per_point"]
    if trials < 5:
        problems.append(f"trials_per_point {trials} < 5 (plan floor)")
    if doc.get("reps") != REPS:
        problems.append(f"reps {doc.get('reps')!r} != {REPS}")
    if doc.get("ub_peak_bytes") != ub_peak_bytes():
        problems.append(f"ub_peak_bytes {doc.get('ub_peak_bytes')!r} != "
                        f"{ub_peak_bytes()}")
    kc = doc.get("kernel_counts")
    if kc != KERNEL_COUNT:
        problems.append(f"kernel_counts {kc!r} != {KERNEL_COUNT!r}")
    grid = doc.get("grid") or []
    session = bool(doc.get("session"))
    if session:
        if (not grid or len(set(map(tuple, grid))) != len(grid)
                or any(tuple(g) not in POINTS for g in grid)):
            problems.append(f"session grid {grid!r} is not a unique "
                            f"subset of {list(POINTS)!r}")
        expected = [tuple(g) for g in grid]
    else:
        expected = list(POINTS)
        if [tuple(g) for g in grid] != expected:
            problems.append(f"grid mismatch: {grid!r} != {expected!r}")
    # R1.0: strict structural audit -- exactly len(expected) unique
    # points, no duplicates, no holes, no extra entries.
    seen = set()
    raw_points = doc.get("points", [])
    if len(raw_points) != len(expected):
        problems.append(f"{len(raw_points)} points != {len(expected)} "
                        "(grid size)")
    verdicts = {}
    for p in raw_points:
        n, b = p.get("n"), p.get("b")
        where = f"point n={n} b={b}"
        if (n, b) not in expected:
            problems.append(f"{where}: not in the attribution grid")
            continue
        if (n, b) in seen:
            problems.append(f"{where}: duplicate point")
            continue
        seen.add((n, b))
        pem = p.get("env_monitor") or {}
        for w in ("before", "after"):
            problems += _snap_problem(pem.get(w),
                                      f"{where} env_monitor.{w}")
        if p.get("kernel_counts") != KERNEL_COUNT:
            problems.append(f"{where}: kernel_counts "
                            f"{p.get('kernel_counts')!r} != {KERNEL_COUNT!r}")
        if p.get("ub_peak_bytes") != ub_peak_bytes():
            problems.append(f"{where}: ub_peak_bytes "
                            f"{p.get('ub_peak_bytes')!r} != {ub_peak_bytes()}")
        lbs = p.get("launch_blocks") or {}
        for impl in IMPLS:
            v = lbs.get(impl)
            if not isinstance(v, int) or v <= 0:
                problems.append(f"{where}: launch_blocks[{impl}] {v!r}")
            got = p.get(f"{impl}_modeled_payload_gm_rw_bytes")
            if got != modeled_payload_gm_rw(impl, n, b):
                problems.append(f"{where}: {impl} "
                                f"modeled_payload_gm_rw_bytes {got!r} != "
                                f"{modeled_payload_gm_rw(impl, n, b)}")
        pairs = p.get("pairs") or []
        if len(pairs) != trials:
            problems.append(f"{where}: {len(pairs)} pairs != {trials}")
        for i, pair in enumerate(pairs):
            tag = f"{where} pair[{i}]"
            if pair.get("trial") != i:
                problems.append(f"{tag}: trial {pair.get('trial')!r} != "
                                f"index {i}")
            order = str(pair.get("order", ""))
            if order != "-".join(order_for_trial(i)):
                problems.append(f"{tag}: order {order!r} != trial-parity "
                                "alternation")
            for impl in IMPLS:
                run = pair.get(impl) or {}
                rtag = f"{tag} {impl}"
                # R1.0: independent per-run re-verification of the raw
                # attestation -- rc, PASS, boundary_impl, then the
                # scope/segment/time contract on the stored values.
                if run.get("rc") != 0:
                    problems.append(f"{rtag}: rc {run.get('rc')!r} != 0")
                if run.get("pass") is not True:
                    problems.append(f"{rtag}: pass {run.get('pass')!r} "
                                    "!= True")
                if run.get("boundary_impl") != impl:
                    problems.append(f"{rtag}: boundary_impl "
                                    f"{run.get('boundary_impl')!r} != "
                                    f"{impl!r}")
                seg = run.get("segments")
                if not isinstance(seg, dict) or set(seg) != set(
                        scopes.SEGMENT_FIELDS if impl == "separate"
                        else scopes.FUSED_SEGMENT_FIELDS):
                    problems.append(f"{rtag}: segments "
                                    f"{sorted(seg) if isinstance(seg, dict) else seg!r} "
                                    f"!= {impl} contract")
                else:
                    for k, sv in seg.items():
                        if (not isinstance(sv, (int, float))
                                or not math.isfinite(sv) or sv <= 0):
                            problems.append(f"{rtag}: segment {k} {sv!r} "
                                            "not finite positive")
                    if not math.isclose(
                            sum(seg.values()),
                            run.get("chain", float("nan")),
                            rel_tol=0.01, abs_tol=20.0):
                        problems.append(f"{rtag}: sum(segments) "
                                        f"{sum(seg.values())} != chain "
                                        f"{run.get('chain')}")
                mr = run.get("max_rel")
                if mr is None or not math.isfinite(mr) or mr > thr:
                    problems.append(f"{rtag}: max_rel {mr!r}")
                if not isinstance(run.get("blocks"), int) or run["blocks"] <= 0:
                    problems.append(f"{rtag}: blocks {run.get('blocks')!r}")
                chain = run.get("chain")
                e2e = run.get("e2e")
                for name, v in (("chain", chain), ("e2e", e2e)):
                    if not isinstance(v, (int, float)) or not math.isfinite(v) \
                            or v <= 0:
                        problems.append(f"{rtag}: {name} {v!r}")
            dc = pair.get("delta_chain_pct")
            de = pair.get("delta_e2e_pct")
            for name, dv in (("delta_chain_pct", dc), ("delta_e2e_pct", de)):
                if not isinstance(dv, (int, float)) or not math.isfinite(dv):
                    problems.append(f"{tag}: {name} {dv!r} not finite")
            if isinstance(dc, (int, float)) and math.isfinite(dc) and isinstance(
                    pair.get("separate", {}).get("chain"), (int, float)):
                sep = pair["separate"]["chain"]
                fus = pair["fused"]["chain"]
                if sep and not math.isclose(dc, (fus - sep) / sep * 100.0,
                                            rel_tol=1e-9, abs_tol=1e-9):
                    problems.append(f"{tag}: delta_chain_pct "
                                    f"{dc!r} != recomputed")
            if isinstance(de, (int, float)) and math.isfinite(de) and isinstance(
                    pair.get("separate", {}).get("e2e"), (int, float)):
                sep = pair["separate"]["e2e"]
                fus = pair["fused"]["e2e"]
                if sep and not math.isclose(de, (fus - sep) / sep * 100.0,
                                            rel_tol=1e-9, abs_tol=1e-9):
                    problems.append(f"{tag}: delta_e2e_pct {de!r} != recomputed")
        if pairs and len(pairs) == trials:
            stats = pair_stats(pairs)
            for k, v in stats.items():
                if not math.isfinite(v):
                    problems.append(f"{where}: stats {k} not finite")
                a = (p.get("stats") or {}).get(k)
                if a is None or not math.isfinite(a) or not math.isclose(
                        a, v, rel_tol=1e-9, abs_tol=1e-9):
                    problems.append(f"{where}: stats {k}={a!r} != {v!r}")
            verdict = point_verdict(stats, trials=trials)
            verdicts[f"{n}x{b}"] = verdict
            if p.get("verdict") != verdict:
                problems.append(f"{where}: verdict {p.get('verdict')!r} != "
                                f"{verdict!r}")
        else:
            problems.append(f"{where}: cannot verify stats/verdict "
                            "without exactly trials_per_point clean pairs")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        if missing:
            problems.append(f"points missing from grid: {missing!r}")
    if sorted(verdicts) == sorted(f"{n}x{b}" for n, b in expected):
        gate = evaluate_gate(verdicts)
        if doc.get("gate") != gate:
            problems.append(f"gate {doc.get('gate')!r} != recomputed {gate!r}")
    else:
        problems.append("points do not cover the full attribution grid")
    problems += [f"manifest: {x}" for x in
                 coll.verify_manifest(doc["manifest"], doc["boundary"])]
    incomplete = coll.manifest_incomplete(doc["manifest"])
    status = ("fail" if problems else
              "incomplete" if incomplete else "pass")
    out = list(problems)
    if status == "incomplete":
        out += [f"incomplete: {x}" for x in incomplete]
    if doc.get("status") != status:
        out.append(f"status {doc.get('status')!r} != recomputed {status!r}")
    if list(doc.get("problems") or []) != problems:
        out.append("stored problems list != recomputation "
                   f"({len(doc.get('problems') or [])} vs {len(problems)})")
    if list(doc.get("incomplete") or []) != incomplete:
        out.append("stored incomplete list != recomputation")
    return out


# ---------- collection ----------------------------------------------------
def collect(trials=TRIALS, allow_dirty=False, points=None):
    binary = ROOT / "build" / "fft_check"
    if not binary.is_file():
        return None, ["build/fft_check missing; run scripts/build.sh check"]
    if trials < 5:
        return None, ["trials must be >= 5 (plan floor)"]
    selected = tuple(points) if points else POINTS
    manifest = coll.build_manifest("device", allow_dirty)
    ub = ub_peak_bytes()
    problems, transcripts = [], []
    points_out = []
    env_before_doc = coll.env_monitor_snapshot()
    for n, b in selected:
        env_before = coll.env_monitor_snapshot()
        pairs, run_probs = [], []
        for i in range(trials):
            order = order_for_trial(i)
            samples, clean = {}, True
            for impl in order:
                sample, out = run_impl(n, b, impl)
                transcripts.append(
                    f"===== n={n} b={b} pair={i} impl={impl} =====\n{out}")
                # R1.0: unconditional hard acceptance.  Every violation
                # (non-zero rc, missing PASS/scopes/segments, wrong
                # boundary_impl, ...) is recorded AND poisons the pair --
                # a failed run may never enter a pair's timing sample.
                probs = run_problems(sample, impl)
                for x in probs:
                    run_probs.append(f"pair[{i}] {x}")
                if probs:
                    clean = False
                samples[impl] = sample
            if clean:
                pairs.append(make_pair(i, order, samples))
            else:
                run_probs.append(f"pair[{i}] rejected "
                                 "(run contract failed; not timed)")
        if len(pairs) != trials:
            run_probs.append(f"only {len(pairs)}/{trials} pairs completed")
        point = {
            "n": n, "b": b,
            "kernel_counts": dict(KERNEL_COUNT),
            "ub_peak_bytes": ub,
            "separate_modeled_payload_gm_rw_bytes":
                modeled_payload_gm_rw("separate", n, b),
            "fused_modeled_payload_gm_rw_bytes":
                modeled_payload_gm_rw("fused", n, b),
            # launch grid blocks (not measured active AIVs -- R1.0 rename)
            "launch_blocks": {
                impl: (int(statistics.median(
                    [p[impl]["blocks"] for p in pairs])) if pairs else None)
                for impl in IMPLS},
            "pairs": pairs,
        }
        if len(pairs) == trials:
            stats = pair_stats(pairs)
            point["stats"] = stats
            point["verdict"] = point_verdict(stats, trials=trials)
        point["env_monitor"] = {"before": env_before,
                                "after": coll.env_monitor_snapshot()}
        problems += [f"point n={n} b={b}: {x}" for x in run_probs]
        points_out.append(point)
    verdicts = {f"{p['n']}x{p['b']}": p["verdict"]
                for p in points_out if "verdict" in p}
    gate = (evaluate_gate(verdicts) if len(verdicts) == len(selected)
            else {"error": "incomplete verdicts"})
    problems += [f"gate: {x}" for x in
                 ([] if "error" not in gate else [gate.pop("error")])]
    incomplete = coll.manifest_incomplete(manifest)
    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "command": "python3 scripts/collect_boundary_attribution.py",
        "binary": "build/fft_check (AB_INPUT_SEQ named input modes)",
        "boundary": "device",
        "impls": list(IMPLS),
        "grid": [list(p) for p in selected],
        "trials_per_point": trials,
        "reps": REPS,
        "threshold": THRESHOLD,
        "kernel_counts": dict(KERNEL_COUNT),
        "ub_peak_bytes": ub,
        "env_monitor": {"before": env_before_doc,
                        "after": coll.env_monitor_snapshot()},
        "points": points_out,
        "gate": gate,
        "problems": problems,
        "incomplete": incomplete,
        "status": ("fail" if problems else
                   "incomplete" if incomplete else "pass"),
    }
    if selected != POINTS:
        # R1.1: a subset run is an auxiliary cross-session record, not
        # the plan-gate archive; verify_attribution checks it against
        # its declared unique subset grid.
        document["session"] = (
            "cross-session reproducibility subset "
            + "/".join(f"{n}x{b}" for n, b in selected)
            + " (PR-B review R1.1)")
    return (document, transcripts), problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=TRIALS,
                    help="paired trials per point (>=5, plan floor)")
    ap.add_argument("--points", metavar="NxB[,NxB...]",
                    help="subset of the attribution grid (default: all "
                         "4 points); a subset run is archived as an "
                         "independent cross-session record (R1.1)")
    ap.add_argument("--out-dir", metavar="DIR",
                    help="output directory (default: "
                         "results/evidence/long-fft-boundary-attribution)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="record a patch digest instead of refusing on a "
                         "dirty tree")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="publish and exit 0 even when provenance is "
                         "incomplete (status stays 'incomplete')")
    ap.add_argument("--verify", metavar="PATH",
                    help="verify a stored attribution.json and exit")
    args = ap.parse_args(argv)

    if args.verify:
        probs = verify_attribution(Path(args.verify))
        if probs:
            for x in probs:
                print(x, file=sys.stderr)
            print(f"{args.verify}: {len(probs)} problem(s)",
                  file=sys.stderr)
            return 1
        print(f"{args.verify}: archive accepted")
        return 0

    sha, dirty = coll.git_state()
    if dirty and not args.allow_dirty:
        print("refusing to publish attribution from a dirty tree "
              f"({sha[:8]}; commit first or pass --allow-dirty)",
              file=sys.stderr)
        return 2

    selected = None
    if args.points:
        try:
            selected = parse_points(args.points)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    result, problems = collect(trials=args.trials,
                               allow_dirty=args.allow_dirty,
                               points=selected)
    if result is None:
        for p in problems:
            print(p, file=sys.stderr)
        return 2
    document, transcripts = result
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs.txt").write_text("\n".join(transcripts),
                                         encoding="utf-8")
    (out_dir / "attribution.json").write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for p in document["points"]:
        st = p.get("stats")
        if st:
            print(f"n={p['n']} b={p['b']}: chain "
                  f"{st['chain_separate_median']:.1f} -> "
                  f"{st['chain_fused_median']:.1f} us "
                  f"({st['median_delta_chain_pct']:+.1f}%), "
                  f"not_slower {st['not_slower_pairs']}/{args.trials}, "
                  f"verdict={p['verdict']}")
    print(f"gate: candidates={document['gate'].get('candidates')} "
          f"fallback_separate={document['gate'].get('fallback_separate')} "
          f"promoted={document['gate'].get('promoted')} -> "
          f"{document['status']} -> {out_dir}")
    if problems:
        for p in problems:
            print(f"  problem: {p}", file=sys.stderr)
        return 1
    if document["status"] == "incomplete":
        for r in document["incomplete"]:
            print(f"  incomplete: {r}", file=sys.stderr)
        if not args.allow_incomplete:
            print("attribution incomplete; rerun with --allow-incomplete "
                  "to publish with status=incomplete anyway",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
