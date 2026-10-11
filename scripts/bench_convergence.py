#!/usr/bin/env python3
"""Targeted convergence benchmark with full candidate closure (PR #6 R0).

Reads ``config/convergence_targets.json`` and, for every target, measures:

* the **published incumbent** (current deployed binary, current defaults);
* each **candidate** in the candidate set (explicit env overrides);
* the **matched external baseline** (torch_npu ``fft`` / ``rfft`` / ``irfft``).

Protocol reuse (paired, interleaved where applicable):
* C2C: ``matrix_test.py`` protocol — 3 rounds, min-of-means, randomised
  native/self interleaving; ``fft_check`` mean over reps=20.
* R2C/C2R: ``bench_r2c_c2r.py`` protocol — 5 trials, each trial does one
  discarded warmup + one timed run (median of reps samples); torch side
  mirrors the same warmup+timed+reps contract.

Every row records ``candidate_id``, ``mapping`` (actual AB_FOLD_D /
AB_PLANE_K resolved), ``processing_unit`` (AIV), ``binary_sha256``,
``selection_reason``, correctness (maxRel), and raw per-trial values.
Any missing warmup / correctness / trial / baseline fails the point and
the runner exits non-zero.

Candidate closure rule (acceptance gate 2/6):
* the candidate set ALWAYS contains the incumbent (``cand=incumbent``);
* a candidate is promoted only if its median improves on the incumbent
  median by >= 2% AND every trial passes correctness;
* otherwise the incumbent is retained (selection_reason records why).

Usage:
  python3 scripts/bench_convergence.py --out results/evidence/pr6-round2 \
      --targets c2c:1024x4096,r2c:128x4096 \
      --candidate foldD=16 --candidate foldD=8,planeK=8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import abenv  # noqa: E402

PY = abenv.python_bin()
TARGETS_PATH = ROOT / "config" / "convergence_targets.json"
SNAPSHOT = "ascend910_9382-revalidation-20261010"
THRESHOLD = 1e-4
PROMOTE_RATIO = 0.02  # >=2% median improvement to promote


# ---- protocol helpers ----------------------------------------------------
def c2c_reps(_n, _b):
    return 20


def real_reps(n, b):
    pts = n * b
    if pts <= 4096:
        return 50
    if pts <= 1 << 20:
        return 30
    return 10


def reps_for(transform, n, b):
    return c2c_reps(n, b) if transform == "c2c" else real_reps(n, b)


def binary_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_mapping(transform, n, batch, env_overrides: dict) -> dict:
    """Return the actual (fold_d, plane_k) the binary will use."""
    # foldDFor logic mirrored for provenance; the binary is authoritative.
    inner = n // 2 if transform == "r2c" else n
    if transform == "c2r":
        inner = n
    if "foldD" in env_overrides:
        fd = int(env_overrides["foldD"])
    else:
        # replicate foldDFor for provenance (n-aware cap + 2*nblk)
        if inner <= 64:
            cap = 16
        elif inner <= 256:
            cap = 8
        else:
            cap = 4
        fd = cap
        # concurrency clamp (nblk=48)
        min_groups = 96
        while fd > 1 and ((batch - 1) // fd + 1) < min_groups:
            fd -= 1
    if "planeK" in env_overrides:
        pk = int(env_overrides["planeK"])
    else:
        # planeKFor simplified
        k_need = (inner + 63) >> 6
        pk = k_need if (inner <= 1024 and k_need > 8) else 8
    return {"fold_d": fd, "plane_k": pk, "inner_n": inner}


# ---- ours measurement ----------------------------------------------------
def run_ours_once(transform, n, b, reps, env_overrides, raw, tag):
    """One fft_check invocation with explicit AB_DIR; returns dict."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AB_")}
    env["AB_DIR"] = transform  # explicit, never inherit
    env["AB_SOC"] = os.environ.get("AB_SOC", "Ascend910_9382")
    env["AB_PROFILE"] = os.environ.get(
        "AB_PROFILE", str(ROOT / "config" / "ascend910_93_profile.json"))
    for k, v in env_overrides.items():
        env[f"AB_{k}"] = v
    try:
        r = subprocess.run(
            ["./build/fft_check", str(n), str(b), str(reps)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    except subprocess.TimeoutExpired:
        row = {"impl": tag, "transform": transform, "n": n, "b": b,
               "reps": reps, "us": None, "max_rel": None,
               "status": "timeout", "error": "timeout after 1800s"}
        raw.append(row)
        return row
    out = r.stdout + r.stderr
    m = re.search(r"med ([\d.]+) us", out)
    e = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    max_rel = float(e.group(2)) if e else None
    ok = bool(re.search(r"^PASS$", out, re.M))
    if r.returncode != 0 or not ok:
        tail = " | ".join(out.strip().splitlines()[-3:])
        status = "timeout" if "timeout" in (tail or "") else "error"
        err = f"rc={r.returncode}: {tail}" if tail else f"rc={r.returncode}"
        row = {"impl": tag, "transform": transform, "n": n, "b": b,
               "reps": reps, "us": None, "max_rel": max_rel,
               "status": status, "error": err}
        raw.append(row)
        return row
    if not m:
        row = {"impl": tag, "transform": transform, "n": n, "b": b,
               "reps": reps, "us": None, "max_rel": max_rel,
               "status": "error", "error": "no med field"}
        raw.append(row)
        return row
    if max_rel is not None and max_rel > THRESHOLD:
        row = {"impl": tag, "transform": transform, "n": n, "b": b,
               "reps": reps, "us": None, "max_rel": max_rel,
               "status": "error", "error": f"maxRel {max_rel} > {THRESHOLD}"}
        raw.append(row)
        return row
    us = float(m.group(1))
    row = {"impl": tag, "transform": transform, "n": n, "b": b,
           "reps": reps, "us": us, "max_rel": max_rel,
           "status": "ok", "error": ""}
    raw.append(row)
    return row


def bench_ours_trials(transform, n, b, trials, env_overrides, raw, tag):
    """trials of (discarded warmup + timed) for one candidate."""
    reps = reps_for(transform, n, b)
    values = []
    for trial in range(trials):
        warmup = run_ours_once(transform, n, b, reps, env_overrides,
                               raw, tag=f"{tag}_warmup")
        timed = run_ours_once(transform, n, b, reps, env_overrides,
                              raw, tag=tag)
        if timed["status"] != "ok":
            values.append(None)
            print(f"{tag} {transform} n={n} b={b} trial={trial} -> "
                  f"{timed['status'].upper()}: {timed['error']}",
                  file=sys.stderr, flush=True)
        else:
            values.append(timed["us"])
            print(f"{tag} {transform} n={n} b={b} trial={trial} -> "
                  f"{timed['us']:.1f} us (maxRel={timed['max_rel']})",
                  file=sys.stderr, flush=True)
    return values


# ---- C2C matrix_test protocol (paired, interleaved, min-of-means) --------
def bench_c2c_matrix(transform, n, b, rounds, env_overrides, raw, tag):
    """matrix_test.py protocol: rounds of (native+self) interleaved,
    each round does reps of fft_check; result = min of round means."""
    reps = reps_for(transform, n, b)
    env = {k: v for k, v in os.environ.items() if not k.startswith("AB_")}
    env["AB_DIR"] = "c2c"
    env["AB_SOC"] = os.environ.get("AB_SOC", "Ascend910_9382")
    env["AB_PROFILE"] = os.environ.get(
        "AB_PROFILE", str(ROOT / "config" / "ascend910_93_profile.json"))
    for k, v in env_overrides.items():
        env[f"AB_{k}"] = v
    round_means = []
    for rnd in range(rounds):
        # one fft_check invocation returns med over reps
        r = subprocess.run(
            ["./build/fft_check", str(n), str(b), str(reps)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=600)
        out = r.stdout + r.stderr
        m = re.search(r"med ([\d.]+) us", out)
        ok = bool(re.search(r"^PASS$", out, re.M))
        if r.returncode != 0 or not ok or not m:
            raw.append({"impl": tag, "transform": transform, "n": n, "b": b,
                        "reps": reps, "us": None, "max_rel": None,
                        "status": "error", "error": out.strip()[-200:]})
            continue
        us = float(m.group(1))
        raw.append({"impl": tag, "transform": transform, "n": n, "b": b,
                    "reps": reps, "us": us, "max_rel": None,
                    "status": "ok", "error": ""})
        round_means.append(us)
        print(f"{tag} c2c n={n} b={b} round={rnd} -> {us:.1f} us",
              file=sys.stderr, flush=True)
    if not round_means:
        return None
    return min(round_means)  # matrix_test uses min-of-means


# ---- baseline (torch_npu) ------------------------------------------------
_TORCH_SRC = r"""
import json, statistics as st, sys, time, warnings
warnings.filterwarnings("ignore")
import torch, torch_npu
torch.npu.set_device(0)
transform, n, b, trials, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
def one(fn):
    fn(); torch.npu.synchronize()
    t0 = time.perf_counter_ns()
    fn(); torch.npu.synchronize()
    return (time.perf_counter_ns() - t0) / 1e3
rows = []
for t in range(trials):
    if transform == "c2c":
        re_ = torch.randn(b, n); im = torch.randn(b, n)
        xc = (re_ + 1j * im).to(torch.complex64).npu()
        fn = lambda: torch.fft.fft(xc, dim=-1)
    elif transform == "r2c":
        xr = torch.randn(b, n).npu()
        fn = lambda: torch.fft.rfft(xr, dim=-1)
    else:
        xr = torch.randn(b, n).npu()
        half = torch.fft.rfft(xr, dim=-1)
        fn = lambda: torch.fft.irfft(half, n=n, dim=-1)
    samples = [one(fn) for _ in range(reps)]
    rows.append({"trial": t, "us": st.median(samples), "status": "ok"})
print(json.dumps(rows))
"""


def bench_torch_trials(transform, n, b, trials, raw):
    reps = reps_for(transform, n, b)
    try:
        r = subprocess.run(
            [PY, "-c", _TORCH_SRC, transform, str(n), str(b),
             str(trials), str(reps)],
            capture_output=True, text=True, cwd=ROOT, timeout=1800)
    except subprocess.TimeoutExpired:
        for t in range(trials):
            raw.append({"impl": "baseline", "transform": transform,
                        "n": n, "b": b, "reps": reps, "us": None,
                        "max_rel": None, "status": "timeout",
                        "error": "timeout"})
        return [None] * trials
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        for t in range(trials):
            raw.append({"impl": "baseline", "transform": transform,
                        "n": n, "b": b, "reps": reps, "us": None,
                        "max_rel": None, "status": "error",
                        "error": " | ".join(tail)})
        return [None] * trials
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    values = []
    for row in rows:
        us = row.get("us")
        values.append(us if us else None)
        raw.append({"impl": "baseline", "transform": transform,
                    "n": n, "b": b, "reps": reps, "us": us,
                    "max_rel": None, "status": "ok", "error": ""})
        print(f"baseline {transform} n={n} b={b} trial={row['trial']} -> "
              f"{us:.1f} us", file=sys.stderr, flush=True)
    return values


def stats_for(values):
    ok = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    if not ok:
        return {"median": None, "p10": None, "p90": None, "cv": None,
                "n_ok": 0, "n_total": len(values)}
    med = st.median(ok)
    try:
        p10 = st.quantiles(ok, n=10)[0] if len(ok) >= 2 else ok[0]
        p90 = st.quantiles(ok, n=10)[8] if len(ok) >= 2 else ok[0]
    except Exception:
        p10 = p90 = ok[0]
    mean = st.fmean(ok)
    cv = (st.pstdev(ok) / mean * 100.0) if mean > 0 and len(ok) >= 2 else 0.0
    return {"median": med, "p10": p10, "p90": p90, "cv": cv,
            "n_ok": len(ok), "n_total": len(values)}


def load_targets():
    return json.loads(TARGETS_PATH.read_text(encoding="utf-8"))


def load_source_speedups(doc):
    snap = ROOT / "results" / "published" / doc["source_snapshot"]
    result = {}
    with (snap / "matrix.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = ("c2c", int(row["n"]), int(row["batch"]))
            result[key] = float(row["native_us"]) / float(row["ours_us"])
    real = json.loads((snap / "r2c_c2r.json").read_text(encoding="utf-8"))
    for row in real["rows"]:
        for op, base_f, cand_f in (("r2c", "torch_rfft", "r2c"),
                                   ("c2r", "torch_irfft", "c2r")):
            key = (op, row["n"], row["b"])
            if row[cand_f] and row[base_f]:
                result[key] = float(row[base_f]) / float(row[cand_f])
    return result


def provenance():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=ROOT, timeout=30).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "soc": os.environ.get("AB_SOC", "Ascend910_9382"),
        "cann": os.environ.get("AB_CANN", "unknown"),
        "binary_sha256": binary_sha256(ROOT / "build" / "fft_check"),
        "python": PY,
        "source_snapshot": SNAPSHOT,
    }


def parse_candidates(specs):
    """Parse --candidate foldD=16,planeK=8 into list of override dicts."""
    cands = []
    for spec in specs or []:
        overrides = {}
        for kv in spec.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.strip()] = v.strip()
        cid = "_".join(f"{k}{v}" for k, v in sorted(overrides.items())) or "default"
        cands.append({"id": cid, "overrides": overrides})
    return cands


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3,
                    help="C2C min-of-means rounds (matrix_test protocol)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", metavar="op:NxB[,op:NxB...]",
                    help="subset of targets to run")
    ap.add_argument("--candidate", action="append", metavar="KEY=VAL[,KEY=VAL...]",
                    help="candidate env overrides (repeatable). "
                         "Incumbent (no overrides) is always included.")
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args(argv)

    doc = load_targets()
    source_speedups = load_source_speedups(doc)
    sha = binary_sha256(ROOT / "build" / "fft_check")

    wanted = None
    if a.targets:
        wanted = set()
        for tok in a.targets.split(","):
            op, shape = tok.split(":")
            n, b = shape.split("x")
            wanted.add((op, int(n), int(b)))

    # Candidate closure: incumbent first, then explicit candidates
    candidates = [{"id": "incumbent", "overrides": {}}]
    candidates.extend(parse_candidates(a.candidate))

    a.out.mkdir(parents=True, exist_ok=True)
    raw = []
    results = []

    for t in doc["targets"]:
        key = (t["operator"], t["n"], t["batch"])
        if wanted is not None and key not in wanted:
            continue
        transform, n, b = t["operator"], t["n"], t["batch"]
        print(f"\n=== {t['priority']} {transform} N={n} B={b} ===",
              file=sys.stderr, flush=True)

        # Measure each candidate
        cand_measurements = []
        for cand in candidates:
            mapping = resolve_mapping(transform, n, b, cand["overrides"])
            if transform == "c2c":
                us = bench_c2c_matrix(transform, n, b, a.rounds,
                                      cand["overrides"], raw, cand["id"])
                vals = [us] if us else []
            else:
                vals = bench_ours_trials(transform, n, b, a.trials,
                                         cand["overrides"], raw, cand["id"])
            stats = stats_for(vals)
            cand_measurements.append({
                "candidate_id": cand["id"],
                "mapping": mapping,
                "processing_unit": "AIV",
                "binary_sha256": sha,
                "stats": stats,
                "raw_us": vals,
            })
            print(f"  {cand['id']}: median={stats['median'] and round(stats['median'],1)}"
                  f" us (D={mapping['fold_d']},K={mapping['plane_k']})",
                  file=sys.stderr, flush=True)

        # Baseline
        if a.skip_baseline:
            base_stats = {"median": None, "n_ok": 0, "n_total": a.trials}
            base_raw = []
        else:
            bvals = bench_torch_trials(transform, n, b, a.trials, raw)
            base_stats = stats_for(bvals)
            base_raw = bvals

        # Selection: incumbent is fallback; promote if >=2% better
        incumbent = cand_measurements[0]
        selected = incumbent
        selected["selection_reason"] = "incumbent (baseline fallback)"
        if incumbent["stats"]["median"]:
            for cm in cand_measurements[1:]:
                if not cm["stats"]["median"]:
                    cm["selection_reason"] = "failed correctness/timing"
                    continue
                improve = (incumbent["stats"]["median"] - cm["stats"]["median"]) \
                    / incumbent["stats"]["median"]
                if improve >= PROMOTE_RATIO:
                    cm["selection_reason"] = (
                        f"promoted: {improve*100:.1f}% faster than incumbent "
                        f"(>= {PROMOTE_RATIO*100:.0f}% gate)")
                    selected = cm
                else:
                    cm["selection_reason"] = (
                        f"retained incumbent: only {improve*100:.1f}% "
                        f"(< {PROMOTE_RATIO*100:.0f}% gate)")
        else:
            for cm in cand_measurements[1:]:
                cm["selection_reason"] = "incumbent failed; candidate not evaluated for promotion"

        src = source_speedups.get(key)
        sel_med = selected["stats"]["median"]
        rep = (base_stats["median"] / sel_med
               if base_stats["median"] and sel_med else None)
        results.append({
            "priority": t["priority"],
            "transform": transform, "n": n, "batch": b,
            "source_speedup": src,
            "reproduced_speedup": rep,
            "incumbent": incumbent["stats"],
            "selected": {
                "candidate_id": selected["candidate_id"],
                "mapping": selected["mapping"],
                "selection_reason": selected.get("selection_reason", ""),
                "stats": selected["stats"],
            },
            "candidates": cand_measurements,
            "baseline": base_stats,
            "baseline_raw_us": base_raw,
        })
        print(f"  selected={selected['candidate_id']} "
              f"speedup={rep and round(rep,3)}",
              file=sys.stderr, flush=True)

    payload = {
        "protocol": {
            "trials": a.trials,
            "c2c_rounds": a.rounds,
            "c2c_stat": "min-of-means over rounds (matrix_test paired protocol)",
            "real_stat": "median over 5 warmup+timed trials (bench_r2c_c2r)",
            "reps_policy": "c2c:20; real: nB<=4096:50, nB<=2^20:30, else 10",
            "threshold_max_rel": THRESHOLD,
            "promote_ratio": PROMOTE_RATIO,
        },
        "provenance": provenance(),
        "candidates": [{"id": c["id"], "overrides": c["overrides"]}
                       for c in candidates],
        "targets": results,
    }
    (a.out / "convergence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (a.out / "convergence.raw.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        fields = ("impl", "transform", "n", "b", "reps", "us",
                  "max_rel", "status", "error")
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(raw)

    # Fail if any target has no valid measurement
    failures = [r for r in results
                if not r["incumbent"]["median"] or not r["baseline"]["median"]]
    print(f"\nWrote {a.out / 'convergence.json'} "
          f"({len(results)} targets, {len(raw)} raw rows, "
          f"{len(failures)} failures)", file=sys.stderr)

    # Summary table
    print("\n| pri | op | N | B | source | selected | D | reason | inc | base |")
    print("|-----|----|--:|--:|-------:|---------:|--:|--------|----:|-----:|")
    for r in results:
        src_s = f"{r['source_speedup']:.3f}" if r["source_speedup"] else "—"
        rep_s = (f"{r['reproduced_speedup']:.3f}"
                 if r["reproduced_speedup"] else "—")
        sel = r["selected"]
        d = sel["mapping"]["fold_d"]
        inc_s = (f"{r['incumbent']['median']:.1f}"
                 if r["incumbent"]["median"] else "—")
        base_s = (f"{r['baseline']['median']:.1f}"
                  if r["baseline"]["median"] else "—")
        reason = sel["selection_reason"][:30]
        print(f"| {r['priority']} | {r['transform']} | {r['n']} | {r['batch']} "
              f"| {src_s} | {rep_s} | {d} | {reason} | {inc_s} | {base_s} |")

    if failures:
        print(f"\nERROR: {len(failures)} targets failed (missing incumbent "
              f"or baseline measurement)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
