#!/usr/bin/env python3
"""Targeted convergence benchmark for the 14 frozen sub-unity points (PR #6).

Reads ``config/convergence_targets.json`` and, for every target, measures
under the *source snapshot's* timing protocol:

* C2C: ``fft_check <n> <b> 20`` mean, 3 rounds, min-of-means (matrix_test).
* R2C/C2R: ``fft_check <n> <b> <reps>`` with the symmetric 5-trial
  warmup+timed protocol from ``bench_r2c_c2r`` (reps by n*B policy).

Every target also measures the matched external baseline (torch_npu
``fft`` / ``rfft`` / ``irfft``) with the same trial/warmup/reps contract
so the speedup ratio is protocol-matched.  The candidate set always
contains the incumbent (current deployed configuration); this runner
records the incumbent measurement only — candidate search is a separate
step that reuses this protocol.

Output (JSON + raw trial CSVs):
  {
    "protocol": {...},
    "provenance": {...},
    "targets": [
      {"transform","n","batch","priority",
       "incumbent":{"us","correct","max_rel","raw":[...]},
       "baseline":{"us","correct","raw":[...]},
       "source_speedup","reproduced_speedup", ...}
    ]
  }

  python3 scripts/bench_convergence.py --targets-only \
      --out results/runs/<UTC>-convergence-repro
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


# ---- protocol helpers (mirror matrix_test / bench_r2c_c2r) ---------------
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


def run_ours(transform, n, b, reps, trial, raw, tag="incumbent"):
    """One fft_check invocation; returns (us, max_rel, status, error)."""
    env = dict(os.environ)
    if transform != "c2c":
        env["AB_DIR"] = transform
    try:
        r = subprocess.run(
            ["./build/fft_check", str(n), str(b), str(reps)],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    except subprocess.TimeoutExpired:
        raw.append(_raw_row(tag, transform, n, b, trial, reps, None,
                            None, "timeout", "timeout after 1800 s"))
        return None, None, "timeout", "timeout"
    out = r.stdout + r.stderr
    m = re.search(r"med ([\d.]+) us", out)
    e = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    max_rel = float(e.group(2)) if e else None
    ok = bool(re.search(r"^PASS$", out, re.M))
    if r.returncode != 0 or not ok:
        tail = " | ".join(out.strip().splitlines()[-3:])
        status = "timeout" if "timeout" in (tail or "") else "error"
        err = f"rc={r.returncode}: {tail}" if tail else f"rc={r.returncode}"
        raw.append(_raw_row(tag, transform, n, b, trial, reps, None,
                            max_rel, status, err))
        return None, max_rel, status, err
    if not m:
        raw.append(_raw_row(tag, transform, n, b, trial, reps, None,
                            max_rel, "error", "no med field"))
        return None, max_rel, "error", "no med field"
    if max_rel is not None and max_rel > THRESHOLD:
        raw.append(_raw_row(tag, transform, n, b, trial, reps, None,
                            max_rel, "error", f"maxRel {max_rel} > {THRESHOLD}"))
        return None, max_rel, "error", f"maxRel {max_rel}"
    us = float(m.group(1))
    raw.append(_raw_row(tag, transform, n, b, trial, reps, us,
                        max_rel, "ok", ""))
    return us, max_rel, "ok", ""


def _raw_row(tag, transform, n, b, trial, reps, us, max_rel, status, error):
    return {"impl": tag, "transform": transform, "n": n, "b": b,
            "trial": trial, "reps": reps, "us": us, "max_rel": max_rel,
            "status": status, "error": error}


def bench_ours_trials(transform, n, b, trials, raw):
    """trials of (warmup discard + timed) for the incumbent path."""
    reps = reps_for(transform, n, b)
    values = []
    for trial in range(trials):
        run_ours(transform, n, b, reps, trial, raw, tag="incumbent_warmup")
        # the warmup row is kept in raw for audit but not used in stats
        us, max_rel, status, err = run_ours(
            transform, n, b, reps, trial, raw, tag="incumbent")
        values.append(us if status == "ok" else None)
        print(f"incumbent {transform} n={n} b={b} trial={trial} -> "
              + (f"{us:.1f} us (maxRel={max_rel})" if status == "ok"
                 else f"{status.upper()}: {err}"), file=sys.stderr, flush=True)
    return values


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
        re = torch.randn(b, n)
        im = torch.randn(b, n)
        xc = (re + 1j * im).to(torch.complex64).npu()
        fn = lambda: torch.fft.fft(xc, dim=-1)
    elif transform == "r2c":
        xr = torch.randn(b, n).npu()
        fn = lambda: torch.fft.rfft(xr, dim=-1)
    else:  # c2r
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
            raw.append(_raw_row("baseline", transform, n, b, t, reps,
                                None, None, "timeout", "timeout"))
        return [None] * trials
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        for t in range(trials):
            raw.append(_raw_row("baseline", transform, n, b, t, reps,
                                None, None, "error", " | ".join(tail)))
        return [None] * trials
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    values = []
    for row in rows:
        us = row.get("us")
        status = row.get("status", "ok")
        values.append(us if status == "ok" and us else None)
        raw.append(_raw_row("baseline", transform, n, b, row["trial"], reps,
                            us, None, status, ""))
        print(f"baseline {transform} n={n} b={b} trial={row['trial']} -> "
              f"{us:.1f} us", file=sys.stderr, flush=True)
    return values


def stats_for(values):
    ok = [v for v in values if v is not None and math.isfinite(v) and v > 0]
    if not ok:
        return {"median": None, "p10": None, "p90": None, "cv": None,
                "n_ok": 0, "n_total": len(values)}
    med = st.median(ok)
    p10 = st.quantiles(ok, n=10)[0] if len(ok) >= 2 else ok[0]
    p90 = st.quantiles(ok, n=10)[8] if len(ok) >= 2 else ok[0]
    mean = st.fmean(ok)
    cv = (st.pstdev(ok) / mean * 100.0) if mean > 0 and len(ok) >= 2 else 0.0
    return {"median": med, "p10": p10, "p90": p90, "cv": cv,
            "n_ok": len(ok), "n_total": len(values)}


def load_targets():
    doc = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    return doc


def load_source_speedups(doc):
    """speedup from the frozen manifest's recorded targets (if present)."""
    # the check script derives them; recompute from the snapshot for audit
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
        "python": PY,
        "source_snapshot": SNAPSHOT,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3,
                    help="C2C min-of-means rounds (matrix_test protocol)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", metavar="op:NxB[,op:NxB...]",
                    help="subset of targets to run")
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args(argv)

    doc = load_targets()
    source_speedups = load_source_speedups(doc)
    wanted = None
    if a.only:
        wanted = set()
        for tok in a.only.split(","):
            op, shape = tok.split(":")
            n, b = shape.split("x")
            wanted.add((op, int(n), int(b)))

    a.out.mkdir(parents=True, exist_ok=True)
    raw = []
    results = []
    for t in doc["targets"]:
        key = (t["operator"], t["n"], t["batch"])
        if wanted is not None and key not in wanted:
            continue
        transform = t["operator"]
        n, b = t["n"], t["batch"]
        print(f"=== {t['priority']} {transform} N={n} B={b} ===",
              file=sys.stderr, flush=True)
        if transform == "c2c":
            # matrix_test protocol: rounds of full measurement, min-of-means
            round_means = []
            for rnd in range(a.rounds):
                vals = bench_ours_trials(transform, n, b, a.trials, raw)
                ok = [v for v in vals if v]
                if ok:
                    round_means.append(st.fmean(ok))
            inc = stats_for([min(round_means)] if round_means else [])
            # also keep per-trial values for raw audit
            inc_all_ok = [v for v in raw
                          if v["impl"] == "incumbent" and v["status"] == "ok"
                          and v["n"] == n and v["b"] == b]
            inc_raw_us = [v["us"] for v in inc_all_ok]
        else:
            vals = bench_ours_trials(transform, n, b, a.trials, raw)
            inc = stats_for(vals)
            inc_raw_us = vals

        if a.skip_baseline:
            base = {"median": None, "p10": None, "p90": None, "cv": None,
                    "n_ok": 0, "n_total": a.trials}
            base_raw_us = []
        else:
            bvals = bench_torch_trials(transform, n, b, a.trials, raw)
            base = stats_for(bvals)
            base_raw_us = bvals

        src = source_speedups.get(key)
        rep = (base["median"] / inc["median"]
               if base["median"] and inc["median"] else None)
        results.append({
            "priority": t["priority"],
            "transform": transform, "n": n, "batch": b,
            "source_speedup": src,
            "reproduced_speedup": rep,
            "incumbent": inc,
            "incumbent_raw_us": inc_raw_us,
            "baseline": base,
            "baseline_raw_us": base_raw_us,
            "drift": (None if not (src and rep) else rep / src - 1.0),
        })
        print(f"  source={src and round(src,3)} reproduced={rep and round(rep,3)} "
              f"drift={results[-1]['drift'] and round(results[-1]['drift']*100,1)}%",
              file=sys.stderr, flush=True)

    payload = {
        "protocol": {
            "trials": a.trials,
            "c2c_rounds": a.rounds,
            "c2c_stat": "min-of-means over rounds of trial-means (matrix_test)",
            "real_stat": "median over 5 warmup+timed trials (bench_r2c_c2r)",
            "reps_policy": "c2c:20; real: nB<=4096:50, nB<=2^20:30, else 10",
            "threshold_max_rel": THRESHOLD,
        },
        "provenance": provenance(),
        "targets": results,
    }
    (a.out / "convergence_repro.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (a.out / "convergence_repro.raw.csv").open("w", newline="",
                                                    encoding="utf-8") as fh:
        fields = ("impl", "transform", "n", "b", "trial", "reps", "us",
                  "max_rel", "status", "error")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(raw)
    print(f"\nWrote {a.out / 'convergence_repro.json'} "
          f"({len(results)} targets, {len(raw)} raw rows)",
          file=sys.stderr)
    # summary table
    print("\n| pri | op | N | B | source | reproduced | drift | inc_med | base_med |")
    print("|-----|----|--:|--:|-------:|-----------:|------:|--------:|---------:|")
    for r in results:
        src_s = f"{r['source_speedup']:.3f}" if r["source_speedup"] else "—"
        rep_s = f"{r['reproduced_speedup']:.3f}" if r["reproduced_speedup"] else "—"
        drift_s = f"{r['drift']*100:+.1f}%" if r["drift"] is not None else "—"
        inc_s = f"{r['incumbent']['median']:.1f}" if r["incumbent"]["median"] else "—"
        base_s = f"{r['baseline']['median']:.1f}" if r["baseline"]["median"] else "—"
        print(f"| {r['priority']} | {r['transform']} | {r['n']} | {r['batch']} "
              f"| {src_s} | {rep_s} | {drift_s} | {inc_s} | {base_s} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
