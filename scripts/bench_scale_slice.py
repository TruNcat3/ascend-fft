#!/usr/bin/env python3
"""A3 规模切片基准：N=8192..65536 的 C2C 正确性 + 计时（逐 trial raw）。

协议与 bench_r2c_c2r.py 的自研侧一致：
  * 每 trial 先做一次丢弃的 warmup fft_check，再一次 timed fft_check；
  * timed 的 maxRel ≤ 1e-4 才准入（PASS 行），失败/超时保留在 raw
    （status=error/timeout，计时字段 None）；
  * device_us 在每次 fft_check 内取 reps 次 device_chain 的最小值，再对 trials
    个有效样本汇总 median/p10/p90/CV；maxRel 取有效 trial 的最小值。

计时口径（n≥8192 走长链后端，逐次 launch 含段间传输，kfft_fwd 的
us/call 是宿主墙钟；见 src/host/fft_check.cpp scopes 行）：
  * device_us = device_chain 事件跨度（纯 device 计算，两 pass 内核和）——
    A3 "device" 列的口径；
  * wall_us   = kfft_fwd us/call（宿主墙钟，长链含逐次传输与宿主段）；
  * h2d_us / d2h_us = 同次 launch 的传输事件跨度（scopes 行，min 口径）。

输出（--out 指 JSON）：
  <out>                    {"protocol", "provenance", "rows"}
  <out 同名>.raw_trials.{json,csv}   逐 trial 原始行
  <out 目录>/scale_slice.csv         汇总（含 device/wall/maxRel/ok）
stdout 打印 PASS 计数与几何均值；全部 ok 才返回 0。
"""
import argparse, csv, json, math, os, re, statistics as st, subprocess, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_r2c_c2r import provenance  # noqa: E402

DEFAULT_NS = "8192,16384,32768,65536"
DEFAULT_BS = "1,16,256"
THRESHOLD = 1e-4
RAW_FIELDS = ("impl", "n", "b", "trial", "reps", "device_us", "wall_us",
              "h2d_us", "d2h_us", "med_us", "max_abs", "max_rel",
              "status", "error")


def reps_for(n, b):
    pts = n * b
    if pts <= 4096:
        return 50
    if pts <= 1 << 20:
        return 30
    return 10


def pctl(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return {"median": None, "p10": None, "p90": None, "cv": None,
                "trials_ok": 0, "trials_total": len(values)}
    mean = st.fmean(valid)
    cv = (st.stdev(valid) / mean) if (len(valid) >= 2 and mean) else 0.0
    return {"median": st.median(valid), "p10": pctl(valid, 0.10),
            "p90": pctl(valid, 0.90), "cv": cv,
            "trials_ok": len(valid), "trials_total": len(values)}


def once(n, b, reps):
    """一次 fft_check：返回计时字典 + (max_abs, max_rel, err, status)。"""
    blank = {"device_us": None, "wall_us": None, "h2d_us": None,
             "d2h_us": None, "med_us": None}
    try:
        r = subprocess.run(["./build/fft_check", str(n), str(b), str(reps)],
                           capture_output=True, text=True, cwd=ROOT,
                           timeout=3600)
    except subprocess.TimeoutExpired:
        return blank, None, None, "timeout after 3600 s", "timeout"
    out = r.stdout + r.stderr
    m = re.search(r"med ([\d.]+) us", out)
    w = re.search(r"kfft_fwd: ([\d.]+) us/call", out)
    e = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    sc_line = next((ln for ln in out.splitlines() if ln.startswith("scopes:")),
                   "")

    def span(field):
        m2 = re.search(rf"{field}=([\d.]+|NA)(?: us)?", sc_line)
        if not m2 or m2.group(1) == "NA":
            return None
        return float(m2.group(1))

    timings = {"device_us": span("device_chain"), "h2d_us": span("h2d"),
               "d2h_us": span("d2h"),
               "wall_us": float(w.group(1)) if w else None,
               "med_us": float(m.group(1)) if m else None}
    max_abs = float(e.group(1)) if e else None
    max_rel = float(e.group(2)) if e else None
    if r.returncode != 0 or not re.search(r"^PASS$", out, re.M):
        tail = " | ".join(out.strip().splitlines()[-3:])
        return timings, max_abs, max_rel, (f"rc={r.returncode}: {tail}" if tail
                                           else f"rc={r.returncode}"), "error"
    if not m:
        return timings, max_abs, max_rel, "no med field in output", "error"
    if max_rel is not None and max_rel > THRESHOLD:
        return timings, max_abs, max_rel, f"maxRel {max_rel} > {THRESHOLD}", "error"
    return timings, max_abs, max_rel, "", "ok"


def main():
    started_utc = datetime.now(timezone.utc).isoformat()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default=DEFAULT_NS)
    ap.add_argument("--bs", default=DEFAULT_BS)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", required=True,
                    help="汇总 JSON 路径（同目录写 scale_slice.csv）")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    protocol = {"trials": a.trials,
                "warmup": "one discarded full-reps execution inside every trial",
                "statistic": "device_us is the minimum device_chain span within "
                             "each trial; summary reports median/p10/p90/CV "
                             "over valid trial minima only; wall_us is the "
                             "per-trial mean host wall time",
                "reps_policy": "nB<=4096:50, nB<=2^20:30, else 10",
                "correctness_threshold": THRESHOLD,
                "raw_rows": "one row per (N,B,trial) incl. failed/"
                            "timeout with status+error"}
    prov = provenance()
    prov["command"] = " ".join(sys.argv)
    prov["started_utc"] = started_utc

    raw, rows = [], []
    for n in ns:
        for b in bs:
            reps = reps_for(n, b)
            dev_vals, wall_vals = [], []
            for t in range(a.trials):
                once(n, b, reps)  # trial 内 warmup，丢弃
                timings, max_abs, max_rel, err, status = once(n, b, reps)
                dev_vals.append(timings["device_us"] if status == "ok" else None)
                wall_vals.append(timings["wall_us"] if status == "ok" else None)
                raw.append(dict(impl="ours_c2c", n=n, b=b, trial=t, reps=reps,
                                device_us=timings["device_us"],
                                wall_us=timings["wall_us"],
                                h2d_us=timings["h2d_us"],
                                d2h_us=timings["d2h_us"],
                                med_us=timings["med_us"],
                                max_abs=max_abs, max_rel=max_rel,
                                status=status, error=err))
                print(f"c2c n={n} b={b} trial={t} -> "
                      + (f"device={timings['device_us']} us "
                         f"wall={timings['wall_us']} us (maxRel={max_rel})"
                         if status == "ok"
                         else f"{status.upper()}: {err}"),
                      file=sys.stderr, flush=True)
            s = summarize(dev_vals)
            s_wall = summarize(wall_vals)
            rels = [r["max_rel"] for r in raw
                    if r["n"] == n and r["b"] == b and r["status"] == "ok"
                    and r["max_rel"] is not None]
            row = dict(n=n, batch=b, device_us=s["median"],
                       wall_us=s_wall["median"], stats=s,
                       stats_wall=s_wall,
                       maxRel=min(rels) if rels else None,
                       trials_ok=s["trials_ok"], trials_total=a.trials)
            row["ok"] = s["trials_ok"] == a.trials
            rows.append(row)

    prov["ended_utc"] = datetime.now(timezone.utc).isoformat()
    document = {"protocol": protocol, "provenance": prov, "rows": rows}
    dst = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    out_dir = os.path.dirname(dst)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump(document, open(dst, "w"), indent=1)
    base, _ = os.path.splitext(dst)
    json.dump({"protocol": protocol, "provenance": prov, "raw_trials": raw},
              open(base + ".raw_trials.json", "w"), indent=1)
    with open(base + ".raw_trials.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        w.writeheader()
        for rec in raw:
            w.writerow(rec)
    summary_csv = os.path.join(out_dir or ".", "scale_slice.csv")
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "batch", "device_us", "wall_us", "maxRel", "ok"])
        for r in rows:
            w.writerow([r["n"], r["batch"],
                        f"{r['device_us']:.1f}" if r["device_us"] else "",
                        f"{r['wall_us']:.1f}" if r["wall_us"] else "",
                        r["maxRel"] if r["maxRel"] is not None else "",
                        "PASS" if r["ok"] else "FAIL"])

    ok = sum(1 for r in rows if r["ok"])
    print(f"PASS {ok}/{len(rows)}", file=sys.stderr)
    vals = [r["device_us"] for r in rows if r["device_us"]]
    if vals:
        print("geo device      %8.2f us"
              % math.exp(sum(map(math.log, vals)) / len(vals)))
    print(f"wrote {dst} (+ .raw_trials.json/.raw_trials.csv, {summary_csv})",
          file=sys.stderr)
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
