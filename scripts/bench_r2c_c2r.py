#!/usr/bin/env python3
"""r2c/c2r 基准（P1-B 对称协议）：自研 kfft_r2c/kfft_c2r vs torch.fft.rfft/irfft（torch_npu）。

对称协议（两种实现同一口径，PR #1 P1-B 发布契约）：
  * 5 个独立 trial；**每个 trial 先做一次丢弃的 warmup 执行，再做 timed reps**；
  * reps 政策：n·B<=4096:50、<=2^20:30、否则 10，逐 (方向,N,B) 记录进 raw 行；
  * trial 值 = 该 trial timed reps 样本的 **median**；
    - 自研：warmup 一次 `fft_check <n> <b> <reps>`（丢弃），timed 一次取其
      `med`，并把该次的 `maxAbs`/`maxRel` 记入行（PASS 才准入 timing）；
    - torch：warmup 一轮（丢弃）+ timed 一轮，样本 + torch.npu.synchronize；
  * 汇总 = 5 个 trial 的 median / p10 / p90 / CV，只统计 status=ok 的 trial；
  * 失败、超时、不支持的行全部保留在 raw（status=error/timeout/unsupported，
    us=None）；汇总只由有效 raw 行派生；
  * 附 provenance（硬件 / CANN / torch / 编译器 / git 修订与 dirty）与逐 trial
    raw_trials.{json,csv}。

aclRfft1D 列沿用 results/e2e.json 的 bare_dev（单点来源，不做 trial 化）。

  python3 scripts/bench_r2c_c2r.py --trials 5 --warmup-per-trial \
      --ns 64,...,8192 --bs 1,...,4096 --out results/runs/<UTC>-r2c-c2r/r2c_c2r.json

输出 JSON：{"protocol":..., "provenance":..., "rows":[...]};
raw 逐 trial 记录写到 <out>.raw_trials.json 与 <out>.raw_trials.csv。
比值一律 >1 表示自研更快；stdout 打印几何均值汇总，进度到 stderr。
"""
import argparse, csv, json, math, os, re, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abenv  # noqa: E402  与 scripts/env.sh 共用同一套路径探测
PY = abenv.python_bin()

NS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
BS = [1, 4, 16, 64, 256, 1024, 4096]
TRIALS = 5
REPS_POLICY = "nB<=4096:50, nB<=2^20:30, else 10"
THRESHOLD = 1e-4

RAW_FIELDS = ("impl", "n", "b", "trial", "reps", "us", "max_abs", "max_rel",
              "status", "error")


def reps_for(n, b):
    pts = n * b
    if pts <= 4096:
        return 50
    if pts <= 1 << 20:
        return 30
    return 10


def supported(dirn, n):
    if dirn == "r2c":
        return 128 <= n <= 8192
    return 64 <= n <= 4096


def pctl(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
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


def bench_ours_trials(dirn, n, b, trials, raw, impl):
    """每 trial：一次丢弃的 warmup fft_check + 一次 timed fft_check（取 med）。"""
    reps = reps_for(n, b)
    env = dict(os.environ, AB_DIR=dirn)
    values = []

    def once():
        try:
            r = subprocess.run(
                ["./build/fft_check", str(n), str(b), str(reps)],
                capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
        except subprocess.TimeoutExpired:
            return None, None, None, "timeout after 1800 s", "timeout"
        out = r.stdout + r.stderr
        m = re.search(r"med ([\d.]+) us", out)
        e = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
        max_abs = float(e.group(1)) if e else None
        max_rel = float(e.group(2)) if e else None
        ok = bool(re.search(r"^PASS$", out, re.M))
        if r.returncode != 0 or not ok:
            tail = " | ".join(out.strip().splitlines()[-3:])
            status = "timeout" if "timeout" in (tail or "") else "error"
            return None, max_abs, max_rel, (f"rc={r.returncode}: {tail}" if tail
                                            else f"rc={r.returncode}"), status
        if not m:
            return None, max_abs, max_rel, "no med field in output", "error"
        if max_rel is not None and max_rel > THRESHOLD:
            return None, max_abs, max_rel, f"maxRel {max_rel} > {THRESHOLD}", "error"
        return float(m.group(1)), max_abs, max_rel, "", "ok"

    for trial in range(trials):
        once()  # trial 内 warmup，丢弃
        us, max_abs, max_rel, err, status = once()
        values.append(us if status == "ok" else None)
        raw.append(dict(impl=impl, n=n, b=b, trial=trial, reps=reps, us=us,
                        max_abs=max_abs, max_rel=max_rel, status=status,
                        error=err))
        print(f"{impl} n={n} b={b} trial={trial} -> "
              + (f"{us:.1f} us (maxRel={max_rel})" if status == "ok"
                 else f"{status.upper()}: {err}"), file=sys.stderr, flush=True)
    return values


_TORCH_SRC = r"""
import json, statistics as st, sys, time, warnings
warnings.filterwarnings("ignore")
import torch, torch_npu
ns = [int(x) for x in sys.argv[1].split(",")]
bs = [int(x) for x in sys.argv[2].split(",")]
trials = int(sys.argv[3])
def reps_for(n, b):
    p = n * b
    return 50 if p <= 4096 else (30 if p <= 1 << 20 else 10)
def one_round(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return st.median(ts)
raw = []
for n in ns:
    x = torch.randn(1, n, dtype=torch.float32).npu()
    for b in bs:
        try:
            xc = x.expand(b, n).contiguous()
            half = torch.fft.rfft(xc)
            r = reps_for(n, b)
            for t in range(trials):
                for impl, fn in (("torch_rfft", lambda: torch.fft.rfft(xc)),
                                 ("torch_irfft",
                                  lambda: torch.fft.irfft(half, n=n))):
                    try:
                        one_round(fn, r)  # trial 内 warmup，丢弃
                        us = one_round(fn, r)
                        raw.append(dict(impl=impl, n=n, b=b, trial=t, reps=r,
                                        us=us, max_abs=None, max_rel=None,
                                        status="ok", error=""))
                    except Exception as exc:
                        raw.append(dict(impl=impl, n=n, b=b, trial=t, reps=r,
                                        us=None, max_abs=None, max_rel=None,
                                        status="error",
                                        error=str(exc).replace("\n", " ")[:300]))
            print(f"torch n={n} b={b}", file=sys.stderr, flush=True)
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:300]
            for impl in ("torch_rfft", "torch_irfft"):
                for t in range(trials):
                    raw.append(dict(impl=impl, n=n, b=b, trial=t,
                                    reps=reps_for(n, b), us=None,
                                    max_abs=None, max_rel=None,
                                    status="error", error=msg))
json.dump(raw, sys.stdout)
"""


def bench_torch(ns, bs, trials):
    r = subprocess.run([PY, "-c", _TORCH_SRC, ",".join(map(str, ns)),
                        ",".join(map(str, bs)), str(trials)],
                       capture_output=True, text=True, cwd=ROOT, timeout=7200)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(r.stderr[-4000:])
        raise


def load_acl():
    """results/e2e.json 各点的 aclRfft1D device 时间（bare_dev），缺文件则空。"""
    path = os.path.join(ROOT, "results", "e2e.json")
    if not os.path.exists(path):
        return {}
    try:
        rows = json.load(open(path)).get("rows", [])
        return {(d["n"], d["batch"]): d.get("bare_dev") for d in rows
                if d.get("bare_dev")}
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def detect_cann_version():
    env = os.environ.get("AB_CANN_VERSION")
    if env:
        return env
    target = os.path.realpath("/usr/local/Ascend/ascend-toolkit/latest")
    name = os.path.basename(target)
    if name.startswith("cann-"):
        return name[len("cann-"):]
    if os.path.isdir(target):
        return name
    return "unknown"


def provenance():
    def run(cmd, cwd=ROOT):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                               timeout=30)
            return r.stdout.strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    sha = run(["git", "rev-parse", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    torch_ver = run([PY, "-c",
                     "import torch,torch_npu;"
                     "print(torch.__version__)"]) or ""
    npu_name = run([PY, "-c",
                    "import torch;"
                    "print(torch.npu.get_device_name(0))"]) or ""
    cann = detect_cann_version()
    smi = run(["npu-smi", "info"])[:500]
    compiler = (run(["g++", "--version"]).splitlines() or [""])[0]
    return {"hardware": npu_name or "unknown", "npu_smi": smi,
            "cann": cann or "unknown", "torch": torch_ver or "unknown",
            "torch_npu": run([PY, "-c",
                              "import torch_npu;"
                              "print(getattr(torch_npu,'__version__','unknown'))"])
            or "unknown",
            "compiler": compiler or "unknown",
            "python": sys.version.split()[0],
            "git_sha": sha, "git_dirty": dirty}


def geo(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--bs", default=",".join(map(str, BS)))
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--rounds", type=int, default=None,
                    help="deprecated alias for --trials")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.rounds is not None:
        if a.trials != TRIALS:
            ap.error("pass either --trials or --rounds, not both")
        a.trials = a.rounds
    if a.trials < 1:
        ap.error("--trials must be >=1")
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    protocol = {"trials": a.trials,
                "warmup": "one discarded full-reps execution inside every trial",
                "statistic": "per-trial median; summary (median/p10/p90/CV) over "
                             "valid raw rows only",
                "reps_policy": REPS_POLICY, "correctness_threshold": THRESHOLD,
                "raw_rows": "one row per (impl,N,B,trial) incl. failed/"
                            "unsupported/timeout with status+error"}
    prov = provenance()
    raw = []

    print(f"[1/3] torch rfft/irfft ({len(ns)}x{len(bs)} x {a.trials}) ...",
          file=sys.stderr)
    raw += bench_torch(ns, bs, a.trials)
    acl = load_acl()

    print(f"[2/3] 自研 kfft_r2c/kfft_c2r ({len(ns)}x{len(bs)} x {a.trials}) ...",
          file=sys.stderr)
    rows, fails = [], []
    for n in ns:
        for b in bs:
            for dirn, impl in (("r2c", "ours_r2c"), ("c2r", "ours_c2r")):
                if not supported(dirn, n):
                    raw.append(dict(impl=impl, n=n, b=b, trial=0,
                                    reps=reps_for(n, b), us=None,
                                    max_abs=None, max_rel=None,
                                    status="unsupported",
                                    error=f"{dirn} unsupported at N={n}"))
                    continue
                vals = bench_ours_trials(dirn, n, b, a.trials, raw, impl)
                if any(v is None for v in vals):
                    fails.append((dirn, n, b))

            def stats_for(impl):
                return summarize([r["us"] for r in raw
                                  if r["impl"] == impl and r["n"] == n
                                  and r["b"] == b
                                  and r["status"] != "unsupported"])

            s_r2c, s_c2r = stats_for("ours_r2c"), stats_for("ours_c2r")
            s_tr, s_ti = stats_for("torch_rfft"), stats_for("torch_irfft")
            acl_us = acl.get((n, b))
            def flag(s):
                if s["trials_total"] == 0:
                    return None
                return s["trials_ok"] == a.trials

            row = dict(n=n, b=b,
                       r2c=s_r2c["median"], c2r=s_c2r["median"],
                       torch_rfft=s_tr["median"], torch_irfft=s_ti["median"],
                       r2c_ok=flag(s_r2c), c2r_ok=flag(s_c2r),
                       acl_rfft=acl_us,
                       r2c_stats=s_r2c, c2r_stats=s_c2r,
                       torch_rfft_stats=s_tr, torch_irfft_stats=s_ti,
                       r2c_vs_torch=(s_tr["median"] / s_r2c["median"])
                       if s_r2c["median"] and s_tr["median"] else None,
                       r2c_vs_acl=(acl_us / s_r2c["median"])
                       if s_r2c["median"] and acl_us else None,
                       c2r_vs_torch=(s_ti["median"] / s_c2r["median"])
                       if s_c2r["median"] and s_ti["median"] else None)
            ours_flags = [f for f in (row["r2c_ok"], row["c2r_ok"])
                          if f is not None]
            row["ok"] = bool(ours_flags) and all(ours_flags)
            rows.append(row)
            print(f"n={n:<5} b={b:<5} r2c={row['r2c']} c2r={row['c2r']}",
                  file=sys.stderr, flush=True)

    print("[3/3] 汇总与 raw trials ...", file=sys.stderr)
    document = {"protocol": protocol, "provenance": prov, "rows": rows}
    if a.out:
        dst = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        out_dir = os.path.dirname(dst)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        json.dump(document, open(dst, "w"), indent=1)
        base, _ = os.path.splitext(dst)
        json.dump({"protocol": protocol, "provenance": prov,
                   "raw_trials": raw}, open(base + ".raw_trials.json", "w"),
                  indent=1)
        with open(base + ".raw_trials.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for rec in raw:
                writer.writerow(rec)
        print(f"wrote {dst} (+ .raw_trials.json/.raw_trials.csv)", file=sys.stderr)
    if fails:
        print("FAILS:", fails)
        return 1

    common = [n for n in ns if n >= 128]
    print("geo ours r2c      %8.2f us (n>=128)" % geo([d["r2c"] for d in rows if d["r2c"]]))
    print("geo ours c2r      %8.2f us" % geo([d["c2r"] for d in rows if d["c2r"]]))
    print("geo torch rfft    %8.2f us (same pts)"
          % geo([d["torch_rfft"] for d in rows if d["n"] in common]))
    print("geo torch irfft   %8.2f us (paired points)"
          % geo([d["torch_irfft"] for d in rows if d["c2r"] and d["torch_irfft"]]))
    print("geo aclRfft1D     %8.2f us (paired points)"
          % geo([d["acl_rfft"] for d in rows if d["r2c"] and d["acl_rfft"]]))
    g_ours_r2c = geo([d["r2c"] for d in rows if d["r2c"] and d["n"] in common])
    g_torch_r2c = geo([d["torch_rfft"] for d in rows if d["n"] in common])
    print("r2c vs torch rfft : %5.2fx faster" % (g_torch_r2c / g_ours_r2c))
    paired_acl = [d for d in rows if d["r2c"] and d["acl_rfft"]]
    if paired_acl:
        print("r2c vs aclRfft1D  : %5.2fx faster"
              % (geo([d["acl_rfft"] for d in paired_acl])
                 / geo([d["r2c"] for d in paired_acl])))
    paired_c2r = [d for d in rows if d["c2r"] and d["torch_irfft"]]
    print("c2r vs torch irfft: %5.2fx faster"
          % (geo([d["torch_irfft"] for d in paired_c2r])
             / geo([d["c2r"] for d in paired_c2r])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
