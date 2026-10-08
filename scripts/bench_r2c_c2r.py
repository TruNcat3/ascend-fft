#!/usr/bin/env python3
"""r2c/c2r 基准：自研 kfft_r2c/kfft_c2r vs torch.fft.rfft/irfft（torch_npu）vs aclRfft1D。

口径与 scripts/matrix_test.py、bench_stdlib.py 一致：
  * 自研 —— `fft_check <n> <b> <reps>`（AB_DIR=r2c|c2r），整链 device 时间取 **min**；
  * torch —— 同 reps 政策（n·B<=4096:50、<=2^20:30、否则 10），每轮 median 取跨轮 min，
    torch.npu.synchronize 计时；
  * aclRfft1D —— 复用 results/e2e.json 各点的 bare_dev（无则该列留空）。

  python3 scripts/bench_r2c_c2r.py --ns 64,...,4096 --bs 1,...,4096 --out results/r2c_c2r.json

输出 JSON 行：
  {n, b, r2c, c2r, r2c_ok, c2r_ok, ok, torch_rfft, torch_irfft,
   acl_rfft, r2c_vs_torch, r2c_vs_acl, c2r_vs_torch}
比值一律 >1 表示自研更快；stdout 打印几何均值汇总，进度到 stderr。
"""
import argparse, json, math, os, re, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abenv  # noqa: E402  与 scripts/env.sh 共用同一套路径探测
PY = abenv.python_bin()

NS = [64, 128, 256, 512, 1024, 2048, 4096]
BS = [1, 4, 16, 64, 256, 1024, 4096]
ROUNDS = 3


def reps_for(n, b):
    pts = n * b
    if pts <= 4096:
        return 50
    if pts <= 1 << 20:
        return 30
    return 10


def bench_ours(dirn, n, b):
    """fft_check 整链 device 时间（min），并校验 PASS。"""
    env = dict(os.environ, AB_DIR=dirn)
    r = subprocess.run(["./build/fft_check", str(n), str(b), str(reps_for(n, b))],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=3600)
    out = r.stdout + r.stderr
    m = re.search(r"min ([\d.]+) us\)", out)
    return (float(m.group(1)) if m else None), bool(re.search(r"^PASS$", out, re.M))


_TORCH_SRC = r"""
import json, statistics as st, sys, time, warnings
warnings.filterwarnings("ignore")
import torch, torch_npu
ns = [int(x) for x in sys.argv[1].split(",")]
bs = [int(x) for x in sys.argv[2].split(",")]
rounds = int(sys.argv[3])
def reps_for(n, b):
    p = n * b
    return 50 if p <= 4096 else (30 if p <= 1 << 20 else 10)
def bench(fn, reps):
    best = float("inf")
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e6)
        best = min(best, st.median(ts))
    return best
rows = []
for n in ns:
    x = torch.randn(1, n, dtype=torch.float32).npu()
    for b in bs:
        xc = x.expand(b, n).contiguous()
        half = torch.fft.rfft(xc)
        r = reps_for(n, b)
        rows.append(dict(n=n, b=b,
                         r2c=bench(lambda: torch.fft.rfft(xc), r),
                         c2r=bench(lambda: torch.fft.irfft(half, n=n), r)))
        print(f"torch n={n} b={b}", file=sys.stderr, flush=True)
json.dump(rows, sys.stdout)
"""


def bench_torch(ns, bs, rounds):
    """子进程跑 torch_npu 计时（与自研同 reps 政策），返回 {(n,b): row}。"""
    r = subprocess.run([PY, "-c", _TORCH_SRC, ",".join(map(str, ns)),
                        ",".join(map(str, bs)), str(rounds)],
                       capture_output=True, text=True, cwd=ROOT, timeout=3600)
    try:
        rows = json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(r.stderr[-4000:])
        raise
    return {(d["n"], d["b"]): d for d in rows}


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


def geo(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--bs", default=",".join(map(str, BS)))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]

    print(f"[1/2] torch rfft/irfft ({len(ns)}x{len(bs)} x {a.rounds}) ...", file=sys.stderr)
    torch_rows = bench_torch(ns, bs, a.rounds)
    acl = load_acl()

    print(f"[2/2] 自研 kfft_r2c/kfft_c2r ({len(ns)}x{len(bs)}) ...", file=sys.stderr)
    rows, fails = [], []
    for n in ns:
        for b in bs:
            ours, ours_ok, t = {}, {}, {}
            for dirn in ("r2c", "c2r"):
                if dirn == "r2c" and (n < 128 or n > 8192):
                    continue
                if dirn == "c2r" and (n < 64 or n > 4096):
                    continue
                us, ok = bench_ours(dirn, n, b)
                ours[dirn] = us
                ours_ok[dirn] = ok
                if not ok:
                    fails.append((dirn, n, b))
                tdir = "r2c" if dirn == "r2c" else "c2r"
                tt = torch_rows.get((n, b), {})
                t[tdir] = tt.get("r2c" if dirn == "r2c" else "c2r")
            acl_us = acl.get((n, b))
            row = dict(n=n, b=b,
                       r2c=ours.get("r2c"), c2r=ours.get("c2r"),
                       r2c_ok=ours_ok.get("r2c"), c2r_ok=ours_ok.get("c2r"),
                       ok=all(ours_ok.values()) if ours_ok else False,
                       torch_rfft=t.get("r2c"), torch_irfft=t.get("c2r"),
                       acl_rfft=acl_us,
                       r2c_vs_torch=(t.get("r2c") / ours["r2c"])
                       if ours.get("r2c") and t.get("r2c") else None,
                       r2c_vs_acl=(acl_us / ours["r2c"])
                       if ours.get("r2c") and acl_us else None,
                       c2r_vs_torch=(t.get("c2r") / ours["c2r"])
                       if ours.get("c2r") and t.get("c2r") else None)
            rows.append(row)
            print(f"n={n:<5} b={b:<5} r2c={row['r2c']} c2r={row['c2r']}", file=sys.stderr,
                  flush=True)

    if a.out:
        dst = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        json.dump(rows, open(dst, "w"), indent=1)
        print(f"wrote {dst}", file=sys.stderr)
    if fails:
        print("FAILS:", fails)
        return 1

    common = [n for n in ns if n >= 128]
    print("geo ours r2c      %8.2f us (n>=128)" % geo([d["r2c"] for d in rows if d["r2c"]]))
    print("geo ours c2r      %8.2f us" % geo([d["c2r"] for d in rows if d["c2r"]]))
    print("geo torch rfft    %8.2f us (same pts)"
          % geo([d["torch_rfft"] for d in rows if d["n"] in common]))
    print("geo torch irfft   %8.2f us" % geo([d["torch_irfft"] for d in rows]))
    print("geo aclRfft1D     %8.2f us (same pts)"
          % geo([d["acl_rfft"] for d in rows if d["n"] in common]))
    g_ours_r2c = geo([d["r2c"] for d in rows if d["r2c"] and d["n"] in common])
    g_torch_r2c = geo([d["torch_rfft"] for d in rows if d["n"] in common])
    g_acl = geo([d["acl_rfft"] for d in rows if d["n"] in common])
    print("r2c vs torch rfft : %5.2fx faster" % (g_torch_r2c / g_ours_r2c))
    if not math.isnan(g_acl):
        print("r2c vs aclRfft1D  : %5.2fx faster" % (g_acl / g_ours_r2c))
    print("c2r vs torch irfft: %5.2fx faster"
          % (geo([d["torch_irfft"] for d in rows])
             / geo([d["c2r"] for d in rows if d["c2r"]])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
