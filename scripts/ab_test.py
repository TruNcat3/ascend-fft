#!/usr/bin/env python3
"""批量 A/B 消融：基线 kernel .o vs 候选 kernel .o（docs/性能优化-*.md §7/§9/§11 的可执行版）。

口径与文档一致 —— **每点 3 轮、每轮 reps 次取 min、逐点交替运行**以抵消共租户噪声，
`fft_check` 的 `.o` 由环境变量 `AB_FFT_O` 指定（**不要为了 A/B 去重编译**）。

  python3 scripts/ab_test.py --base build/fft_radix2_v1.o --cand build/fft_radix2.o \\
      --points 4096x4096,64x4096,128x4 --reps 50 --rounds 3

  # 只看某几个形状 + 输出 JSON
  python3 scripts/ab_test.py --base a.o --cand b.o --points 64x1 --json results/ab.json

输出（stdout）：
  # A/B  base=<.o>  cand=<.o>  reps=<K>  rounds=<R>  points=<P>    # 首行头
  AB n=<n> b=<b> base=<us> cand=<us> speedup=<x> rel=<maxRel> PASS|FAIL[  < REGRESS]
  AB n=<n> b=<b>  < 不完整（缺 base 或 cand 的读数）                # 某一侧没读到数
  [ FAIL ] round<rd> <base|cand> n=<n> b=<b>: <fft_check 报错尾行>  # 该次运行失败
  geomean speedup = <x>x   (...)   points=<k>/<P>  errors=<e>      # 最后一行：逐点几何均值
  -> <json 文件>                                                    # 仅给了 --json 时

历史基线 `.o`（如文档里出现过的 `/tmp/op/overlap.o`）可用
`scripts/baseline_o.sh <git-rev>` 从对应提交重新构建。
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_RE = re.compile(r"\(min\s+([\d.]+)\s+us\)")
REL_RE = re.compile(r"maxRel=([0-9.eE+-]+)")


def run(o_path, n, b, reps):
    env = dict(os.environ, AB_FFT_O=o_path)
    p = subprocess.run([os.path.join(ROOT, "build", "fft_check"), str(n), str(b), str(reps)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    out = (p.stdout or "") + (p.stderr or "")
    m, r = MIN_RE.search(out), REL_RE.search(out)
    if p.returncode != 0 or not m:
        return None, None, out.strip().splitlines()[-1] if out.strip() else f"rc={p.returncode}"
    return float(m.group(1)), (float(r.group(1)) if r else None), out


def geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="基线 kernel .o（AB_FFT_O 指向它）")
    ap.add_argument("--cand", required=True, help="候选 kernel .o")
    ap.add_argument("--points", default="4096x4096,64x4096,128x4",
                    help="n×b 列表，逗号分隔")
    ap.add_argument("--reps", type=int, default=50, help="每次调用内部 reps（默认 50）")
    ap.add_argument("--rounds", type=int, default=3, help="轮数，逐点取 min（默认 3）")
    ap.add_argument("--json", help="把结果另存到这个 JSON 文件")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="speedup 低于该值记为 REGRESS（默认 1.0 = 不许变慢）")
    a = ap.parse_args()

    base = a.base if os.path.isabs(a.base) else os.path.join(ROOT, a.base)
    cand = a.cand if os.path.isabs(a.cand) else os.path.join(ROOT, a.cand)
    for p in (base, cand):
        if not os.path.exists(p):
            print(f"找不到 .o：{p}", file=sys.stderr)
            return 2
    if not os.path.exists(os.path.join(ROOT, "build", "fft_check")):
        print("build/fft_check 不存在，先跑 scripts/build.sh check", file=sys.stderr)
        return 2

    pts = []
    for tok in a.points.split(","):
        tok = tok.strip().replace("x", ",").replace("*", ",")
        n, b = (int(v) for v in tok.split(","))
        pts.append((n, b))

    rows, sps, fails = [], [], 0
    print(f"# A/B  base={a.base}  cand={a.cand}  reps={a.reps}  rounds={a.rounds}  "
          f"points={len(pts)}")
    for n, b in pts:
        best = {"base": None, "cand": None}
        for rd in range(a.rounds):
            order = ("base", "cand") if rd % 2 == 0 else ("cand", "base")
            for side in order:
                o = base if side == "base" else cand
                us, rel, out = run(o, n, b, a.reps)
                if us is None:
                    print(f"  [ FAIL ] round{rd} {side} n={n} b={b}: {out}")
                    fails += 1
                    continue
                if best[side] is None or us < best[side]:
                    best[side] = us
                    if side == "cand":
                        best["rel"] = rel
        if best["base"] and best["cand"]:
            sp = best["base"] / best["cand"]
            ok = best.get("rel") is not None and best["rel"] <= 1e-4
            tag = "PASS" if ok else "FAIL"
            flag = "" if sp >= a.tol else "  < REGRESS"
            sps.append(sp)
            rows.append({"n": n, "b": b, "base_us": best["base"], "cand_us": best["cand"],
                         "speedup": sp, "maxRel": best.get("rel"), "ok": ok})
            print(f"AB n={n:<5d} b={b:<5d} base={best['base']:10.1f} cand={best['cand']:10.1f} "
                  f"speedup={sp:6.3f}x rel={best.get('rel'):.2e} {tag}{flag}")
        else:
            fails += 1
            print(f"AB n={n:<5d} b={b:<5d}  < 不完整（缺 base 或 cand 的读数）")
        sys.stdout.flush()

    print(f"\ngeomean speedup = {geo(sps):.3f}x   "
          f"({'全部 ≥ %.3g×' % a.tol if sps and min(sps) >= a.tol else '有回退点'})   "
          f"points={len(sps)}/{len(pts)}  errors={fails}")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"base": a.base, "cand": a.cand, "reps": a.reps,
                       "rounds": a.rounds, "geomean": geo(sps), "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"-> {a.json}")
    return 0 if fails == 0 and sps and min(sps) >= a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
