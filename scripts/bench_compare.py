#!/usr/bin/env python3
"""汇总性能对比：标准库(numpy/torch, CPU) vs CANN 原生算子(NPU) vs 自研 kfft_fwd(NPU)。

CANN 原生算子有两套：
  * CANN 原生复数 FFT —— torch.fft.fft 跑在 NPU 上（torch_npu 后端，同为复->复，可直接对比）
  * aclRfft1D —— CANN 的实->复 rfft（norm=1），变换不同，仅作参照

  python3 scripts/bench_compare.py [--ns 256,1024,4096] [--bs 1,64,4096] [--reps 5]
输出 markdown 表 + 逐项 PASS/FAIL。
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abenv  # noqa: E402  与 scripts/env.sh 共用同一套路径探测
PY = abenv.python_bin()


def sh(cmd, env=None, cwd=ROOT, timeout=3600):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=e,
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def num(pat, s, default=float('nan')):
    m = re.search(pat, s)
    return float(m.group(1)) if m else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', default='256,512,1024,2048,4096')
    ap.add_argument('--bs', default='1,64,4096')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--no-rfft', action='store_true', help='跳过 aclRfft1D（省 ~2.2GB workspace）')
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(',')]
    bs = [int(x) for x in a.bs.split(',')]

    # 1) 标准库一次跑完
    print("[1/4] 标准库基准 (numpy / torch, CPU)...", file=sys.stderr)
    out = sh(f"{PY} scripts/bench_stdlib.py perf --ns {a.ns} --bs {a.bs} --reps {max(a.reps,5)}")
    std = {}
    for ln in out.splitlines():
        m = re.match(r"STDLIB n=(\d+) b=(\d+) numpy_us=([\d.]+) torch_us=([\d.]+)", ln)
        if m:
            std[(int(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))

    # 2) CANN 原生复数 FFT（NPU，一次跑完）
    print("[2/4] CANN 原生复数 FFT (torch_npu)...", file=sys.stderr)
    out = sh(f"{PY} scripts/bench_native_npu.py --ns {a.ns} --bs {a.bs} --reps {max(a.reps,10)}")
    nat, nat_ok = {}, True
    for ln in out.splitlines():
        # native_us=最小值 / native_mean_us=均值（口径需与本表"自研当前"列一致，取均值）
        m = re.match(r"NATIVE n=(\d+) b=(\d+) native_us=([\d.]+) native_mean_us=([\d.]+) "
                     r"maxRel=([\d.eE+-]+) (\w+)", ln)
        if m:
            nat[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))
            nat_ok &= (m.group(6) == "PASS")
    print(f"    native {len(nat)} 点 {'PASS' if nat_ok else 'FAIL'}", file=sys.stderr)

    # 3) NPU 三套 kernel
    rows = []
    for i, n in enumerate(ns):
        for b in bs:
            cur = sh(f"./build/fft_check {n} {b} {a.reps}")
            v1 = sh(f"AB_FFT_O=build/fft_radix2_v1.o ./build/fft_check {n} {b} {a.reps}")
            us_cur = num(r"kfft_fwd: ([\d.]+) us/call", cur)
            us_v1 = num(r"kfft_fwd: ([\d.]+) us/call", v1)
            ok = 'PASS' if re.search(r'^PASS$', cur, re.M) and re.search(r'^PASS$', v1, re.M) else 'FAIL'
            us_rf = float('nan')
            if not a.no_rfft:
                _tmpdir = os.path.join(abenv.work(), "bench_cmp")
                os.makedirs(_tmpdir, exist_ok=True)
                rf = sh(f"./build/baseline_rfft {n} {b} 1 {_tmpdir}/r.bin", timeout=600)
                us_rf = num(r"aclRfft1D n=\d+ b=\d+ : ([\d.]+) us/call", rf)
            nu, th = std.get((n, b), (float('nan'), float('nan')))
            us_na = nat.get((n, b), float('nan'))
            rows.append((n, b, nu, th, us_na, us_rf, us_v1, us_cur, ok))
            print(f"  n={n:<5} b={b:<5} {ok}  ours={us_cur:.1f}us", file=sys.stderr)

    # 4) 表
    print()
    print("# 性能对比：标准库 vs CANN 原生算子 vs 自研 kernel\n")
    print("> **NPU = Ascend910_9382（48 AIV）**；numpy/torch(CPU) 跑在 x86 CPU 上"
          "（`torch 2.10.0+cpu`, `numpy 1.26.4`）。")
    print("> **CANN 原生复数 FFT** = `torch.fft.fft` 走 torch_npu 后端（CANN 算子，复->复），"
          "与自研 `kfft_fwd` 同一变换，可直接对比。")
    print("> **`aclRfft1D`** 是实->复 rfft（`norm=1` 唯一合法），变换不同，仅作参照。\n")
    print("| n | batch | numpy (CPU) | torch (CPU) | **CANN 原生复数 FFT** | aclRfft1D | "
          "自研 v1 | **自研当前** | 自研/原生 | 正确性 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for n, b, nu, th, na, rf, v1, cur, ok in rows:
        f = lambda x: "—" if x != x else f"{x:,.1f}"
        if cur == cur and na == na and na > 0:
            r = na / cur
            vs = f"**{r:.2f}×**" if r >= 1 else f"{r:.2f}×"
        else:
            vs = "—"
        print(f"| {n} | {b} | {f(nu)} | {f(th)} | **{f(na)}** | {f(rf)} | {f(v1)} | "
              f"**{f(cur)}** | {vs} | {ok} |")
    bad = [r for r in rows if r[8] != 'PASS' or not nat_ok]
    print(f"\n**正确性：{len(rows)-len(bad)}/{len(rows)} PASS**（判据 maxRel ≤ 1e-4，"
          f"参考为双精度 CPU 基；numpy/torch/NPU 原生另经 `bench_stdlib.py check` 交叉验证）")
    print("\n`自研/原生` > 1 表示自研更快。")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
