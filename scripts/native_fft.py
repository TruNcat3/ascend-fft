#!/usr/bin/env python3
"""torch_npu 原生复数 FFT 的 **profile 用例**（msprof 采集输入）。

单个 (n, batch) 形状，先 warmup 再连跑 reps 次；默认不打印计时，
避免 host 侧日志混进 profile 的 api_statistic。

  python3 scripts/native_fft.py --n 4096 --b 4096 --reps 10
  python3 scripts/native_fft.py --n 4096 --b 4096 --reps 10 --time   # 顺带打墙钟

采集（用 scripts/profile_test.sh 一键，或手工）：
  source scripts/env.sh
  ab_prof "$AB_WORK/p_nat" --task-time=on --aic-metrics=PipeUtilization -- \
      "$AB_PY" scripts/native_fft.py --n 4096 --b 4096 --reps 10

归属说明：torch_npu 的 `_fft_c2c` 来自其自带 op-plugin，**不是** CANN 算子库条目
（CANN 9.0.0 没有复数→复数 FFT 的 C API）。详见 scripts/bench_native_npu.py 的 docstring。
"""
import argparse
import sys
import time

import torch
import torch_npu


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--b", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=10, help="计入 profile 的迭代次数")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", action="store_true", help="打印墙钟 mean/min（µs）")
    a = ap.parse_args()

    if not torch.npu.is_available():
        print("NPU 不可用", file=sys.stderr)
        return 1

    torch.npu.manual_seed(a.seed)
    x = (torch.randn(a.b, a.n) + 1j * torch.randn(a.b, a.n)).to(torch.complex64).npu()

    for _ in range(max(1, a.warmup)):
        y = torch.fft.fft(x, dim=-1)
    torch.npu.synchronize()

    ts = []
    for _ in range(a.reps):
        if a.time:
            torch.npu.synchronize()
            t0 = time.perf_counter_ns()
        y = torch.fft.fft(x, dim=-1)
        if a.time:
            torch.npu.synchronize()
            ts.append((time.perf_counter_ns() - t0) / 1e3)
    torch.npu.synchronize()

    if a.time:
        print(f"NATIVE n={a.n} b={a.b} reps={a.reps} "
              f"wall_mean={sum(ts) / len(ts):.1f} wall_min={min(ts):.1f} us")
    print(f"NATIVE_PROFILE n={a.n} b={a.b} reps={a.reps} shape={tuple(y.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
