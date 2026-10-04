#!/usr/bin/env python3
"""CANN 原生复数 FFT 的**墙钟对照**：一批 (n, batch) 上跑 mean/min。

与 `bench_native_npu.py`（带双精度参考与 maxRel 的正式基线）不同，本脚本
刻意保持极简：只打墙钟，用于和 msprof 的 `AscendTask.duration` 对照，
把「设备侧 kernel 时长」和「host 侧 API + 调度开销」分开（见
docs/trace与profile诊断-小尺寸与大尺寸.md §6.3.3）。

  python3 scripts/time_native.py                       # 默认 n=64..4096, b=4096
  python3 scripts/time_native.py --ns 64,256,1024 --bs 4096 --reps 10
输出：NATIVE_WALL n=<n> b=<b> mean=<us> min=<us>
"""
import argparse
import sys
import time

import torch
import torch_npu


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--bs", default="4096")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ns = [int(v) for v in a.ns.split(",") if v]
    bs = [int(v) for v in a.bs.split(",") if v]

    if not torch.npu.is_available():
        print("NPU 不可用", file=sys.stderr)
        return 1

    torch.npu.manual_seed(a.seed)
    for n in ns:
        for b in bs:
            x = (torch.randn(b, n) + 1j * torch.randn(b, n)).to(torch.complex64).npu()
            for _ in range(max(1, a.warmup)):
                y = torch.fft.fft(x, dim=-1)
            torch.npu.synchronize()
            ts = []
            for _ in range(a.reps):
                torch.npu.synchronize()
                t0 = time.perf_counter_ns()
                y = torch.fft.fft(x, dim=-1)
                torch.npu.synchronize()
                ts.append((time.perf_counter_ns() - t0) / 1e3)
            print(f"NATIVE_WALL n={n} b={b} mean={sum(ts) / len(ts):.1f} min={min(ts):.1f} "
                  f"reps={a.reps}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
