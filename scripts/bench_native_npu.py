#!/usr/bin/env python3
"""华为 NPU 原生复数 FFT 基准：torch.fft.fft（NPU 后端）。

实现归属：`_fft_c2c` 由 **torch_npu 自带的 op-plugin** 提供
（`FFTc2cKernelNpuOpApi.cpp` / `FFTPlanNpuOpApi.cpp`），带 fft plan cache，
**不是** CANN 算子库条目 —— CANN 9.0.0 的公开头文件里没有复数->复数 FFT 的 C API
（只有 `aclRfft1D` 实->复、`aclSTFT`）。因此本文档/图里沿用的「CANN 原生」一词
指「本卡原生复数 FFT 路径」；裸 CANN C API 一路见 `baseline_rfft --e2e`。

  python3 scripts/bench_native_npu.py --ns 256,1024,4096 --bs 1,64,4096 --reps 20
输出每行：NATIVE n=<n> b=<b> native_us=<us> maxRel=<rel>
maxRel 为 NPU 输出与 torch CPU 双精度参考的相对误差（判据 1e-4）。
"""
import argparse, os, sys, time, warnings

import numpy as np
import torch
import torch_npu

warnings.filterwarnings("ignore")


def bench(fn, reps):
    """返回 (min_us, mean_us)。min 与 Plan::measure 同口径；mean 与 fft_check 同口径。"""
    torch.npu.synchronize()
    fn()  # warmup
    torch.npu.synchronize()
    best = float("inf")
    acc = 0.0
    n = max(1, reps)
    for _ in range(n):
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        u = (time.perf_counter_ns() - t0) / 1e3
        best = min(best, u)
        acc += u
    return best, acc / n




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="256,512,1024,2048,4096")
    ap.add_argument("--bs", default="1,64,4096")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--e2e", action="store_true",
                    help="额外测端到端（H2D + 变换 + D2H），输出 NATIVE_E2E 行；"
                         "该 shape 的首调用会作为冷启动 first_us 单列")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]

    if not torch.npu.is_available():
        print("NPU 不可用", file=sys.stderr)
        return 1

    # 进程级冷启动（模块装载 / 上下文建立）会把首个 shape 的 first_us 拉到上百 ms，
    # 与单点 plan 冷构建混在一起。先用一次极小变换把这部分消耗掉，让 first_us
    # 只反映该 shape 的 plan 构建。
    with torch.npu.device("npu"):
        _w = torch.fft.fft(torch.zeros(8, dtype=torch.complex64, device="npu"))
        _w.cpu()
    torch.npu.synchronize()

    gen = torch.Generator().manual_seed(1234)
    for n in ns:
        for b in bs:
            re = torch.randn(b, n, generator=gen)
            im = torch.randn(b, n, generator=gen)
            ref = torch.fft.fft(re.to(torch.float64) + 1j * im.to(torch.float64), dim=-1)

            x_cpu = (re + 1j * im).to(torch.complex64)

            # ---- 端到端（H2D + 变换 + D2H）：必须排在 device-only 之前，
            #      这样 first_us 才是该 shape 的冷调用（CANN plan 尚未构建）。
            #      主机缓冲默认 pinned（与 fft_check / baseline_rfft 的
            #      AB_E2E_HOST=pinned 同口径）：pageable 的 H2D/D2H 走主机侧
            #      staging + 缺页，带宽随 loadavg 摆 2~4×，会把「拷贝路径没选对」
            #      记成 kernel 的账。AB_E2E_HOST=pageable 可复现受限口径。
            if a.e2e:
                pinned = os.environ.get("AB_E2E_HOST", "pinned") != "pageable"
                xc = x_cpu.pin_memory() if pinned else x_cpu
                xd = torch.empty((b, n), dtype=torch.complex64, device="npu")
                yd = (torch.empty((b, n), dtype=torch.complex64, pin_memory=True)
                      if pinned else None)

                def e2e_once(xc=xc, dev=xd, dst=yd):
                    dev.copy_(xc)                      # H2D
                    out = torch.fft.fft(dev, dim=-1)   # 变换
                    if dst is None:
                        return out.cpu()               # D2H（隐式同步）
                    dst.copy_(out)                     # D2H 到预分配 pinned 缓冲
                    return dst

                torch.npu.synchronize()
                t0 = time.perf_counter_ns()
                e2e_once()
                first_us = (time.perf_counter_ns() - t0) / 1e3
                e_min, e_mean = bench(e2e_once, a.reps)

            x = x_cpu.npu()
            y = torch.fft.fft(x, dim=-1)
            err = (y.cpu().to(torch.complex128) - ref).abs().max().item() / ref.abs().max().item()

            us_min, us_mean = bench(lambda: torch.fft.fft(x, dim=-1), a.reps)
            ok = "PASS" if err <= 1e-4 else "FAIL"
            print(f"NATIVE n={n} b={b} native_us={us_min:.1f} native_mean_us={us_mean:.1f} "
                  f"maxRel={err:.3e} {ok}")
            if a.e2e:
                print(f"NATIVE_E2E n={n} b={b} e2e_us={e_min:.1f} e2e_mean_us={e_mean:.1f} "
                      f"first_us={first_us:.1f} maxRel={err:.3e} {ok}")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
