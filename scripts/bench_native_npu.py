#!/usr/bin/env python3
"""华为 NPU 原生复数 FFT 基准：torch.fft.fft（NPU 后端）。

实现归属：`_fft_c2c` 由 **torch_npu 自带的 op-plugin** 提供
（`FFTc2cKernelNpuOpApi.cpp` / `FFTPlanNpuOpApi.cpp`），带 fft plan cache，
**不是** CANN 算子库条目 —— CANN 9.0.0 的公开头文件里没有复数->复数 FFT 的 C API
（只有 `aclRfft1D` 实->复、`aclSTFT`）。因此本文档/图里沿用的「CANN 原生」一词
指「本卡原生复数 FFT 路径」；裸 CANN C API 一路见 `baseline_rfft --e2e`。

  python3 scripts/bench_native_npu.py --ns 256,1024,4096 --bs 1,64,4096 --reps 20
  python3 scripts/bench_native_npu.py --input ofdm --ns 2048 --bs 14,140 --e2e
输出（stdout）每行：
  NATIVE     n=<n> b=<b> native_us=<min> native_mean_us=<mean> maxRel=<rel> PASS|FAIL
  NATIVE_E2E n=<n> b=<b> e2e_us=<min> e2e_mean_us=<mean> first_us=<us> maxRel=<rel> PASS|FAIL
             （第二行只在 --e2e 时输出；first_us = 该 shape 的冷调用）
maxRel 为 NPU 输出与 torch CPU 双精度参考的相对误差（判据 1e-4）。
"""
import argparse, os, sys, time, warnings

import numpy as np
import torch
import torch_npu

warnings.filterwarnings("ignore")


def _xh(bidx, k):
    """xorshift32 PRNG，与 src/host/fft_check.cpp::appHash 逐位一致。

    bidx 形状 (B,1)、k 形状 (1,n) —— 广播成 (B,n) 后 seed = b*2654435761 + k*40503。
    """
    x = bidx.astype(np.uint32) * np.uint32(2654435761) + k.astype(np.uint32) * np.uint32(40503)
    for _ in range(2):                       # C++ 里同样两轮 (<<13, >>17, <<5)
        x ^= (x << np.uint32(13))
        x ^= (x >> np.uint32(17))
        x ^= (x << np.uint32(5))
    return x


def app_input(mode, n, b):
    """应用形状输入（与 fft_check 的 AB_INPUT 同公式同分布，位级一致）。

    ofdm  16-QAM 子载波（DC/保护带置零、每 12 个一个导频）—— 多载波通信/频域均衡
    radar 4 个目标的复指数距离回波（逐脉冲多普勒相位）    —— 雷达距离门
    dl    每 8 点一块的复激活（幅度 + 4PSK 相位）          —— 深度学习频域层
    返回 (re, im) float32，形状 (b, n)。
    """
    bidx = np.arange(b, dtype=np.uint32)[:, None]
    k = np.arange(n, dtype=np.uint32)[None, :]
    x = _xh(bidx, k)
    if mode == "ofdm":
        guard = (k == 0) | (k < n // 16) | (k >= n - n // 16)
        pilot = (k % 12) == 0
        q = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) * np.float32(0.31622776)
        re = np.where(guard, np.float32(0.0),
                      np.where(pilot, np.where((x & 1) != 0, np.float32(1.0), np.float32(-1.0)),
                               q[(x >> 0) & 3]))
        im = np.where(guard, np.float32(0.0),
                      np.where(pilot, np.where((x & 2) != 0, np.float32(1.0), np.float32(-1.0)),
                               q[(x >> 2) & 3]))
        return re.astype(np.float32), im.astype(np.float32)
    if mode == "radar":
        # 常量先按 float32 存（与 C++ 的 float 数组一致）再提升 double，
        # 收尾先 (float) 求和再乘 0.25f —— 这样与 fft_check 逐位一致。
        fr = np.array([1 / 16, 1 / 5, 1 / 3, 0.62], dtype=np.float32).astype(np.float64)
        am = np.array([1, .5, .25, .125], dtype=np.float32).astype(np.float64)
        dop = np.array([0, 1 / 64, -1 / 128, 3 / 256], dtype=np.float32).astype(np.float64)
        sr = np.zeros((b, n), dtype=np.float64)
        si = np.zeros((b, n), dtype=np.float64)
        for t in range(4):
            ph = (2.0 * np.pi * (fr[t] * k + dop[t] * bidx)
                  + (t * 0.7 + (((x >> 8) & 255) / 255.0) * 0.3))
            sr += am[t] * np.cos(ph)
            si += am[t] * np.sin(ph)
        return (sr.astype(np.float32) * np.float32(0.25),
                si.astype(np.float32) * np.float32(0.25))
    if mode == "dl":
        xb = _xh(bidx, k // 8)
        mag = (((xb >> 8) & 255) / 255.0).astype(np.float32)
        phs = (xb >> 16) & 3
        c = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        s = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        return mag * c[phs], mag * s[phs]
    raise ValueError(f"unknown input mode: {mode}")


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
    ap.add_argument("--input", default="rand", choices=["rand", "ofdm", "radar", "dl"],
                    help="输入形状：rand=高斯随机（默认，与既有读数一致）；"
                         "ofdm/radar/dl=应用形状（与 fft_check 的 AB_INPUT 同公式）")
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
            if a.input == "rand":
                re = torch.randn(b, n, generator=gen)
                im = torch.randn(b, n, generator=gen)
            else:
                ar, ai = app_input(a.input, n, b)
                re = torch.from_numpy(ar)
                im = torch.from_numpy(ai)
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
