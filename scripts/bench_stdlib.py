#!/usr/bin/env python3
"""标准库 FFT 基准 + 与 NPU kernel 的交叉验证。

  perf  [--ns n1,n2,...] [--bs b1,b2,...] [--reps K]
        -> 每行: STDLIB n=<n> b=<b> numpy_us=<..> torch_us=<..>
  check <in.bin> <out.bin> <n> <batch>
        -> 与 torch.fft.fft / numpy.fft.fft 比 maxRel（判据 1e-4）

输入/输出布局与 kfft_fwd 一致：float32 交错复数 [batch][2*n]。
"""
import argparse, os, sys, time
import numpy as np

def bench(fn, x, reps):
    for _ in range(3):
        fn(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t)
    return sum(ts) / len(ts) * 1e6, min(ts) * 1e6

def cmd_perf(a):
    try:
        import torch
    except ImportError:
        torch = None
    print("# stdlib baseline: numpy.fft.fft / torch.fft.fft")
    for n in a.ns:
        for b in a.bs:
            xi = (np.sin(0.011 * np.arange(b * 2 * n)) +
                  0.25 * np.cos(0.037 * np.arange(b * 2 * n))).astype(np.float32)
            xc = xi.reshape(b, n, 2)
            cn = (xc[..., 0] + 1j * xc[..., 1]).astype(np.complex64)
            nu, numin = bench(lambda z: np.fft.fft(z, axis=-1), cn, a.reps)
            tu, tumin = float('nan'), float('nan')
            if torch is not None:
                t = torch.from_numpy(cn.copy())
                tu, tumin = bench(lambda z: torch.fft.fft(z), t, a.reps)
            print(f"STDLIB n={n} b={b} numpy_us={nu:.1f} torch_us={tu:.1f} "
                  f"numpy_min_us={numin:.1f} torch_min_us={tumin:.1f}")

def load(f, n, batch):
    v = np.fromfile(f, dtype=np.float32)
    assert v.size == 2 * n * batch, f"{f}: {v.size} != {2*n*batch}"
    return v.reshape(batch, n, 2)

def cmd_check(a):
    import torch
    i, o = load(a.inp, a.n, a.batch), load(a.out, a.n, a.batch)
    ref = (i[..., 0] + 1j * i[..., 1]).astype(np.complex64)
    y_np = np.fft.fft(ref, axis=-1)
    y_th = torch.fft.fft(torch.from_numpy(ref)).numpy()
    got = o[..., 0] + 1j * o[..., 1]
    rows = []
    for name, y in (("numpy", y_np), ("torch", y_th)):
        scale = max(np.abs(y).max(), 1e-30)
        mr = float(np.abs(got - y).max() / scale)
        rows.append((name, mr))
    print(f"cross-check n={a.n} b={a.batch} maxRel(ours vs ref) "
          + "  ".join(f"{n}={m:.3e}" for n, m in rows)
          + ("  PASS" if all(m <= 1e-4 for _, m in rows) else "  FAIL"))
    return 0 if all(m <= 1e-4 for _, m in rows) else 1

def parse(p):
    f = lambda s: [int(x) for x in s.split(',')]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('perf')
    p1.add_argument('--ns', type=f, default=[256, 512, 1024, 2048, 4096])
    p1.add_argument('--bs', type=f, default=[1, 64, 4096])
    p1.add_argument('--reps', type=int, default=20)
    p1.set_defaults(fn=cmd_perf)
    p2 = sub.add_parser('check')
    p2.add_argument('inp'); p2.add_argument('out')
    p2.add_argument('n', type=int); p2.add_argument('batch', type=int)
    p2.set_defaults(fn=cmd_check)
    return ap.parse_args(p)

if __name__ == '__main__':
    a = parse(sys.argv[1:])
    sys.exit(a.fn(a) or 0)
