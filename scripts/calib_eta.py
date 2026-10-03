#!/usr/bin/env python3
"""η 成本模型最小二乘标定。

  python3 scripts/calib_eta.py [--reps 20] [--ns 128,1024,4096] [--bs ...]

模型（A7 重标定，含批折叠 D）：
  eta = launchUs + iters * (opNs * #op + elemNs * #elem)          (µs 口径)
    D        = bfly::foldDFor(n, B, 48)     （A3~A5：批折叠系数）
    groups   = ceil(B/D)                    （拍进 Level-0 repeat 的组数）
    blocks   = min(48, groups)              （活跃核数）
    iters    = ceil(groups/blocks)          （每核要跑的组数；墙钟取最慢核）
    #op/#elem= counts(n, D)                 （一个 group 的口径）
#op / #elem 按 include/butterfly/fft_k.hpp + src/ascendc/fft_radix2.cpp 逐条数出
（与 estimate() 同一份公式），不是自由拟合量。
拟合按 **1/y 加权**（闸门是相对误差，未加权会被 n=4096/B=4096 那种大点主导）。
旧模型没有 D：它假设每组只装 1 个 batch，于是折叠点（D=4）把 #op 少算 D 倍 ⇒
实测 η/实测−1 高达 +67%~+104%（n=64/4096 最坏 +103.6%）。
udCore = 48（Ascend910_9382 的 vector core 数）。
输出拟合参数、逐点残差、留一交叉验证（LOO）。
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD_CORE = 48
CALIB_SET_NS = (64, 256, 1024, 4096)
CALIB_SET_BS = (1, 4, 16, 64, 192, 256, 1024, 4096)


def plane_k_for(n):
    """与 include/butterfly/fft_k.hpp 的 planeKFor() 完全一致（A5 的 kMin 规则）。"""
    k_need = (n + 63) >> 6                    # 使 rows = n/K <= 64 的最小 K
    k_min = k_need if (n <= 1024 and k_need > 8) else 8
    best, best_key = k_min, (1 << 62)
    logn = n.bit_length() - 1
    K = k_min
    while K <= 32:
        if K * 8 > n:
            break
        logK = K.bit_length() - 1
        ops = 0
        si = 0
        while si + 1 < logK:
            h = 1 << si
            blocks = K // (4 * h)
            ops += blocks * (16 + (h - 1) * 28)
            si += 2
        if si < logK:
            h = 1 << si
            pairs, triv = K // 2, K // (2 * h)
            ops += triv * 6 + (pairs - triv) * 8
        for s in range(logK, logn):
            h = 1 << s
            groups = n // (2 * h)
            n_slice = (h + 63) >> 6
            merge = (h >= 8) and ((2 * h) % 8 == 0) and ((2 * h) // 8 <= 255) \
                and (h // 8 <= 255) and (groups <= 255) and (n_slice <= groups)
            ops += 8 * n_slice if merge else 8 * groups
            ops += 1
        ops += 7
        if ops <= best_key:
            best, best_key = K, ops
        K <<= 1
    return best


def fold_d_for(n, b, nblk=48):
    """与 include/butterfly/fft_k.hpp 的 foldDFor() 完全一致。"""
    D = 4                                  # kFoldCap
    if D <= 1:
        return 1
    K = plane_k_for(n)
    if K == 0 or n % K != 0:
        return 1
    if (n >> 3) > 255:
        return 1
    logK = K.bit_length() - 1
    groups_max = n >> (logK + 1)
    if groups_max == 0:
        return 1
    while D > 1 and D * groups_max > 255:
        D -= 1
    if nblk > 1:
        while D > 1 and ((b - 1) // D + 1) < nblk:
            D -= 1
    return D


def counts(n, D=1):
    """返回 (算子调用数, 元素访存数) —— **一个 group（D 个 batch）** 口径，不含 launch。

    与 estimate() 同一份公式；与 src/ascendc/fft_radix2.cpp 逐段对应：
      plane 段  useL0(g>1 且 n/8<=255) 时 Level-0 repeat=g 一次做完 g 批 ⇒ 算子数与
                g 无关、元素数 ×g；rows>64 时按 64 元素行切片，算子 ×nRowSlice。
                否则（g==1 或 n>=2048）走 Level-2 + 逐批 for ⇒ 算子/元素都 ×g（g==1 即原样）。
      planar 段 merge（内核条件含 `rep=g*groups<=255` 与 `nSlice<=groups || g>1`）
                算子 = 1+8*nSlice（与 g 无关）；否则 1+8*g*groups。
      IO 段    1 次 DataCopy + 6 次逐 d Gather/DataCopy ⇒ 1+6g；元素 10n*g。
    """
    g = D
    logn = n.bit_length() - 1
    K = plane_k_for(n)
    logK = K.bit_length() - 1
    rows = n // K
    use_l0 = (g > 1) and ((n >> 3) <= 255)
    n_row_slice = (rows + 63) >> 6

    plane_ops = plane_elems = 0.0
    si = 0
    while si + 1 < logK:
        h = 1 << si
        blocks = K // (4 * h)
        o = blocks * (16 + (h - 1) * 28)
        plane_ops += o
        plane_elems += o * rows
        si += 2
    if si < logK:
        h = 1 << si
        pairs, triv = K / 2, K / (2 * h)
        o = triv * 6 + (pairs - triv) * 8
        plane_ops += o
        plane_elems += o * rows
    if use_l0:
        plane_ops *= n_row_slice                 # Level-0 走一遍 d 循环就 break
        plane_elems *= g * n_row_slice
    else:
        plane_ops *= g                           # Level-2 走满 d 循环
        plane_elems *= g

    planar_ops = planar_elems = 0.0
    for s in range(logK, logn):
        h = 1 << s
        grp = n // (2 * h)
        n_slice = (h + 63) >> 6
        rep = g * grp
        merge = (h >= 8) and ((2 * h) % 8 == 0) and ((2 * h) // 8 <= 255) \
            and (h // 8 <= 255) and (rep <= 255) and (n_slice <= grp or g > 1)
        if merge:
            planar_ops += 1 + 8 * n_slice        # 与 g 无关（Level-0 一次盖 g 批）
        else:
            planar_ops += 1 + 8 * g * grp        # 逐 d × 逐 group
        # 元素数与 merge 与否无关（同一批数据，只是指令打包方式不同）
        planar_elems += 4 * n * g + h

    ops = plane_ops + planar_ops + 1 + 6 * g
    elems = plane_elems + planar_elems + 10 * n * g
    return ops, elems


def measure(n, b, reps, rounds=3):
    """跑 `rounds` 次取 **最小的 mean**。

    本机宿主负载 20~30，单次 mean 会被瞬时离群点抬高 30~60%
    （实测 n=64/B=4096 单次 mean 在 71~111 µs 之间摆），拟合若用单次会被噪声主导。
    取 min-of-means 仍与 matrix 的「mean」同口径，只是剔掉了负载尖峰。
    """
    best = float("nan")
    for _ in range(max(1, rounds)):
        r = subprocess.run(["./build/fft_check", str(n), str(b), str(reps)],
                           cwd=ROOT, capture_output=True, text=True)
        s = r.stdout + r.stderr
        m = re.search(r"([\d.]+) us/call", s)
        if not m or "PASS" not in s:
            continue
        v = float(m.group(1))
        if not (v == v) or not (best == best) or v < best:
            best = v
    if best != best:
        print(f"  !! n={n} B={b} 测量失败")
    return best


def lstsq3(x_op, x_elem, y):
    """解 min || (y - (c0 + c1*x_op + c2*x_elem)) / y ||（**按 1/y 加权**）。

    闸门是相对误差 |η/实测−1| ≤ 15%，而未加权最小二乘按绝对误差优化，
    会被 n=4096/B=4096（1498 µs）这类大点主导、牺牲掉小点。
    """
    import numpy as np
    A = np.column_stack([np.ones(len(y)), x_op, x_elem])
    w = 1.0 / np.asarray(y)
    c, *_ = np.linalg.lstsq(A * w[:, None], np.asarray(y) * w, rcond=None)
    return [float(v) for v in c]


def lstsq4(x_grp, x_op, x_elem, y):
    """解 min || y - (c0 + c1*x_grp + c2*x_op + c3*x_elem) ||。"""
    import numpy as np
    A = np.column_stack([np.ones(len(y)), x_grp, x_op, x_elem])
    c, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)
    return [float(v) for v in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ns", default=",".join(map(str, CALIB_SET_NS)))
    ap.add_argument("--bs", default=",".join(map(str, CALIB_SET_BS)))
    a = ap.parse_args()
    ns, bs = [int(x) for x in a.ns.split(",")], [int(x) for x in a.bs.split(",")]

    rows = []          # (n, b, iters, op_total, elem_total, t)
    for n in ns:
        print(f"n={n}  K={plane_k_for(n)}", flush=True)
        for b in bs:
            D = fold_d_for(n, b, UD_CORE)
            o, e = counts(n, D)
            t = measure(n, b, a.reps)
            groups = -(-b // D)                        # ceil(B/D)
            blocks = min(UD_CORE, groups)
            iters = -(-groups // blocks)               # 每 block 要跑的组数
            rows.append((n, b, iters, o * iters, e * iters, t))
            print(f"    B={b:<5} D={D} groups={groups:<4} iters={iters:<3} "
                  f"op={o:6.1f} elem={e:7.0f}  {t:9.1f} us", flush=True)

    rows = [r for r in rows if r[5] == r[5]]
    y  = [r[5] for r in rows]
    xo = [r[3] for r in rows]
    xe = [r[4] for r in rows]
    c = lstsq3(xo, xe, y)
    launch, opns, elemns = c
    print(f"\nfit: launchUs={launch:.6f} us  opNs={opns*1000:.4f} ns  "
          f"elemNs={elemns*1000:.5f} ns")

    def fit_excluding(i):
        idx = [j for j in range(len(rows)) if j != i]
        return lstsq3([xo[j] for j in idx], [xe[j] for j in idx], [y[j] for j in idx])

    res, loo = [], []
    print("    n     B       fit      LOO")
    for i, (n, b, *_ , t) in enumerate(rows):
        pred = launch + xo[i] * opns + xe[i] * elemns
        e1 = (pred - t) / t * 100.0
        c2 = fit_excluding(i)
        pred2 = c2[0] + xo[i] * c2[1] + xe[i] * c2[2]
        e2 = (pred2 - t) / t * 100.0
        res.append(abs(e1)); loo.append(abs(e2))
        print(f"  {n:<6} {b:<7} {e1:+6.1f}%  {e2:+6.1f}%")
    import numpy as np
    print(f"\nresidual mean={np.mean(res):.1f}%  max={max(res):.1f}%")
    print(f"LOO      mean={np.mean(loo):.1f}%  max={max(loo):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
