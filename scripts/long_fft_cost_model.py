#!/usr/bin/env python3
"""R4/R5：长链融合转置的参数化代价模型 + top-k 实测闭环（回顾性验证）。

数据源：R2-A/R2-B 各归因归档（results/evidence/long-fft-boundary-attribution*/
attribution.json）。每条样本 = (配置, 形状) -> 融合链中位数。特征全部来自
配置与形状（nTiles/条带数/GM 载荷/rebuild/屏障形态/blocks）+ 每形状一次的
 incumbent FFT 段探针（fft1+fft2 段中位数，配置不变 => 选择时可先测一次）。

协议（预注册，写死在代码里）：
  - 校准形状 {8192x1, 16384x47}，测试形状 {32768x47, 65536x47}；
  - OLS 拟合 chain ~ features；对测试形状按预测排序，报 top-k(k=2) 命中
    与选择 regret = (pred_best - true_best)/true_best；
  - 候选闭包 = 已测配置全集（含 incumbent）；nb-session2 仅作稳定性注记。

用法：
  python3 scripts/long_fft_cost_model.py            # 生成 model.json
  python3 scripts/long_fft_cost_model.py --check    # 与已提交文件逐字节比对
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

EVID = "results/evidence"
OUT_DIR = os.path.join(EVID, "long-fft-cost-model")
OUT_PATH = os.path.join(OUT_DIR, "model.json")
CALIB_SHAPES = [(8192, 1), (16384, 47)]
TEST_SHAPES = [(32768, 47), (65536, 47)]
TOPK = 2
FEATURES = ["intercept", "gm_gb", "tiles_m", "stripes_m", "rebuild_m",
            "heavy_barrier_m", "pp", "blocks", "fft_probe_us"]


def long_split(n):
    """与 fft_check.cpp 相同的 N1/N2 分解（sqrt 附近平衡 2 幂因子）。"""
    target = 1
    while (target << 1) * (target << 1) <= n:
        target <<= 1
    for side in (0, 1):
        a = target >> 1 if side else target
        while a >= 64 and a <= 4096:
            if n % a == 0:
                b = n // a
                if 64 <= b <= 4096:
                    return a, b
            a = a >> 1 if side else (a << 1)
            if not side and a > 4096:
                break
    raise ValueError(f"no legal split for n={n}")


def n_tiles(n, batch, th, tw):
    n1, n2 = long_split(n)
    # issueLt(nRows, nCols)：transpose_in/out = (N1, N2)、boundary = (N2, N1)；
    # 内核 tiles = batch * ceil(rows/H) * ceil(cols/W)。
    def tiles(rows, cols):
        return batch * math.ceil(rows / th) * math.ceil(cols / tw)
    return 2 * tiles(n1, n2) + tiles(n2, n1)


def config_of(env):
    tile = env.get("AB_LT_TILE", "128x32")
    th, tw = (int(x) for x in tile.split("x"))
    k = int(env.get("AB_LT_STRIPE_K", "512"))
    resident = env.get("AB_LT_IDX") == "resident"
    pipe = env.get("AB_LT_PIPE", "")
    blocks = int(env.get("AB_LT_BLOCKS", "48"))
    return {"tile": tile, "th": th, "tw": tw, "k": k,
            "resident": resident, "pipe": pipe, "blocks": blocks}


def config_key(c):
    return (f"t{c['tile']}", f"k{c['k']}",
             "ri" if c["resident"] else "rb", c["pipe"] or "-",
             f"b{c['blocks']}")


def load_samples():
    rows = []
    stability = []
    for path in sorted(glob.glob(os.path.join(EVID, "long-fft-boundary-attribution*",
                                              "attribution.json"))):
        arch = os.path.basename(os.path.dirname(path))
        d = json.load(open(path))
        if d.get("problems") or d.get("incomplete"):
            continue
        cfg = config_of(d["manifest"].get("env", {}))
        key = config_key(cfg)
        for p in d["points"]:
            st = p.get("stats", {})
            fused = st.get("chain_fused_median")
            if fused is None:
                pairs = p.get("pairs", [])
                fused = statistics.median(q["fused"]["chain"] for q in pairs)
            segs = [q["fused"]["segments"] for q in p.get("pairs", [])]
            fft_probe = statistics.median(
                s["fft1"] + s["fft2"] for s in segs)
            rec = {
                "archive": arch,
                "config": key,
                "n": p["n"], "b": p["b"],
                "fused_chain_us": round(fused, 3),
                "fft_probe_us": round(fft_probe, 3),
                "gm_gb": p["fused_modeled_payload_gm_rw_bytes"] / 2**30,
                "cfg": cfg,
            }
            if "session2" in arch:
                stability.append(rec)
            else:
                rows.append(rec)
    return rows, stability


def features_of(rec):
    c = rec["cfg"]
    nt = n_tiles(rec["n"], rec["b"], c["th"], c["tw"])
    stripes = math.ceil(c["th"] * c["tw"] / c["k"])
    nb = c["pipe"] in ("nb", "pp")     # 条带内 PIPE_V（两个入口同型）
    heavy = 0.0 if nb else 1.0
    return [
        1.0,
        rec["gm_gb"],
        nt / 1e6,
        nt * stripes / 1e6,
        (0.0 if c["resident"] else nt / 1e6),
        nt * stripes * heavy / 1e6,
        1.0 if c["pipe"] == "pp" else 0.0,
        float(c["blocks"]),
        rec["fft_probe_us"],
    ]


def ols(X, y):
    """高斯-约当求 (X^T X)^-1 X^T y（纯 python，确定性）。"""
    n, m = len(X), len(X[0])
    a = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(m)]
         + [sum(X[k][i] * y[k] for k in range(n))] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        div = a[col][col] or 1e-12
        a[col] = [v / div for v in a[col]]
        for r in range(m):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [rv - f * cv for rv, cv in zip(a[r], a[col])]
    return [a[i][m] for i in range(m)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows, stability = load_samples()
    if not rows:
        print("no attribution samples found", file=sys.stderr)
        return 1
    calib = [r for r in rows if (r["n"], r["b"]) in CALIB_SHAPES]
    test = [r for r in rows if (r["n"], r["b"]) in TEST_SHAPES]
    X = [features_of(r) for r in calib]
    y = [r["fused_chain_us"] for r in calib]
    beta = ols(X, y)
    resid = [yy - sum(b * f for b, f in zip(beta, xx))
             for xx, yy in zip(X, y)]
    rmse = (sum(e * e for e in resid) / len(resid)) ** 0.5
    report = []
    hit = 0
    regrets = []
    for shape in TEST_SHAPES:
        grp = [r for r in test if (r["n"], r["b"]) == shape]
        if not grp:
            continue
        pred = []
        for r in grp:
            p = sum(b * f for b, f in zip(beta, features_of(r)))
            pred.append((p, r["config"], r["fused_chain_us"]))
        pred.sort()
        true = sorted(grp, key=lambda r: r["fused_chain_us"])
        topk = [c for _, c, _ in pred[:TOPK]]
        best = true[0]["config"]
        if best in topk:
            hit += 1
        chosen = next(r for r in grp if r["config"] == pred[0][1])
        regret = (chosen["fused_chain_us"] - true[0]["fused_chain_us"]) \
            / true[0]["fused_chain_us"]
        regrets.append(regret)
        report.append({
            "shape": f"{shape[0]}x{shape[1]}",
            "true_best": best,
            "true_best_us": true[0]["fused_chain_us"],
            "pred_best": pred[0][1],
            "pred_best_measured_us": chosen["fused_chain_us"],
            "topk": topk,
            "topk_hit": best in topk,
            "regret_pct": round(regret * 100, 3),
            "ranking": [{"config": c, "pred_us": round(p, 1),
                         "measured_us": m} for p, c, m in pred],
        })
    n_test = len(report)
    model = {
        "protocol": {
            "calib_shapes": [f"{n}x{b}" for n, b in CALIB_SHAPES],
            "test_shapes": [f"{n}x{b}" for n, b in TEST_SHAPES],
            "topk": TOPK,
            "features": FEATURES,
            "n_calib": len(calib),
            "n_test_points": n_test,
        },
        "coefficients": {k: round(v, 6) for k, v in zip(FEATURES, beta)},
        "calib_rmse_us": round(rmse, 3),
        "topk_hits": f"{hit}/{n_test}",
        "mean_regret_pct": round(100.0 * sum(regrets) / max(1, len(regrets)), 3),
        "test_report": report,
        "session2_stability": [
            {"config": r["config"], "n": r["n"], "b": r["b"],
             "fused_chain_us": r["fused_chain_us"]}
            for r in sorted(stability, key=lambda r: (r["config"], r["n"]))],
        "problems": [],
    }
    blob = json.dumps(model, indent=2, sort_keys=True) + "\n"
    if a.check:
        if not os.path.exists(OUT_PATH):
            print(f"missing {OUT_PATH}", file=sys.stderr)
            return 1
        cur = open(OUT_PATH).read()
        if cur != blob:
            print(f"{OUT_PATH} out of date", file=sys.stderr)
            return 1
        print("long_fft_cost_model --check OK")
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(blob)
    print(f"wrote {OUT_PATH}: topk={model['topk_hits']} "
          f"mean_regret={model['mean_regret_pct']}% rmse={model['calib_rmse_us']}us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
