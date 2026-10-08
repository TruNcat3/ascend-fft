#!/usr/bin/env python3
"""从发布快照生成实验图（PNG，docs/figures/）。

默认数据源来自同一份已发布快照；脚本只画图，不重新测量：
  --matrix  results/published/.../matrix.md  C2C device-only 与模型校准字段
  --std     results/published/.../sixway.md  多基线上下文
  --e2e     results/published/.../e2e.json   C2C 端到端
  --e2e-app results/published/.../e2e_app.json 代表应用 shape
  --real    results/r2c_c2r.json       R2C/C2R 实数变换（fig10 胜/平/负热图）

  python3 scripts/plot_results.py --out docs/figures

图片一律用英文标签：仓库已有 CJK 字体（WenQuanYi）在本机可用，但换机器重跑时
没有 CJK 字体会变成豆腐块，英文标签可移植。中文说明写在 docs/实验对比.md 的图注里。
"""
import argparse, json, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LogNorm

from plot_style import COLORS, apply_publication_style, save_on_canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 取色：自研/基线用同一批，跨图保持一致 ----
C_OURS, C_NAT = "#1f77b4", "#d62728"
C_BASE = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#2ca02c", "#1f77b4"]
PALE = plt.get_cmap("tab20")
apply_publication_style()


PUBLISHED = "results/published/ascend910_9382-cann9.0.0"


def num(s):
    return float(str(s).replace("**", "").replace(",", "").replace("×", "")
                  .replace("x", "").replace("%", "").strip())


def parse_matrix(path):
    """Read both the rich archived table and the compact published table."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for ln in handle:
            if not ln.startswith("|"):
                continue
            cells = [cell.strip() for cell in ln.strip().strip("|").split("|")]
            if (len(cells) not in (5, 11) or not cells[0].isdigit()
                    or not cells[1].isdigit()):
                continue
            if len(cells) == 5:
                rows.append(dict(n=int(cells[0]), b=int(cells[1]),
                                 ours=num(cells[2]), nat=num(cells[3]),
                                 ratio=num(cells[4]), eta=float("nan"),
                                 dev=float("nan"), maxRel=float("nan")))
            else:
                rows.append(dict(n=int(cells[0]), b=int(cells[1]),
                                 ours=num(cells[2]), ours_min=num(cells[3]),
                                 nat=num(cells[4]), nat_min=num(cells[5]),
                                 ratio=num(cells[6]), eta=num(cells[7]),
                                 dev=num(cells[8]), maxRel=num(cells[9])))
    return rows


def parse_stdlib(path):
    """-> list of dict(n,b,numpy,torch,nat,rfft,v1,ours)"""
    rows, grab = [], False
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("## 表1"):
            grab = True
            continue
        if grab and ln.startswith("## 表2"):
            break
        if not grab:
            continue
        m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*"
                     r"\|\s*\**([\d.,]+)\**\s*\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*"
                     r"\|\s*\**([\d.,]+)\**\s*\|", ln)
        if not m:
            continue
        rows.append(dict(n=int(m.group(1)), b=int(m.group(2)),
                         numpy=num(m.group(3)), torch=num(m.group(4)),
                         nat=num(m.group(5)), rfft=num(m.group(6)),
                         v1=num(m.group(7)), ours=num(m.group(8))))
    return rows


def grid(rows, key):
    """list -> (ns, bs, matrix[ns,bs])"""
    ns = sorted({r["n"] for r in rows})
    bs = sorted({r["b"] for r in rows})
    a = np.full((len(ns), len(bs)), np.nan)
    ix, ib = {n: i for i, n in enumerate(ns)}, {b: i for i, b in enumerate(bs)}
    for r in rows:
        a[ix[r["n"]], ib[r["b"]]] = r[key]
    return ns, bs, a


def save(fig, out, name):
    p = os.path.join(out, name)
    save_on_canvas(fig, p)
    kb = os.path.getsize(p) // 1024
    print(f"  {name:32s} {kb:5d} KB")


def geo(values):
    values = np.asarray([v for v in values if v is not None and v > 0], dtype=float)
    return float(np.exp(np.mean(np.log(values)))) if values.size else float("nan")


def fig_overview(rows, e2rows, approws, out):
    """README headline: comparable groups, each with its baseline in the label."""
    groups = [
        ("C2C\ndevice-only", geo([r["ratio"] for r in rows]), "CANN native", len(rows)),
        ("C2C\nend-to-end", geo([r["e2e_ratio"] for r in e2rows]), "CANN native", len(e2rows)),
        ("Applications\nend-to-end", geo([r["e2e_ratio"] for r in approws]), "CANN native", len(approws)),
    ]
    groups = [g for g in groups if g[1] == g[1]]
    fig, ax = plt.subplots(figsize=(12.8, 6.3))
    x = np.arange(len(groups))
    values = [g[1] for g in groups]
    colors = [COLORS["ours"], COLORS["e2e"], COLORS["application"]][:len(groups)]
    bars = ax.bar(x, values, width=.62, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, color=COLORS["native"], linestyle="--", linewidth=1.6,
               label="baseline parity (1.0x)")
    for bar, value, (_, _, baseline, points) in zip(bars, values, groups):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .09,
                f"{value:.2f}x", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color="#101828")
        ax.text(bar.get_x() + bar.get_width() / 2, .12,
                f"vs {baseline}\n{points} points", ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("Geometric-mean speedup (higher is better)")
    ax.set_ylim(0, max(values) * 1.24)
    fig.suptitle("Ascend-FFT performance summary", x=.06, y=.98, ha="left",
                 fontsize=19, fontweight="bold")
    ax.set_title(
        "Ascend910_9382, FP32 C2C; correctness-gated results from one published snapshot",
        loc="left", fontsize=11, color=COLORS["neutral"], pad=12)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=.8, alpha=.65)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, out, "overview_performance.png")


def cell_label(ax, mat, fmt="{:.2f}", fs=7.5, thresh=None):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v != v:
                continue
            if thresh is not None and v < thresh:
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                    color="white" if False else "black")


# ------------------------------------------------------------------ fig1
def fig_speedup(rows, out):
    ns, bs, a = grid(rows, "ratio")
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    # 全网格都 >=1.04（没有 <1 的点），vmin 必须严格小于 vcenter 才能用 TwoSlopeNorm；
    # 下界取 0.9 让 1.0（盈亏平衡）落在色标中部偏左，读者一眼能看出 1.0 的位置。
    im = ax.imshow(a, cmap="RdYlGn",
                   norm=TwoSlopeNorm(vmin=min(0.9, a.min()), vcenter=1.0,
                                     vmax=a.max()), aspect="auto")
    cell_label(ax, a, "{:.2f}", 8)
    ax.set_xticks(range(len(bs)), [f"{b:,}" for b in bs])
    ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
    ax.set_xlabel("batch")
    ax.set_ylabel("n")
    wins = int((a > 1).sum())
    ax.set_title("Device-only speedup = CANN native latency / Ascend-FFT latency\n"
                 f"{wins}/{a.size} points faster, geometric mean "
                 f"{np.exp(np.mean(np.log(a))):.2f}x", fontsize=11)
    for i in range(len(ns) + 1):
        ax.axhline(i - .5, color="w", lw=.6)
        ax.axvline(i - .5, color="w", lw=.6) if i <= len(bs) else None
    cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.02)
    cb.set_label("speedup (x)")
    save(fig, out, "fig1_speedup_heatmap.png")


# ------------------------------------------------------------------ fig2
def fig_latency(rows, out):
    ns, bs, o = grid(rows, "ours")
    _, _, n_ = grid(rows, "nat")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)
    for ax, m, ttl, cm in ((axes[0], o, "Ours  kfft_fwd (us)", "Blues"),
                           (axes[1], n_, "CANN native complex FFT (us)", "Reds")):
        im = ax.imshow(m, cmap=cm, norm=LogNorm(vmin=max(1, np.nanmin(m)),
                                                vmax=np.nanmax(m)), aspect="auto")
        cell_label(ax, m, "{:,.0f}", 7.2)
        ax.set_xticks(range(len(bs)), [f"{b:,}" for b in bs])
        ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlabel("batch")
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.85, pad=.02)
    axes[0].set_ylabel("n")
    fig.suptitle("Latency heatmap, same 49-point grid (log colour scale)",
                 fontsize=12, y=1.02)
    save(fig, out, "fig2_latency_heatmap.png")


# ------------------------------------------------------------------ fig3
def fig_curves(rows, out):
    ns = sorted({r["n"] for r in rows})
    bs = sorted({r["b"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for ax, key, ttl in ((axes[0], "ratio", "Device-only speedup"),
                         (axes[1], "ours", "Ours latency (us)")):
        for i, n in enumerate(ns):
            xs = [r["b"] for r in rows if r["n"] == n]
            ys = [r[key] for r in rows if r["n"] == n]
            ax.plot(xs, ys, marker="o", ms=3.6, lw=1.5,
                    color=PALE(i * 2), label=f"n={n:,}")
        ax.set_xscale("log", base=2)
        # 直接标 batch 数值，别让读者去换算 2^10
        ax.set_xticks(bs, [f"{b:,}" for b in bs])
        ax.set_xlabel("batch")
        ax.grid(alpha=.3, which="both")
        ax.set_title(ttl, fontsize=11)
        if key == "ratio":
            ax.axhline(1.0, color="k", ls="--", lw=1.2)
            ax.annotate("1.0x  (break-even)", xy=(1.2, 1.05), fontsize=8.5)
            ax.set_ylabel("CANN native / Ascend-FFT")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("us")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.suptitle("Speedup and latency vs batch, one line per n", fontsize=12, y=1.03)
    save(fig, out, "fig3_speedup_curve.png")


# ------------------------------------------------------------------ fig4
def fig_baselines(rows, out):
    pick = [(64, 1), (64, 4096), (1024, 1), (1024, 4096), (4096, 1), (4096, 4096)]
    by = {(r["n"], r["b"]): r for r in rows}
    series = [("numpy (CPU)", "numpy", C_BASE[0]),
              ("torch (CPU)", "torch", C_BASE[1]),
              ("aclRfft1D", "rfft", C_BASE[2]),
              ("ours v1", "v1", C_BASE[3]),
              ("CANN native", "nat", C_NAT),
              ("Ours", "ours", C_OURS)]
    x = np.arange(len(pick))
    w = 0.13
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    for k, (lab, key, c) in enumerate(series):
        v = [by[p][key] for p in pick if p in by]
        ax.bar(x + (k - 2.5) * w, v, w, label=lab, color=c)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"n={n:,}\nB={b:,}" for n, b in pick])
    ax.set_ylabel("latency (us, log scale)")
    ax.set_title("Six-way comparison at six representative shapes\n"
                 "(aclRfft1D is real->complex, shown for reference only)",
                 fontsize=11)
    ax.grid(axis="y", alpha=.3, which="both")
    ax.legend(fontsize=8.5, ncol=3)
    for i, p in enumerate(pick):
        if p in by:
            r = by[p]
            # >1 = 自研更快（与 表2 的「÷CANN 原生」同口径），标在自研柱顶
            ax.annotate(f"{r['nat'] / r['ours']:.1f}x", xy=(i + 2.5 * w, r["ours"]),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color=C_OURS, weight="bold")
    save(fig, out, "fig4_sixway_bars.png")


# ------------------------------------------------------------------ fig5
def fig_eta(rows, out):
    valid = [r for r in rows if r.get("eta", float("nan")) > 0]
    if not valid:
        return False
    m = np.array([[r["ours"], r["eta"]] for r in valid])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    ax = axes[0]
    lo = max(1.0, m.min() * .8)
    g = np.linspace(lo, m.max() * 1.25, 50)
    ax.fill_between(g, g * .85, g * 1.15, color="0.85", label="+/-15% band")
    ax.plot(g, g, "k--", lw=1, label="ideal")
    ax.scatter(m[:, 0], m[:, 1], s=26, color=C_OURS, alpha=.85, zorder=3,
               label=f"{len(valid)} points (mean |dev| "
                     f"{np.mean(np.abs(m[:, 1] / m[:, 0] - 1)) * 100:.1f}%)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("measured, ours (us)")
    ax.set_ylabel("model estimate  eta (us)")
    ax.set_title("Estimate vs measurement", fontsize=11)
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    ns, bs, residual = grid(valid, "dev")
    ax = axes[1]
    bound = max(15.0, float(np.nanmax(np.abs(residual))))
    im = ax.imshow(residual, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    cell_label(ax, residual, "{:+.0f}%", 7.3)
    ax.set_xticks(range(len(bs)), [f"{b:,}" for b in bs])
    ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
    ax.set_xlabel("batch")
    ax.set_ylabel("n")
    ax.set_title("Relative residual by shape", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=.82, pad=.02)
    cb.set_label("(estimate / measured - 1), %")
    fig.suptitle("Cost-model calibration-grid check (not a held-out generalization test)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, out, "fig5_eta_scatter.png")
    return True


# ------------------------------------------------------------------ fig6
def fig_e2e(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), sharey=True)
    panels = [
        ("dev_ratio", "Device-only speedup", "speedup"),
        ("e2e_ratio", "End-to-end speedup", "speedup"),
        ("xfer_share", "Time outside device region", "share"),
    ]
    for ax, (key, title, kind) in zip(axes, panels):
        ns, bs, mat = grid(rows, key)
        if kind == "speedup":
            finite = mat[np.isfinite(mat)]
            norm = TwoSlopeNorm(vmin=min(.9, float(finite.min())), vcenter=1.0,
                                vmax=float(finite.max()))
            im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
            cell_label(ax, mat, "{:.2f}", 7.2)
            wins = int((finite > 1).sum())
            title += f"\n{wins}/{finite.size} wins, geo {geo(finite):.2f}x"
            cb_label = "CANN native / Ascend-FFT"
        else:
            im = ax.imshow(mat, cmap="magma", vmin=0.55, vmax=0.9, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j] * 100:.0f}%", ha="center",
                                va="center", fontsize=7.2, color="white")
            title += "\n1 - device / end-to-end"
            cb_label = "fraction"
        ax.set_xticks(range(len(bs)), [f"{b:,}" for b in bs], rotation=45,
                      ha="right")
        ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlabel("batch")
        ax.set_title(title, fontsize=10.2)
        cb = fig.colorbar(im, ax=ax, shrink=.78, pad=.02)
        cb.set_label(cb_label, fontsize=8.5)
    axes[0].set_ylabel("n")
    fig.suptitle("How device-only gains propagate to the host-to-host path",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig6_end_to_end.png")


APP_SHAPES = {
    "OFDM": {(2048, 14), (2048, 140), (4096, 14), (4096, 140)},
    "Radar": {(1024, 64), (1024, 256), (2048, 64), (2048, 256)},
    "DL frequency layer": {(1024, 32), (1024, 128), (4096, 32), (4096, 128)},
}


def fig_applications(rows, out):
    """Representative application-shaped FFT inputs; groups follow e2e_test.py."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True)
    for ax, (name, shapes) in zip(axes, APP_SHAPES.items()):
        subset = sorted((r for r in rows if (r["n"], r["batch"]) in shapes),
                        key=lambda r: (r["n"], r["batch"]))
        x = np.arange(len(subset))
        dev = [r["dev_ratio"] for r in subset]
        e2e = [r["e2e_ratio"] for r in subset]
        ax.bar(x - .19, dev, .38, color=COLORS["ours"], label="device-only")
        ax.bar(x + .19, e2e, .38, color=COLORS["application"], label="end-to-end")
        ax.axhline(1.0, color=COLORS["native"], ls="--", lw=1.2)
        ax.set_xticks(x, [f"{r['n']:,}\nB={r['batch']:,}" for r in subset], fontsize=8)
        ax.set_title(f"{name}\nE2E geo {geo(e2e):.2f}x", fontsize=10.5)
        ax.set_xlabel("FFT length / batch")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("CANN native / Ascend-FFT")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Representative application-shaped FFT inputs (12 correctness-gated points)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, out, "fig9_application_workloads.png")


# ------------------------------------------------------------------ fig7
C_BARE = "#ff7f0e"


def fig_e2e_three(e2e, out):
    """三路端到端/device 延迟 vs batch（两组 n）。裸 CANN 是实->复、搬运量减半，
    只作为「调用路径固定开销」参照，不与前两路算胜负。"""
    want_n = [64, 4096]
    series = [("Ours kfft_fwd", "ours", C_OURS),
              ("CANN native (torch)", "nat", C_NAT),
              ("bare CANN aclRfft1D*", "bare", C_BARE)]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.4))
    for i, (metric, ttl) in enumerate(
            (("dev", "Device-only  (kernel / op call)"),
             ("e2e", "End-to-end  (H2D + transform + D2H)"))):
        for j, n in enumerate(want_n):
            ax = axes[i][j]
            for lab, pre, c in series:
                pts = [(r["batch"], r.get(f"{pre}_{metric}")) for r in e2e
                       if r["n"] == n and r.get(f"{pre}_{metric}") == r.get(f"{pre}_{metric}")]
                if not pts:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                        lw=1.6, color=c, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            bs = sorted({r["batch"] for r in e2e if r["n"] == n})
            ax.set_xticks(bs, [f"{b:,}" for b in bs])
            ax.set_xlabel("batch")
            ax.set_ylabel("us (log)")
            ax.grid(alpha=.3, which="both")
            ax.set_title(f"{ttl}   n={n:,}", fontsize=10.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=8.5)
    fig.suptitle("Three-way latency: ours vs torch_npu vs bare CANN C API\n"
                 "* aclRfft1D is real->complex, onesided, half the transfer bytes: "
                 "reference only, no win/loss counted", fontsize=11.5, y=1.0)
    fig.tight_layout()
    save(fig, out, "fig7_three_way_e2e.png")


# ------------------------------------------------------------------ fig8
def fig_box(rows, e2rows, approws, out):
    """加速比分布箱型图：device-only(49) / 端到端(49) / 应用负载端到端(12)。

    一张图回答「加速比的范围有多大」：箱体 = Q1..Q3，中位线 = median，
    须 = min/max（离群点单独画），红虚线 = 1.0x 打平。
    """
    sets = []
    v = [r["ratio"] for r in rows if r.get("ratio") == r.get("ratio")]
    if v:
        sets.append(("Device-only\n49 points", v))
    v = [r["e2e_ratio"] for r in (e2rows or [])
         if r.get("e2e_ratio") == r.get("e2e_ratio")]
    if v:
        sets.append(("End-to-end\n49 points", v))
    v = [r["e2e_ratio"] for r in (approws or [])
         if r.get("e2e_ratio") == r.get("e2e_ratio")]
    if v:
        sets.append(("App loads,\nend-to-end (%d)" % len(v), v))
    if not sets:
        return
    data = [np.asarray(d, dtype=float) for _, d in sets]
    # min/max 写进 x 轴刻度的第三行，不画在坐标区里 —— 图内标注会压到
    # 坐标区底边与刻度标签上（端到端那箱的 min 还在 1.0x 红线之下）。
    labels = [f"{lb}\n{d.min():.2f}–{d.max():.2f}x"
              for lb, d in zip((lb for lb, _ in sets), data)]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    bp = ax.boxplot(data, positions=list(range(1, len(data) + 1)), widths=0.46,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", ms=3.5, mfc="0.4", mec="none", alpha=.7),
                    medianprops=dict(color="black", lw=1.7),
                    whiskerprops=dict(color="0.35"), capprops=dict(color="0.35"))
    for patch, c in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.32)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.3)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + rng.uniform(-0.14, 0.14, len(d)), d,
                   s=9, color=colors[i - 1], alpha=0.7, linewidths=0, zorder=3)

    ax.axhline(1.0, color="#d62728", ls="--", lw=1.2, zorder=4)
    ax.text(len(data) + 0.45, 1.0, "parity 1.0x", color="#d62728",
            fontsize=8.5, va="center", ha="left")
    for i, d in enumerate(data, 1):
        med = float(np.median(d))
        ax.text(i, med, f"  median {med:.2f}x", fontsize=8.5, va="center",
                ha="left", fontweight="bold")

    geos = " / ".join(f"{np.exp(np.mean(np.log(d))):.2f}x" for d in data)
    ax.set_title("Speedup vs CANN native (>1 = faster)\n"
                 f"geometric means: {geos}   (left to right)", fontsize=11)
    ax.set_ylabel("speedup (x)")
    ax.set_xticks(list(range(1, len(data) + 1)))
    ax.set_xticklabels(labels, fontsize=9.5)
    lo_all = min(float(d.min()) for d in data)
    hi_all = max(float(d.max()) for d in data)
    # 底边留 0.15 让 1.0x 红线与其下唯一的离群点（端到端 min 0.95x）不贴边
    ax.set_ylim(min(0.9, lo_all - 0.15),
                hi_all + max(0.4, 0.12 * (hi_all - lo_all)))
    ax.grid(axis="y", color="0.88", lw=.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, out, "fig8_speedup_boxplot.png")


# ------------------------------------------------------------------ fig10
def wtl_counts(ratios):
    """win/tie/loss with an explicit +/-5% parity band; NaN/None filtered."""
    r = np.asarray([v for v in ratios if v is not None and v == v], dtype=float)
    wins = int((r > 1.05).sum())
    losses = int((r < 0.95).sum())
    return wins, int(r.size - wins - losses), losses, int(r.size)


def fig_real(rows, out):
    """fig10: R2C/C2R device-only speedup vs torch_npu, full win/tie/loss grid."""
    panels = [
        ("R2C  vs  torch.fft.rfft  (n=128..8192)", "r2c_vs_torch"),
        ("C2R  vs  torch.fft.irfft (n=64..4096)", "c2r_vs_torch"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for ax, (title, key) in zip(axes, panels):
        data = [{"n": r["n"], "b": r["b"], "ratio": r.get(key)}
                for r in rows if r.get(key)]
        if not data:
            ax.set_visible(False)
            continue
        ns, bs, a = grid(data, "ratio")
        im = ax.imshow(a, cmap="RdYlGn",
                       norm=TwoSlopeNorm(vmin=min(0.9, float(np.nanmin(a))),
                                         vcenter=1.0,
                                         vmax=max(float(np.nanmax(a)), 1.01)),
                       aspect="auto")
        cell_label(ax, a, "{:.2f}", 7.5)
        ax.set_xticks(range(len(bs)), [f"{b:,}" for b in bs])
        ax.set_yticks(range(len(ns)), [f"{n:,}" for n in ns])
        ax.set_xlabel("batch")
        ax.set_ylabel("N")
        wins, ties, losses, total = wtl_counts([r["ratio"] for r in data])
        ax.set_title(f"{title}\n{wins} win / {ties} tie / {losses} "
                     f"loss of {total} points", fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("speedup (torch / ours)", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Real transforms: device-only speedup vs torch_npu on the same card",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015, "tie band 0.95x..1.05x; every point is correctness-gated "
             "(maxRel <= 1e-4, PASS) before timing", ha="center", fontsize=9,
             color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, out, "fig10_real_speedup_heatmap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=f"{PUBLISHED}/matrix.md")
    ap.add_argument("--std", default=f"{PUBLISHED}/sixway.md")
    ap.add_argument("--e2e", default=f"{PUBLISHED}/e2e.json")
    ap.add_argument("--e2e-app", default=f"{PUBLISHED}/e2e_app.json")
    ap.add_argument("--real", default="results/r2c_c2r.json")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()

    root = ROOT
    ap_ = lambda p: os.path.join(root, p)
    os.makedirs(ap_(a.out), exist_ok=True)

    rows = parse_matrix(ap_(a.matrix))
    if not rows:
        print("矩阵解析失败：没有匹配的行", file=sys.stderr)
        return 1
    print(f"matrix: {len(rows)} points")

    made = []
    print("figures ->", a.out)
    made.append("fig1_speedup_heatmap.png"); fig_speedup(rows, ap_(a.out))
    made.append("fig2_latency_heatmap.png"); fig_latency(rows, ap_(a.out))
    made.append("fig3_speedup_curve.png");   fig_curves(rows, ap_(a.out))
    if fig_eta(rows, ap_(a.out)):
        made.append("fig5_eta_scatter.png")
    else:
        print("矩阵无 eta 字段，跳过 fig5", file=sys.stderr)

    try:
        srows = parse_stdlib(ap_(a.std))
        if srows:
            print(f"stdlib: {len(srows)} points")
            made.append("fig4_sixway_bars.png"); fig_baselines(srows, ap_(a.out))
        else:
            print("stdlib 表1 解析为空，跳过 fig4", file=sys.stderr)
    except FileNotFoundError:
        print("stdlib 文档不存在，跳过 fig4", file=sys.stderr)

    er = []
    try:
        e = json.load(open(ap_(a.e2e), encoding="utf-8"))
        # e2e_test.py 用 "batch"，统一成 "b" 与矩阵表一致
        er = []
        for r in e["rows"]:
            if r.get("e2e_ratio") != r.get("e2e_ratio"):
                continue
            d, g = r.get("ours_dev"), r.get("ours_e2e")
            # 传输占比 = 1 - device/E2E（E2E 内核之外的时间份额），现算不依赖 JSON 旧字段
            share = (1 - d / g) if (d and g and g > 0) else float("nan")
            er.append({**r, "b": r.get("b", r.get("batch")), "xfer_share": share})
        if er:
            print(f"e2e: {len(er)} points")
            made.append("fig6_end_to_end.png"); fig_e2e(er, ap_(a.out))
            if any(r.get("bare_e2e") == r.get("bare_e2e") for r in er):
                made.append("fig7_three_way_e2e.png"); fig_e2e_three(er, ap_(a.out))
            else:
                print("e2e.json 无 bare_* 字段，跳过 fig7", file=sys.stderr)
        else:
            print("e2e.json 为空，跳过 fig6", file=sys.stderr)
    except FileNotFoundError:
        print(f"{a.e2e} 不存在，跳过 fig6（先跑 scripts/e2e_test.py）",
              file=sys.stderr)

    app_r = []
    try:
        aj = json.load(open(ap_(a.e2e_app), encoding="utf-8"))
        app_r = [r for r in aj.get("rows", [])
                 if r.get("e2e_ratio") == r.get("e2e_ratio")]
        if app_r:
            print(f"e2e_app: {len(app_r)} points")
            made.append("fig9_application_workloads.png")
            fig_applications(app_r, ap_(a.out))
    except (FileNotFoundError, ValueError):
        print(f"{a.e2e_app} 不存在，fig8 只画两箱", file=sys.stderr)

    try:
        real_rows = json.load(open(ap_(a.real), encoding="utf-8"))
        if real_rows:
            print(f"real: {len(real_rows)} points")
            made.append("fig10_real_speedup_heatmap.png")
            fig_real(real_rows, ap_(a.out))
    except (FileNotFoundError, ValueError):
        print(f"{a.real} 不存在或无法解析，跳过 fig10", file=sys.stderr)

    made.append("fig8_speedup_boxplot.png"); fig_box(rows, er, app_r, ap_(a.out))

    made.append("overview_performance.png")
    fig_overview(rows, er, app_r, ap_(a.out))

    print(f"done: {len(made)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
