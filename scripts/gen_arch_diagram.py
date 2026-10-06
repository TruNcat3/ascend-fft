#!/usr/bin/env python3
"""生成 docs/figures/architecture.svg：系统分层 + 端到端数据流架构图。

上半：选型闭环四层（硬件事实 → 设计空间 → 成本模型 η → Plan/kernel），
      虚线是 measure() 的实测回填（Measured > Feasible）。
下半：端到端数据流（H2D → kernel → D2H，三方统一 pinned 主机缓冲），
      标出 device-only 与端到端两个计时区。

纯 matplotlib、无网络依赖；重复运行输出字节稳定（SVG 不写时间戳）。

  python3 scripts/gen_arch_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 图内文字全部转成路径（matplotlib 默认 svg.fonttype='path'），
# 这样 GitHub 渲染时不必依赖服务器上的字体；本机需有 CJK 字体。
for _fam in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"):
    if any(f.name == _fam for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = [_fam]
        break
matplotlib.rcParams["svg.fonttype"] = "path"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "architecture.svg")

W, H = 120.0, 66.0

STAGES = [
    ("① 硬件事实 H", "#0b6bcb", [
        "config/*.json ← hw_probe.sh",
        "48 AIV · UB 196608 B",
        "mask ≤ 64 · repeat ≤ 255",
        "无 SIMT ⇒ shuffle 判 Infeasible",
    ]),
    ("② 设计空间 G A P L F Q", "#8a6d0b", [
        "config/butterfly_space.json",
        "ctx.enumerate() → 全部候选",
        "n=4096：816 Infeasible / 48 Feasible",
        "不可行必须给原因",
    ]),
    ("③ 成本模型 η", "#0f7a4f", [
        "estimate() = launchUs +",
        "iters · (opNs·#op + elemNs·#elem)",
        "rank() → topK 实测",
        "偏差 平均 7.7%",
    ]),
    ("④ Plan + AscendC kernel", "#a3312a", [
        "prepare() / run() / measure()",
        "平面级 radix-4 · planar 级 · 批折叠 D ≤ 4",
        "K 由 planeKFor(n) 择优",
        "49 : 0 vs CANN 原生复数 FFT",
    ]),
]


def box(ax, x, y, w, h, title, color, lines, title_fs=11.5, body_fs=8.6):
    """画一个分层框：顶部色带 + 标题，下面若干行正文。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=1.3, edgecolor=color, facecolor="#fbfbfd", zorder=2))
    band = 6.2
    ax.add_patch(FancyBboxPatch(
        (x, y + h - band), w, band, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=0, facecolor=color, zorder=3))
    ax.add_patch(plt.Rectangle((x, y + h - band), w, 1.4,
                               linewidth=0, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + h - band / 2 - 0.15, title, ha="center", va="center",
            fontsize=title_fs, color="white", fontweight="bold", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + 2.0, y + h - band - 3.6 - i * 4.1, ln, ha="left", va="center",
                fontsize=body_fs, color="#1c1c22", zorder=4)


def arrow(ax, x0, y0, x1, y1, color="#41414d", lw=1.9, style="-|>", ls="solid", z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=16, linewidth=lw, color=color,
                                 linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(13.2, 6.82))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---------- 上半：选型闭环四层 ----------
    top_y, top_h = 37.0, 24.0
    gap, n = 3.4, len(STAGES)
    bw = (W - 2 * 3.0 - gap * (n - 1)) / n
    xs = [3.0 + i * (bw + gap) for i in range(n)]

    ax.text(3.0, 64.0, "选型闭环（host framework）", fontsize=12.5,
            fontweight="bold", color="#1c1c22", ha="left", va="center")

    for i, (title, color, lines) in enumerate(STAGES):
        box(ax, xs[i], top_y, bw, top_h, title, color, lines)
        if i < n - 1:
            arrow(ax, xs[i] + bw + 0.5, top_y + top_h / 2,
                  xs[i + 1] - 0.9, top_y + top_h / 2)

    # 实测回填：④ → ③（虚线，绕到下方）
    x4 = xs[3] + bw
    x3c = xs[2] + bw / 2
    ax.plot([x4 - bw / 2, x4 - bw / 2, x3c], [top_y, top_y - 3.4, top_y - 3.4],
            color="#0f7a4f", lw=1.7, ls=(0, (4, 3)), zorder=4)
    arrow(ax, x3c, top_y - 3.4, x3c, top_y - 0.7, color="#0f7a4f", lw=1.7,
          style="-|>", ls=(0, (4, 3)))
    ax.text((x4 - bw / 2 + x3c) / 2, top_y - 4.6,
            "measure() 实测回填 → rank() 让 Measured 优先于 Feasible",
            ha="center", va="top", fontsize=8.8, color="#0f7a4f")

    # ---------- 下半：端到端数据流 ----------
    bot_y, bot_h = 4.0, 17.0
    ax.text(3.0, 30.6, "端到端数据流（device ↔ host）", fontsize=12.5,
            fontweight="bold", color="#1c1c22", ha="left", va="center")

    bw_in, bw_k, bw_out, bh = 24.0, 40.0, 24.0, 16.0
    x_in = 4.0
    x_k = x_in + bw_in + 12.0
    x_out = x_k + bw_k + 12.0
    ky = 8.0

    for (bx, title, color, lines) in [
        (x_in, "输入 float32 交错 [batch][2n]", "#41414d",
         ["host buffer（pinned）", "ctx.select() 后 plan->prepare()"]),
        (x_k, "kfft_fwd（AscendC，AIV × 48）", "#a3312a",
         ["设备上只跑 kernel", "旋转因子 / 索引由 host 预生成"]),
        (x_out, "输出 float32 交错 [batch][2n]", "#41414d",
         ["host buffer（pinned）", "maxRel ≤ 1e-4 判据验收"]),
    ]:
        w = bw_in if bx == x_in else (bw_k if bx == x_k else bw_out)
        box(ax, bx, ky, w, bh, title, color, lines, title_fs=10.2, body_fs=8.4)

    arrow(ax, x_in + bw_in + 0.6, ky + bh / 2, x_k - 0.9, ky + bh / 2,
          color="#41414d")
    arrow(ax, x_k + bw_k + 0.6, ky + bh / 2, x_out - 0.9, ky + bh / 2,
          color="#41414d")
    ax.text(x_in + bw_in + 6.3, ky + bh / 2 + 3.0, "H2D", ha="center",
            va="bottom", fontsize=9.0, color="#41414d")
    ax.text(x_k + bw_k + 6.3, ky + bh / 2 + 3.0, "D2H", ha="center",
            va="bottom", fontsize=9.0, color="#41414d")

    # 计时区标注
    y_dev = ky - 2.4
    ax.plot([x_k + 1.0, x_k + bw_k - 1.0], [y_dev, y_dev], color="#a3312a", lw=1.6)
    ax.plot([x_k + 1.0, x_k + 1.0], [y_dev, y_dev + 1.4], color="#a3312a", lw=1.6)
    ax.plot([x_k + bw_k - 1.0, x_k + bw_k - 1.0], [y_dev, y_dev + 1.4],
            color="#a3312a", lw=1.6)
    ax.text(x_k + bw_k / 2, y_dev - 1.6, "device-only 计时区（launch + 同步）",
            ha="center", va="top", fontsize=9.0, color="#a3312a")

    y_e2e = ky + bh + 2.4
    x_l, x_r = x_in + 1.0, x_out + bw_out - 1.0
    ax.plot([x_l, x_r], [y_e2e, y_e2e], color="#0b6bcb", lw=1.6)
    for xx in (x_l, x_r):
        ax.plot([xx, xx], [y_e2e - 1.4, y_e2e], color="#0b6bcb", lw=1.6)
    ax.text((x_l + x_r) / 2, y_e2e + 1.2,
            "端到端计时区 = H2D + 变换 + D2H（三方主机缓冲统一 pinned，AB_E2E=10）",
            ha="center", va="bottom", fontsize=9.4, color="#0b6bcb")

    fig.savefig(OUT, format="svg",
                metadata={"Date": None, "Creator": "scripts/gen_arch_diagram.py"})
    plt.close(fig)

    # matplotlib 给 clip-path 生成随机 uuid，抹掉以便重复运行字节稳定
    import re
    with open(OUT, encoding="utf-8") as f:
        svg = f.read()
    ids = sorted(set(re.findall(r'\bid="(p[0-9a-f]{8,})"', svg)))
    for i, old in enumerate(ids):
        new = "clip%d" % i
        svg = svg.replace(old, new).replace("url(#%s)" % old, "url(#%s)" % new)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"-> {os.path.relpath(OUT, ROOT)}  ({os.path.getsize(OUT) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
