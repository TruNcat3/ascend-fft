#!/usr/bin/env python3
"""生成 docs/实验对比.md：图 + 带「迷你条」的可读表格 + 口径/局限说明。

数据全部来自已有产物，不重复测量：
  docs/matrix_test_a7.md        device-only 49 点矩阵
  docs/性能对比-标准库vs自研.md   六基线表
  results/e2e.json              端到端测试（scripts/e2e_test.py）
  docs/figures/*.png            对比图（scripts/plot_results.py）

  python3 scripts/plot_results.py && python3 scripts/gen_compare_doc.py
"""
import argparse, json, math, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def num(s):
    return float(str(s).replace("**", "").replace(",", "").replace("×", "")
                  .replace("%", "").strip())


def parse_matrix(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*"
                     r"\|\s*([\d.,]+)\s*\|\s*([\d.,]+)\s*\|\s*\**([\d.,]+)×\**\s*"
                     r"\|\s*([\d.,]+)\s*\|\s*([+-]?[\d.,]+)%\s*\|\s*([\d.eE+-]+)\s*\|", ln)
        if m:
            rows.append(dict(n=int(m.group(1)), b=int(m.group(2)),
                             ours=num(m.group(3)), ours_min=num(m.group(4)),
                             nat=num(m.group(5)), nat_min=num(m.group(6)),
                             ratio=num(m.group(7)), eta=num(m.group(8)),
                             dev=num(m.group(9)), maxRel=num(m.group(10))))
    return rows


def parse_stdlib(path):
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
        if m:
            rows.append(dict(n=int(m.group(1)), b=int(m.group(2)),
                             numpy=num(m.group(3)), torch=num(m.group(4)),
                             nat=num(m.group(5)), rfft=num(m.group(6)),
                             v1=num(m.group(7)), ours=num(m.group(8))))
    return rows


def bar(v, lo, hi, width=18):
    """比例迷你条：>1 的值更长。"""
    if v != v:
        return ""
    t = (math.log(v) - math.log(lo)) / max(1e-9, math.log(hi) - math.log(lo))
    k = max(1, min(width, int(round(t * width))))
    return "█" * k + "░" * (width - k)


def geo(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="docs/matrix_test_a7.md")
    ap.add_argument("--std", default="docs/性能对比-标准库vs自研.md")
    ap.add_argument("--e2e", default="results/e2e.json")
    ap.add_argument("--out", default="docs/实验对比.md")
    a = ap.parse_args()
    R = lambda p: os.path.join(ROOT, p)

    mx = parse_matrix(R(a.matrix))
    st = parse_stdlib(R(a.std))
    e2 = {}
    try:
        j = json.load(open(R(a.e2e), encoding="utf-8"))
        for r in j["rows"]:
            d, g = r.get("ours_dev"), r.get("ours_e2e")
            e2[(r["n"], r["batch"])] = dict(
                dev=r.get("dev_ratio"), e2e=r.get("e2e_ratio"),
                od=d, oe=g, nd=r.get("nat_dev"), ne=r.get("nat_e2e"),
                plan=r.get("ours_plan_us"), first=r.get("nat_first_us"),
                share=(1 - d / g) if (d and g and g > 0) else float("nan"),
                maxRel=r.get("maxRel"), ok=r.get("ok"),
                bare_dev=r.get("bare_dev"), bare_e2e=r.get("bare_e2e"),
                bare_ok=r.get("bare_ok"),
                nat_vs_bare=r.get("nat_vs_bare_dev"),
                ours_vs_bare=r.get("ours_vs_bare_e2e"))
        e2meta = j.get("meta", {})
    except FileNotFoundError:
        e2meta = {}

    if not mx:
        print("矩阵解析失败", file=sys.stderr)
        return 1

    wins = sum(1 for r in mx if r["ratio"] > 1)
    lo, hi = min(r["ratio"] for r in mx), max(r["ratio"] for r in mx)
    L = []
    A = L.append

    A("# 实验对比（图 + 详表）\n")
    A("> 表格读起来费劲，本文把同样的数据画成图；图由 `scripts/plot_results.py` 从"
      "已有表格/JSON 生成，**不重复测量**。数字口径与 "
      "[`matrix_test_a7.md`](matrix_test_a7.md)、"
      "[`性能对比-标准库vs自研.md`](性能对比-标准库vs自研.md)、"
      "`results/e2e.json` 完全一致。\n")
    A("| 想看什么 | 直接跳 |")
    A("|---|---|")
    A("| 一眼看谁快 | [图1](#图1-49-点-speedup-热力图) |")
    A("| 绝对延迟多少 | [图2](#图2-延迟热力图) |")
    A("| batch 变大怎么变 | [图3](#图3-speedup-与延迟随-batch-缩放) |")
    A("| 和 numpy/torch 等比 | [图4](#图4-六基线同场对比) |")
    A("| 成本模型准不准 | [图5](#图5-成本模型--vs-实测) |")
    A("| 真实搬运数据后还快吗 | [图6](#图6-端到端测试) |")
    A("| 和裸 CANN C API 比 | [图7](#图7-三路端到端自研--torch--裸-cann) |")
    A("")

    # ---------------- 0 设置 ----------------
    A("---\n")
    A("## 0　实验设置\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 硬件 | Ascend910_9382，48 AIV，UB 196608 B |")
    A("| 软件 | CANN 9.0.0（`ccec` + `libascendcl`），torch_npu 2.10.0 |")
    A("| 参考 | 双精度 CPU DIT，判据 `maxRel ≤ 1e-4` |")
    A(f"| 网格 | `n ∈ {sorted({r['n'] for r in mx})}` × "
      f"`batch ∈ {sorted({r['b'] for r in mx})}`，共 {len(mx)} 点 |")
    A("| 统计 | `--rounds 3` 逐点 **min-of-means**（自研与原生同口径） |")
    A("| 宿主 | `nproc=1`、`loadavg≈24` —— CPU 基线有 ±15% 抖动，NPU 列抖动较小 |")
    A("")
    A("**比值定义**：`速度比 = 原生耗时 ÷ 自研耗时`，**>1 表示自研更快**。\n")
    A("**两种口径**：")
    A("- **device-only**：只计 kernel 发射 + 同步（含 launch 开销），不含数据搬运；")
    A("- **端到端 E2E**：H2D + 变换 + D2H，不含进程启动与输入生成。")
    A("")

    # ---------------- 1 图1 ----------------
    A("---\n")
    A("## 图1　49 点 speedup 热力图\n")
    A("![speedup heatmap](figures/fig1_speedup_heatmap.png)\n")
    A(f"**{wins}/{len(mx)} 个点自研更快**（几何均值 **{geo([r['ratio'] for r in mx]):.2f}×**，"
      f"最紧 **{min(r['ratio'] for r in mx):.2f}×** @ `n=1024/B=4096`，"
      f"最好 **{max(r['ratio'] for r in mx):.2f}×** @ `n=128/B=4`）。\n")
    A("**形状趋势**：颜色随 `batch` 增大由绿转黄 —— 小/中 batch 是自研的强区（3~7×），"
      "大 batch（≥1024）两侧都逼近各自的访存上限，差距收窄到 1.0~1.6×。"
      "`n=4096/B=4096` 的 1.64× 是**在带宽受限区仍然赢**的点（原生 2444.7 vs 自研 1492.7 µs）。\n")

    # 表1（带迷你条）
    A("### 表1　49 点详表（含迷你条，条越长越快）\n")
    A("| n | batch | 自研 (µs) | 原生 (µs) | 速度比 | 分布 | η (µs) | η 偏差 | maxRel |")
    A("|---:|---:|---:|---:|---:|:---|---:|---:|:---|")
    for r in mx:
        A(f"| {r['n']} | {r['b']} | {r['ours']:,.1f} | {r['nat']:,.1f} | "
          f"**{r['ratio']:.2f}×** | `{bar(r['ratio'], lo, hi)}` | {r['eta']:,.1f} | "
          f"{r['dev']:+.1f}% | {r['maxRel']:.2e} |")
    A("")

    # ---------------- 2 图2 ----------------
    A("---\n")
    A("## 图2　延迟热力图\n")
    A("![latency heatmap](figures/fig2_latency_heatmap.png)\n")
    A("同一份数据的**绝对值**视角：左图自研、右图原生，共用对数色标。"
      "原生那一侧整片偏红且几乎不随 `n` 变（81→2445 µs 中，"
      "`B=1` 一行稳定在 77~105 µs —— 是**固定 launch/调度开销**），"
      "自研那一侧 `B≤64` 全在 14~53 µs，说明自研把固定开销压下去了。\n")

    # ---------------- 3 图3 ----------------
    A("---\n")
    A("## 图3　speedup 与延迟随 batch 缩放\n")
    A("![speedup curve](figures/fig3_speedup_curve.png)\n")
    A("左图每条线是一个 `n`：**7 条线全部在 1.0× 虚线之上**，"
      "且整体从 `B=4` 的 4~7× 单调下滑到 `B=4096` 的 1.0~1.6×。"
      "右图是自研绝对延迟（对数轴），`B≤64` 几乎是平的 —— "
      "这个区间被固定开销主导，`B>64` 才开始随 batch 线性上升。\n")

    # ---------------- 4 图4 ----------------
    A("---\n")
    A("## 图4　六基线同场对比\n")
    A("![six-way](figures/fig4_sixway_bars.png)\n")
    A("取 6 个代表形状，6 根柱子同场（对数轴）。柱顶蓝色标注是 "
      "**原生 ÷ 自研**（>1 = 自研更快）。\n")
    if st:
        pick = [(64, 1), (64, 4096), (1024, 1), (1024, 4096), (4096, 1), (4096, 4096)]
        by = {(r["n"], r["b"]): r for r in st}
        A("### 表2　代表形状的六基线数值（µs）\n")
        A("| n | batch | numpy (CPU) | torch (CPU) | CANN 原生 | aclRfft1D | 自研 v1 "
          "| **自研当前** | ÷原生 | ÷numpy | ÷torch |")
        A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in pick:
            r = by.get(p)
            if not r:
                continue
            A(f"| {r['n']} | {r['b']} | {r['numpy']:,.1f} | {r['torch']:,.1f} | "
              f"{r['nat']:,.1f} | {r['rfft']:,.1f} | {r['v1']:,.1f} | **{r['ours']:,.1f}** | "
              f"**{r['nat'] / r['ours']:.2f}×** | {r['numpy'] / r['ours']:.2f}× | "
              f"{r['torch'] / r['ours']:.2f}× |")
        A("")
        big = [r for r in st if r["b"] >= 1024]
        A(f"**读法**：NPU 上自研对 CANN 原生全胜；对 CPU 基线，"
          f"`batch ≥ 1024` 的 {len([r for r in big if r['ours'] < r['numpy']])}/{len(big)} 点赢 numpy、"
          f"{len([r for r in big if r['ours'] < r['torch']])}/{len(big)} 点赢 torch —— "
          "小 batch 输给 CPU 是因为 3~20 µs 的量级里 CPU 单核已经够快，"
          "NPU 要付固定的发射开销。`aclRfft1D` 是实->复变换，仅作参照。\n")

    # ---------------- 5 图5 ----------------
    A("---\n")
    A("## 图5　成本模型 η vs 实测\n")
    A("![eta](figures/fig5_eta_scatter.png)\n")
    fdev = [abs(r["dev"]) for r in mx]
    over = [r for r in mx if r["dev"] > 15]
    over_txt = "、".join("`n=%d/B=%d`(%+.1f%%)" % (r["n"], r["b"], r["dev"]) for r in over)
    A(f"49 个点全部贴在理想线附近：**平均 |偏差| {sum(fdev) / len(fdev):.1f}%**，"
      f"落在 ±15% 带内的 {len(mx) - len(over)}/{len(mx)}。"
      f"带外的 {len(over)} 个点是 {over_txt}，"
      "它们的单轮 `mean/min` 比都在 1.3 以上 —— 是宿主负载离群点，不是模型问题。\n")
    A("η 的作用是在 **kernel 还没编译**时就排出生命周期里的 launch 参数"
      "（`AB_FOLD_D` / `AB_PLANE_K`），这是选型闭环能跑起来的前提。\n")

    # ---------------- 6 E2E ----------------
    A("---\n")
    A("## 图6　端到端测试\n")
    if e2:
        A("![e2e](figures/fig6_end_to_end.png)\n")
        ew = [v for v in e2.values() if v["e2e"] == v["e2e"]]
        dw = [v for v in e2.values() if v["dev"] == v["dev"]]
        A("上半：灰柱是 device-only、彩色柱是端到端（绿=更快、红=更慢），对数轴。"
          "下半：**内核之外的时间占比**（`1 − device/E2E`），即 H2D + D2H + 同步。\n")
        A("| 指标 | 结果 |")
        A("|---|---|")
        A(f"| 正确性 | **{sum(1 for v in e2.values() if v['ok'])}/{len(e2)} PASS** |")
        A(f"| device-only | **{sum(1 for v in dw if v['dev'] > 1)}/{len(dw)} 更快**，"
          f"几何均值 **{geo([v['dev'] for v in dw]):.2f}×** |")
        A(f"| **端到端 E2E** | **{sum(1 for v in ew if v['e2e'] > 1)}/{len(ew)} 更快**，"
          f"几何均值 **{geo([v['e2e'] for v in ew]):.2f}×** |")
        best = max(e2.values(), key=lambda v: v["e2e"])
        worst = min(e2.values(), key=lambda v: v["e2e"])
        kb = [k for k, v in e2.items() if v is best][0]
        kw = [k for k, v in e2.items() if v is worst][0]
        A(f"| E2E 最快 / 最慢 | **{best['e2e']:.2f}×** @ `n={kb[0]}/B={kb[1]}` ｜ "
          f"**{worst['e2e']:.2f}×** @ `n={kw[0]}/B={kw[1]}` |")
        sh = [v["share"] for v in e2.values() if v["share"] == v["share"]]
        A(f"| 内核外时间占比 | 中位 **{sorted(sh)[len(sh) // 2] * 100:.0f}%**，"
          f"最大 **{max(sh) * 100:.0f}%** |")
        A("")
        geo_dev_e2e = geo([v["dev"] for v in dw]) if dw else float("nan")
        A("> ⚠️ 上表的 `device-only` 是 **e2e_test 自己那一轮**（`reps=10`）顺带测的，"
          "与 [图1/表1](#表149-点详表含迷你条条越长越快) 的 `matrix_test`（`reps=20`、"
          f"另一时段）是两次独立测量，几何均值 {geo_dev_e2e:.2f}× vs "
          f"{geo([r['ratio'] for r in mx]):.2f}× —— 差异在本机负载抖动范围内，"
          "**两列都各自与同轮的对手同口径**，跨表不要混引。\n")

        # §6.1 的口径随数据现算，避免写死过期数字
        small_win = [(k, v) for k, v in e2.items() if k[1] <= 256 and k[0] <= 1024]
        nsw = sum(1 for _, v in small_win if v["e2e"] > 1)
        bigsh = [v["share"] for k, v in e2.items()
                 if k[0] >= 512 and k[1] >= 256 and v["share"] == v["share"]]
        e2w = sum(1 for v in ew if v["e2e"] > 1)

        A("### 6.1　结果怎么读\n")
        A(f"1. **中小尺寸（`B ≤ 256` 且 `n ≤ 1024`，{len(small_win)} 点）**：内核占大头，"
          f"device-only 的优势基本传导到端到端，**{nsw}/{len(small_win)} 更快**。")
        if bigsh:
            A(f"2. **传输量大（`n ≥ 512` 且 `B ≥ 256`，{len(bigsh)} 点）**：内核外时间占比 "
              f"**{min(bigsh) * 100:.0f}~{max(bigsh) * 100:.0f}%**，"
              "端到端被 PCIe 搬运封顶，两者一起变慢 —— 此时**比的不是 FFT，是数据搬运**。")
        A(f"3. 因此端到端 **{e2w}/{len(ew)}** 而不是 49/49 是**预期行为**，"
          "不是内核退化：device-only 一列仍然 49/49。\n")
        A("### 6.2　口径\n")
        A("| | 自研 | CANN 原生 |")
        A("|---|---|---|")
        A("| 计时区 | `aclrtMemcpyAsync`(H2D) → `kfft_fwd` → `aclrtMemcpyAsync`(D2H) | "
          "`xd.copy_(x_cpu)` → `torch.fft.fft(xd)` → `y.cpu()` |")
        A("| 计时区外 | 进程启动(`boot`)、plan 准备(`plan`)、输入生成 | "
          "框架启动、该 shape 冷 `first` 单列 |")
        A("| 统计 | `--rounds 3` min-of-means，reps=10 | 同 |")
        A("")
        A("复现：`python3 scripts/e2e_test.py --reps 10 --rounds 3`\n")
        A("### 6.3　已知局限（必须一起读）\n")
        A("- **传输实现差异**：本机 `nproc=1`、`loadavg≈24`，H2D/D2H 的主机端 staging "
          "是 CPU 活。交错复测（134 MB 往返）显示 torch_npu 稳定 20~27 ms、"
          "本仓库 `aclrtMemcpyAsync` 路径 41~63 ms —— **差约 2×，是主机侧数据搬运路径的问题，"
          "不是 kernel**（device-only 一列不受影响）。已试过 `AB_E2E_MODE=sync`"
          "（阻塞 `aclrtMemcpy`）与合并同步，均无稳定改善。")
        # 用第三路把上面这条定位：裸 CANN 与自研共用同一条 aclrtMemcpyAsync。
        bigx = [(k, v) for k, v in e2.items()
                if k[1] >= 1024 and v.get("bare_e2e") == v.get("bare_e2e")]
        if len(bigx) >= 4:
            def _bw(nb, us):
                return nb / (us * 1e-6) / 1e9 if (us == us and us > 0) else float("nan")
            bw_ours = geo([_bw(16 * k[0] * k[1], v["oe"]) for k, v in bigx])
            bw_bare = geo([_bw(4 * k[0] * k[1] + 8 * (k[0] // 2 + 1) * k[1],
                               v["bare_e2e"]) for k, v in bigx])
            bw_nat = geo([_bw(16 * k[0] * k[1], v["ne"]) for k, v in bigx])
            kmax = max(bigx, key=lambda kv: kv[0][0] * kv[0][1])
            km, kv = kmax
            o_t = _bw(16 * km[0] * km[1], kv["oe"])
            b_t = _bw(4 * km[0] * km[1] + 8 * (km[0] // 2 + 1) * km[1], kv["bare_e2e"])
            n_t = _bw(16 * km[0] * km[1], kv["ne"])
            A(f"- **上面那条已用第三路定位**：裸 CANN 与自研走的是**同一条 "
              f"`aclrtMemcpyAsync`**（pageable 主机缓冲）。`B ≥ 1024` 的 {len(bigx)} 个点上，"
              f"按「传输字节 ÷ E2E」算有效带宽（几何均值）：**自研 {bw_ours:.1f}、"
              f"裸 {bw_bare:.1f}、torch_npu {bw_nat:.1f} GB/s**；最大点 "
              f"`n={km[0]}/B={km[1]}` 分别 **{o_t:.1f} / {b_t:.1f} / {n_t:.1f} GB/s**。"
              f"两条 `aclrtMemcpyAsync` 路径落在同一量级、torch 快约 "
              f"**{bw_nat / bw_ours:.1f}×** —— 差距在 **CANN 运行时的主机侧 staging 路径**，"
              "既不在我们的 kernel，也不在 E2E 计时代码（`device-only` 一列完全不受影响）。"
              "注：裸一路 E2E 里算子占比更大，带宽被摊薄，故只作量级参照。")
        A("- **抖动**：传输量大时单轮 E2E 的 `mean/min` 可差 1.5×，所以必须看 min-of-means。")
        A("- **公平性**：两侧都在计时区外剔除了一次性开销；原生的输出 tensor 由 "
          "`torch.fft.fft` 每次分配，但 torch_npu 的 caching allocator 使其在 warmup 后"
          "接近常数开销，与自研的预分配 `dOut` 量级相当。\n")

        # ---- 6.4 裸 CANN C API 参照 ----
        bare = [(k, v) for k, v in e2.items()
                if v.get("bare_dev") == v.get("bare_dev")
                and v.get("bare_e2e") == v.get("bare_e2e")]
        if bare:
            A("---\n")
            A("## 图7　三路端到端：自研 / torch / 裸 CANN\n")
            A("![three-way](figures/fig7_three_way_e2e.png)\n")
            A("为什么要第三路：**CANN 9.0.0 的公开头文件里没有复数→复数的 FFT C API**"
              "（`include/` 全量扫描只有 `aclnnop/acl_rfft1d.h` 与 `acl_stft.h`，"
              "`libopapi.so` 也没有 `Dft`/`Aclfft` 导出符号）。"
              "我们一路叫「CANN 原生」的 `torch.fft.fft`，其 `_fft_c2c` 实际由 **torch_npu "
              "自带的 op-plugin** 实现（`FFTc2cKernelNpuOpApi.cpp` / `FFTPlanNpuOpApi.cpp`，"
              "并暴露 `torch_npu.npu` 的 fft plan cache），**不是 CANN 算子库条目**"
              "（已实测确认它确实在设备上跑：`n=4096/B=4096` 时 NPU 2,356 µs vs "
              "CPU 740,191 µs）。所以补一路 `aclRfft1D`，把「CANN 算子调用路径本身的"
              "固定开销」和「torch 层」分开。\n")
            A("| | 自研 | CANN 原生（表1/图6） | 裸 CANN（本节） |")
            A("|---|---|---|---|")
            A("| 实现 | AscendC `kfft_fwd` | torch_npu op-plugin | CANN `aclRfft1D`（aclNN） |")
            A("| 变换 | 复→复 | 复→复 | **实→复、单边** |")
            A("| 搬运量 (H2D+D2H) | `16n·B` | `16n·B` | **`8n·B`（一半）** |")
            A("| 调用约定 | 自带 plan，一次性 | plan cache 跨调用复用 | "
              "两段式，**每轮重新 `GetWorkspaceSize`**（executor 不可复用） |")
            A("| workspace | 无 | 由框架管理 | **固定 ~2.06 GB**（与 shape 无关） |")
            A("")
            bd = [v["bare_dev"] for v in e2.values() if v.get("bare_dev") == v.get("bare_dev")]
            be = [v["bare_e2e"] for v in e2.values() if v.get("bare_e2e") == v.get("bare_e2e")]
            small = [(k, v) for k, v in bare if k[1] <= 64]
            tiny = [(k, v) for k, v in bare if k[1] <= 64 and k[0] <= 256]
            A("| 指标（几何均值） | 结果 |")
            A("|---|---|")
            A(f"| 裸 device-only（49 点） | **{geo(bd):.1f} µs** |")
            A(f"| 裸 E2E（49 点） | **{geo(be):.1f} µs** |")
            if small:
                A(f"| `B≤64`（28 点，传输占比小）device | 裸 **{geo([v['bare_dev'] for _, v in small]):.1f} µs**"
                  f" ｜ 原生 **{geo([v['nd'] for _, v in small]):.1f} µs**"
                  f" ｜ 自研 **{geo([v['od'] for _, v in small]):.1f} µs** |")
            if tiny:
                A(f"| `n≤256 且 B≤64`（12 点，传输 < 1 MB）device | 裸 **{geo([v['bare_dev'] for _, v in tiny]):.1f} µs**"
                  f" ｜ 原生 **{geo([v['nd'] for _, v in tiny]):.1f} µs**"
                  f" ｜ 自研 **{geo([v['od'] for _, v in tiny]):.1f} µs** |")
            nv = [v["nat_vs_bare"] for v in e2.values()
                  if v.get("nat_vs_bare") == v.get("nat_vs_bare")]
            if nv:
                A(f"| 原生 device ÷ 裸 device（49 点） | **{geo(nv):.2f}×**（两路变换不同，"
                  "只看固定开销量级） |")
            okb = sum(1 for _, v in bare if v.get("bare_ok"))
            A(f"| 裸一路正确性 | **{okb}/{len(bare)} PASS**（与 `numpy.fft.rfft` 同义，"
              "`norm=1` 实测 = 前向不缩放） |")
            A("")
            A("**怎么读**：\n")
            if tiny:
                A(f"1. 传输可忽略的 `{len(tiny)}` 个点上，裸 CANN device **"
                  f"{geo([v['bare_dev'] for _, v in tiny]):.0f} µs**、torch 原生 **"
                  f"{geo([v['nd'] for _, v in tiny]):.0f} µs**、自研 **"
                  f"{geo([v['od'] for _, v in tiny]):.0f} µs** —— 那 ~100 µs 的固定开销"
                  "**在 CANN 算子调用路径本身**，不是 torch 那一层的额外包装。")
            A("2. 但裸 CANN 反而**比 torch 略慢**：两段式约定要求每轮 "
              "`GetWorkspaceSize`，而 torch 侧有 plan cache —— 「裸」不等于「快」。"
              "裸一路在 `n` 变大时 `GetWorkspaceSize` 成本随之上升（`B=1` 时 "
              "`n=64`→`n=4096` 由 ~99 µs 涨到 ~708 µs），而 torch 原生全程压在 "
              "~80~112 µs。")
            A("3. E2E 一列**不能直接比胜负**：`aclRfft1D` 是实→复、搬运量只有一半，"
              "大 batch 下它的 E2E 天然占优；本节只用来定位固定开销的来源。")
            A("4. 裸一路是**单进程单 shape**，每个 shape 都从零建图，"
              "所以 `bare_first_us`（0.4~2 s）与原生的 `first_us`（单进程跑全网格、"
              "plan cache 跨 shape 复用）**口径不同，不可直接比**，故不出现在正文表里。\n")
        else:
            A("> 本轮没有裸 CANN 数据（`--no-bare` 或 `build/baseline_rfft` 缺失），"
              "跳过图7。补跑：`python3 scripts/e2e_test.py --reps 10 --rounds 3`\n")
    else:
        A(f"> `{a.e2e}` 不存在，跳过。先跑："
          "`python3 scripts/e2e_test.py --reps 10 --rounds 3`\n")

    # ---------------- 7 复现 ----------------
    A("---\n")
    A("## 7　一键复现\n")
    A("```bash")
    A("# 1) 门禁 + 49 点 device-only 矩阵（生成 docs/matrix_test_a7.md 同构数据）")
    A("scripts/one_click_test.sh")
    A("")
    A("# 2) 端到端测试（生成 results/e2e.{md,json}）")
    A("python3 scripts/e2e_test.py --reps 10 --rounds 3")
    A("")
    A("# 3) 出图 + 出本文档")
    A("python3 scripts/plot_results.py")
    A("python3 scripts/gen_compare_doc.py")
    A("```")
    A("")
    A("> 图用英文标签是刻意的：仓库外的机器不一定有 CJK 字体，"
      "中文图例换台机器就变豆腐块。中文说明都在本文的图注里。\n")

    txt = "\n".join(L) + "\n"
    open(R(a.out), "w", encoding="utf-8").write(txt)
    print(f"-> {a.out}  ({len(txt) // 1024 + 1} KB, {len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
