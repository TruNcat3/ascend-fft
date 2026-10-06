#!/usr/bin/env python3
"""端到端对比测试：H2D + 变换 + D2H，自研 kfft_fwd vs CANN 原生复数 FFT vs 裸 CANN C API。

口径（三路逐段对应，详见 docs/实验对比.md §6）：

  自研   :  aclrtMemcpyAsync(H2D) -> kfft_fwd -> aclrtMemcpyAsync(D2H)   （fft_check AB_E2E=1）
  CANN 原生:  xd.copy_(x_cpu)      -> torch.fft.fft(xd) -> yd.copy_(out)  （bench_native_npu.py --e2e）
  裸 CANN :  aclrtMemcpyAsync(H2D) -> aclRfft1D -> aclrtMemcpyAsync(D2H)  （baseline_rfft --e2e）

三路的主机缓冲默认都是 **pinned**（`AB_E2E_HOST`，见 docs/实验对比.md §6.3）：
自研/裸用 `aclrtMallocHost`，torch 用 `.pin_memory()` 源张量 + 预分配 pinned 目的张量；
`AB_E2E_HOST=pageable` 切回受限口径做 A/B。

第三路是**唯一可用的裸 CANN C API FFT**：CANN 9.0.0 公开头里只有 `aclRfft1D`（实->复）
与 `aclSTFT`，**没有复数->复数的 C API**，所以它与前两路变换不同（实->复、单边），
**仅作参照、不参与胜负统计**。它的作用是把「CANN 算子调用路径本身的固定开销」
与「torch 调度 + op-plugin 内核」分开 —— torch_npu 的 `_fft_c2c` 由其自带
op-plugin 实现（`FFTc2cKernelNpuOpApi.cpp`），并非 CANN 算子库条目。

两侧都在计时区外完成：进程/框架启动、设备上下文建立、输入生成。
  * 自研 `boot_us`  = aclInit + SetDevice + CreateStream（进程级一次性）
    `plan_us`  = 旋转因子/索引生成 + 显存分配 + 首次上传 + kernel 二进制装载
  * 原生 `first_us` = 该 shape 的冷调用（CANN plan 尚未构建，含 H2D/变换/D2H）；
    进程级冷启动已在 bench_native_npu.py 里用一次极小变换提前消耗掉。
  * 裸 CANN 同理：`boot_us`/`setup_us` 计时区外，进程级冷启动由 --e2e 内部的
    极小变换消耗；`first_us` = 该 shape 首次 GetWorkspaceSize + 首次执行。

用法：
  python3 scripts/e2e_test.py [--ns ...] [--bs ...] [--reps 10] [--rounds 3]
                              [--out results/e2e.md] [--json results/e2e.json] [--no-bare]

`--rounds K`：整套跑 K 遍、逐点取 min-of-means（与 matrix_test.py 同口径，
宿主负载 20~30 下单次 mean 会被离群点污染）。markdown 表打到 stdout 并同时写入
`--out` 文件；JSON 写 `--json` 文件（供 scripts/plot_results.py 画图）。
进度行（`[1/3] ...`、`    n=.. PASS ..`）与结尾的 `-> <out>` / `-> <json>` 两行打到 stderr。
"""
import argparse, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abenv  # noqa: E402  与 scripts/env.sh 共用同一套路径探测
PY = abenv.python_bin()


def sh(cmd, env=None, cwd=ROOT, timeout=3600):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=e,
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def f(pat, s, d=float("nan")):
    m = re.search(pat, s)
    return float(m.group(1).replace(",", "")) if m else d


def minof(prev, cur):
    if prev is None:
        return cur
    if cur != cur:
        return prev
    if prev != prev:
        return cur
    return min(prev, cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--bs", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="results/e2e.md")
    ap.add_argument("--json", default="results/e2e.json")
    ap.add_argument("--no-bare", action="store_true",
                    help="跳过裸 CANN aclRfft1D 端到端（需 build/baseline_rfft）")
    ap.add_argument("--app", default="", choices=["", "ofdm", "radar", "dl", "all"],
                    help="应用负载模式：按典型应用的 (n,batch) 组合 + 应用形状输入"
                         "（fft_check AB_INPUT / bench_native_npu --input，两者逐位同式）"
                         "重跑三方端到端；输出默认 results/e2e_app.{md,json}，"
                         "并跳过实->复的裸 CANN 一路（变换不同，不参与胜负）")
    a = ap.parse_args()

    # 典型应用的代表形状（都是既有网格里的点：n 为 2 的幂、batch 任意）
    #   ofdm  4G LTE 20 MHz 的 FFT 点数，一子帧 14 符号 / 一帧 140 符号
    #   radar 一帧 64 个脉冲（×4 接收通道 = 256），距离门数取 1024/2048
    #   dl    频域层的复激活，batch = 样本 × 通道
    APPS = {
        "ofdm":  ("2048,4096", "14,140",
                  "多载波通信 / 频域均衡（16-QAM 子载波，DC/保护带置零）"),
        "radar": ("1024,2048", "64,256",
                  "雷达成像距离门（4 目标复指数回波 + 逐脉冲多普勒相位）"),
        "dl":    ("1024,4096", "32,128",
                  "深度学习频域层（每 8 点一块的复激活）"),
    }
    runs, inp = [], {}            # runs: (app, ns_str, bs_str)；inp: (n,b) -> 应用输入模式
    if a.app:
        for nm in (["ofdm", "radar", "dl"] if a.app == "all" else [a.app]):
            ns_s, bs_s, _desc = APPS[nm]
            runs.append((nm, ns_s, bs_s))
            for n in [int(x) for x in ns_s.split(",")]:
                for b in [int(x) for x in bs_s.split(",")]:
                    inp[(n, b)] = nm
        if a.out == "results/e2e.md":
            a.out = "results/e2e_app.md"
        if a.json == "results/e2e.json":
            a.json = "results/e2e_app.json"
        a.no_bare = True          # 裸一路是实->复、搬运量减半，与这三类应用不可比
    else:
        runs = [("", a.ns, a.bs)]
    pairs = [(int(n), int(b)) for _, ns_s, bs_s in runs
             for n in ns_s.split(",") for b in bs_s.split(",")]
    if len(set(pairs)) != len(pairs):
        # shape 跨应用重叠会让三方记录（按 (n,b) 键）互相污染，必须唯一
        print(f"ERR: shape 重复：{pairs}", file=sys.stderr)
        return 2
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    rounds = max(1, a.rounds)

    ours, nat, bare, boot, plan = {}, {}, {}, {}, {}
    nat_ok = True
    bare_ok = True

    # ---- 原生：一个进程跑完全网格（进程级冷启动只消耗一次）----
    what = (f"{len(runs)} 个应用 × {len(pairs)} 个 shape"
            if a.app else f"{len(ns)}x{len(bs)}")
    print(f"[1/3] CANN 原生 E2E  ({what} x{rounds}) ...", file=sys.stderr)
    for _ in range(rounds):
        for nm, ns_s, bs_s in runs:
            out = sh(f"{PY} scripts/bench_native_npu.py --ns {ns_s} --bs {bs_s} "
                     f"--reps {max(a.reps, 20)} --e2e"
                     + (f" --input {nm}" if nm else ""))
            for ln in out.splitlines():
                m = re.match(r"NATIVE n=(\d+) b=(\d+) native_us=([\d.]+) native_mean_us=([\d.]+) "
                             r"maxRel=([\d.eE+-]+) (\w+)", ln)
                if m:
                    k = (int(m.group(1)), int(m.group(2)))
                    d = nat.setdefault(k, {})
                    d["dev_mean"] = minof(d.get("dev_mean"), float(m.group(4)))
                    d["dev_min"] = minof(d.get("dev_min"), float(m.group(3)))
                    nat_ok &= (m.group(6) == "PASS")
                m = re.match(r"NATIVE_E2E n=(\d+) b=(\d+) e2e_us=([\d.]+) e2e_mean_us=([\d.]+) "
                             r"first_us=([\d.]+) maxRel=([\d.eE+-]+) (\w+)", ln)
                if m:
                    k = (int(m.group(1)), int(m.group(2)))
                    d = nat.setdefault(k, {})
                    d["e2e_min"] = minof(d.get("e2e_min"), float(m.group(3)))
                    d["e2e_mean"] = minof(d.get("e2e_mean"), float(m.group(4)))
                    d["first_us"] = minof(d.get("first_us"), float(m.group(5)))
    print(f"    native {len(nat)} 点 {'PASS' if nat_ok else 'FAIL'}", file=sys.stderr)

    # ---- 裸 CANN C API：逐点一个进程（每进程自带 boot + 极小变换冷启动）----
    bare_bin = os.path.join(ROOT, "build", "baseline_rfft")
    run_bare = (not a.no_bare) and os.path.isfile(bare_bin)
    if not a.no_bare and not run_bare:
        print("    [warn] build/baseline_rfft 不存在，跳过裸 CANN 一路（make rfft）",
              file=sys.stderr)
    if run_bare:
        print(f"[2/3] 裸 CANN aclRfft1D E2E ({len(ns)}x{len(bs)} x{rounds}) ...",
              file=sys.stderr)
        pat = (r"BARE n=(\d+) b=(\d+) dev_us=([\d.]+) dev_mean_us=([\d.]+) "
               r"e2e_us=([\d.]+) e2e_mean_us=([\d.]+) first_us=([\d.]+) "
               r"boot_us=([\d.]+) setup_us=([\d.]+) ws_mb=([\d.]+) "
               r"maxRel=([\d.eE+-]+) (\w+)")
        for n, b in pairs:
                cmd = (f"./build/baseline_rfft {n} {b} 1 {abenv.work()}/ab_bare.bin "
                       f"--e2e --reps={max(a.reps, 10)}")
                d = bare.setdefault((n, b), {})
                for _ in range(rounds):
                    m = re.search(pat, sh(cmd, timeout=1800))
                    if not m:
                        continue
                    d["dev_min"] = minof(d.get("dev_min"), float(m.group(3)))
                    d["dev_mean"] = minof(d.get("dev_mean"), float(m.group(4)))
                    d["e2e_min"] = minof(d.get("e2e_min"), float(m.group(5)))
                    d["e2e_mean"] = minof(d.get("e2e_mean"), float(m.group(6)))
                    d["first_us"] = minof(d.get("first_us"), float(m.group(7)))
                    r = float(m.group(11))
                    d["maxRel"] = min(d.get("maxRel", float("inf")), r)
                    d["ok"] = bool(d.get("ok", False) or m.group(12) == "PASS")
                bare_ok &= bool(d.get("ok"))
                print(f"    n={n:<5} b={b:<5} {'PASS' if d.get('ok') else 'FAIL'}  "
                      f"dev={d.get('dev_mean', float('nan')):.1f}  "
                      f"e2e={d.get('e2e_mean', float('nan')):.1f} us", file=sys.stderr)
    else:
        print(f"[2/3] 裸 CANN aclRfft1D E2E — 跳过", file=sys.stderr)

    # ---- 自研：逐点一个进程（boot/plan 是进程级一次性，天然逐点独立）----
    print(f"[3/3] 自研 kfft_fwd E2E ({what}, reps={a.reps} x{rounds}) ...",
          file=sys.stderr)
    for n, b in pairs:
            cmd = (f"{('AB_INPUT=' + inp[(n, b)] + ' ') if (n, b) in inp else ''}"
                   f"AB_E2E={a.reps} ./build/fft_check {n} {b} {a.reps}")
            d = ours.setdefault((n, b), {})
            for _ in range(rounds):
                s = sh(cmd)
                d["dev_mean"] = minof(d.get("dev_mean"), f(r"kfft_fwd: ([\d.]+) us/call", s))
                d["dev_min"] = minof(d.get("dev_min"), f(r"\(min ([\d.]+) us\)", s))
                d["e2e_mean"] = minof(d.get("e2e_mean"), f(r"e2e_us=([\d.]+)", s))
                d["e2e_min"] = minof(d.get("e2e_min"), f(r"e2e_min_us=([\d.]+)", s))
                d["first_us"] = minof(d.get("first_us"), f(r"first_us=([\d.]+)", s))
                boot[(n, b)] = minof(boot.get((n, b)), f(r"boot_us=([\d.]+)", s))
                plan[(n, b)] = minof(plan.get((n, b)), f(r"plan_us=([\d.]+)", s))
                r = f(r"maxRel=([\d.eE+-]+)", s)
                if r == r:
                    d["maxRel"] = min(d.get("maxRel", float("inf")), r)
                if re.search(r"^PASS$", s, re.M):
                    d["ok"] = True
                d.setdefault("ok", False)
            print(f"    n={n:<5} b={b:<5} {'PASS' if d.get('ok') else 'FAIL'}  "
                  f"dev={d.get('dev_mean', float('nan')):.1f}  "
                  f"e2e={d.get('e2e_mean', float('nan')):.1f} us", file=sys.stderr)

    # ---------------- 表 ----------------
    def fm(x, d=1):
        return "—" if x != x else f"{x:,.{d}f}"

    def ratio(o, nn):
        try:
            if o != o or nn != nn or o <= 0:
                return float("nan")
            return nn / o
        except Exception:
            return float("nan")

    lines, L = [], []
    L.append(("#应用负载端到端对比：H2D + 变换 + D2H（自研 vs CANN 原生）\n"
              if a.app else
              "# 端到端对比：H2D + 变换 + D2H（自研 vs CANN 原生 vs 裸 CANN C API）\n"))
    L.append(f"> 硬件 Ascend910_9382（48 AIV）；reps={a.reps}；`--rounds {rounds}` "
             f"逐点 min-of-means（与 `matrix_test.py` 同口径）。\n")
    if a.app:
        names = ["ofdm", "radar", "dl"] if a.app == "all" else [a.app]
        L.append("> **应用负载模式** `--app " + a.app + "`：三方都喂**应用形状的输入**"
                 "（自研 `AB_INPUT=<app>`、原生 `--input <app>`，C++ 与 numpy 同一套 "
                 "xorshift32、**逐位同式**），仍然与各自双精度参考逐点比对（判据 1e-4）。\n")
        for nm in names:
            ns_s, bs_s, desc = APPS[nm]
            L.append(f"> * **{nm}** —— {desc}；形状 n ∈ {{{ns_s}}} × batch ∈ {{{bs_s}}}，"
                     f"搬运量 16·n·batch 字节。\n")
        L.append("> \n")
    L.append("**口径**（三路逐段对应，计时区外均剔除进程/框架启动与输入生成）：\n"
             "1. `自研` = H2D → `kfft_fwd` → D2H（复数→复数，`fft_check AB_E2E=1`）\n"
             "2. `原生` = `xd.copy_(x_cpu)` → `torch.fft.fft(xd)` → `yd.copy_(out)`（源/目的均 pinned）"
             "（复数→复数，torch_npu op-plugin 内核）\n")
    if a.app:
        L.append("3. `裸 CANN aclRfft1D` —— **应用负载模式下跳过**（实→复、搬运量减半，"
                 "与这三类应用不可比；网格口径的三方对比见 `results/e2e.md`）\n")
    else:
        L.append("3. `裸 CANN` = H2D → `aclRfft1D` → D2H（**实→复、单边**，唯一可用的裸 "
                 "CANN C API FFT —— CANN 9.0.0 没有复数→复数 C API）\n")
    L.append(" \n"
             "**表1 的 `E2E 比值 > 1` 表示自研更快**；只算 kernel 的 `device 比值` "
             "见 `matrix_test_a7.md`。")
    if a.app:
        L.append("\n")
    else:
        L.append("**表2 仅作参照、不参与胜负统计**：\n"
                 "* 变换不同（实→复 vs 复→复），且 **搬运量正好是一半**"
                 "（`8n·B` vs `16n·B` 字节），大 batch 的 E2E 不可直接比；\n"
                 "* 它的价值在 **device-only**：与传输无关，用来把「CANN 算子调用路径的"
                 "固定开销」和「torch 调度 + op-plugin」分开。\n")
    L.append("")

    # ---------------- 表1：自研 vs 原生（同算法、同搬运量）----------------
    L.append("## 表1　自研 vs torch_npu（复数→复数，同算法同搬运量）\n")
    L.append("| n | batch | 自研 device | 原生 device | device 比值 | **自研 E2E** | **原生 E2E** "
             "| **E2E 比值** | 自研 plan 一次 | 原生 冷首调 | maxRel | 正确性 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")

    rows = []
    for n, b in pairs:
            o = ours.get((n, b), {})
            d = nat.get((n, b), {})
            br = bare.get((n, b), {})
            od = o.get("dev_mean", float("nan"))
            nd = d.get("dev_mean", float("nan"))
            oe = o.get("e2e_mean", float("nan"))
            ne = d.get("e2e_mean", float("nan"))
            bd = br.get("dev_mean", float("nan"))
            be = br.get("e2e_mean", float("nan"))
            dr, er = ratio(od, nd), ratio(oe, ne)
            rel = o.get("maxRel", float("nan"))
            ok = "PASS" if (o.get("ok") and nat_ok and (bare_ok or not run_bare)) else "FAIL"
            L.append(f"| {n} | {b} | {fm(od)} | {fm(nd)} | {fm(dr)}x | **{fm(oe)}** "
                     f"| **{fm(ne)}** | **{fm(er)}x** | {fm(plan.get((n, b)))} "
                     f"| {fm(d.get('first_us', float('nan')))} | {rel:.3e} | {ok} |")
            rows.append({
                "n": n, "batch": b,
                "app": inp.get((n, b)),
                "ours_dev": od, "nat_dev": nd, "dev_ratio": dr,
                "ours_e2e": oe, "nat_e2e": ne, "e2e_ratio": er,
                "ours_plan_us": plan.get((n, b), float("nan")),
                "ours_boot_us": boot.get((n, b), float("nan")),
                "nat_first_us": d.get("first_us", float("nan")),
                "maxRel": rel, "ok": bool(o.get("ok") and nat_ok and (bare_ok or not run_bare)),
                "xfer_share": (1 - od / oe) if (oe == oe and od == od and oe > 0) else float("nan"),
                # ---- 裸 CANN aclRfft1D（实->复，仅参照）----
                "bare_dev": bd,
                "bare_dev_mean": br.get("dev_min", float("nan")),
                "bare_e2e": be,
                "bare_e2e_min": br.get("e2e_min", float("nan")),
                "bare_first_us": br.get("first_us", float("nan")),
                "bare_maxRel": br.get("maxRel", float("nan")),
                "bare_ok": bool(br.get("ok", False)),
                "nat_vs_bare_dev": ratio(bd, nd),
                "ours_vs_bare_e2e": ratio(be, oe),
                # 搬运量（字节，H2D + D2H）：c2c 为 16n·B；rfft 为 4n·B + 8(n/2+1)·B
                "xfer_bytes_c2c": 16 * n * b,
                "xfer_bytes_rfft": 4 * n * b + 8 * (n // 2 + 1) * b,
            })

    # ---------------- 表2：裸 CANN C API 参照 ----------------
    if run_bare:
        def fsize(v):
            if v != v:
                return "—"
            if v < 1024:
                return f"{v:.0f} B"
            if v < 1048576:
                return f"{v/1024:.1f} KB"
            return f"{v/1048576:.1f} MB"

        L.append("")
        L.append("## 表2　裸 CANN C API 参照（`aclRfft1D`，实→复、单边，**仅参照**）\n")
        L.append("| n | batch | 裸 device | 裸 E2E | 裸搬运量 | maxRel | 正确性 |")
        L.append("|---:|---:|---:|---:|---:|---:|:---|")
        for r in rows:
            ok = "PASS" if r["bare_ok"] else "FAIL"
            L.append(f"| {r['n']} | {r['batch']} | {fm(r['bare_dev'])} | **{fm(r['bare_e2e'])}** "
                     f"| {fsize(r['xfer_bytes_rfft'])} | {r['bare_maxRel']:.3e} | {ok} |")
        L.append("")
        L.append("> `裸 冷首调` 见 JSON 的 `bare_first_us`：裸一路是**单进程单 shape**，"
                 "每个 shape 都要从头建图，与 `原生 冷首调`（单进程跑完 49 点、plan cache "
                 "跨 shape 复用）**口径不同，不可直接比**。")

    dev_w = [r for r in rows if r["dev_ratio"] == r["dev_ratio"]]
    e2e_w = [r for r in rows if r["e2e_ratio"] == r["e2e_ratio"]]
    bare_w = [r for r in rows if r["bare_dev"] == r["bare_dev"] and r["bare_e2e"] == r["bare_e2e"]]

    def geo(vals):
        import math
        vals = [v for v in vals if v == v and v > 0]
        return pow(math.prod(vals), 1 / len(vals)) if vals else float("nan")

    L.append("")
    L.append("## 汇总\n")
    L.append(f"- 正确性：**{sum(1 for r in rows if r['ok'])}/{len(rows)} PASS**（maxRel ≤ 1e-4）")
    if dev_w:
        L.append(f"- **device-only**：{sum(1 for r in dev_w if r['dev_ratio'] > 1)}/{len(dev_w)} 点自研更快，"
                 f"几何均值 {geo([r['dev_ratio'] for r in dev_w]):.2f}x")
    if e2e_w:
        L.append(f"- **端到端 E2E**：{sum(1 for r in e2e_w if r['e2e_ratio'] > 1)}/{len(e2e_w)} 点自研更快，"
                 f"几何均值 {geo([r['e2e_ratio'] for r in e2e_w]):.2f}x")
    if bare_w:
        L.append(f"- **裸 CANN `aclRfft1D`**（实→复，仅参照，{len(bare_w)}/{len(rows)} 点）："
                 f"device 几何均值 **{geo([r['bare_dev'] for r in bare_w]):.1f} µs**、"
                 f"E2E 几何均值 **{geo([r['bare_e2e'] for r in bare_w]):.1f} µs**；"
                 f"`原生 device ÷ 裸 device` 几何均值 **{geo([r['nat_vs_bare_dev'] for r in bare_w]):.2f}x**"
                 f"（两路变换不同，只看固定开销量级）")
        small = [r for r in bare_w if r["batch"] <= 64]
        if small:
            L.append(f"  - 小 batch（`B ≤ 64`，传输可忽略）：裸 device "
                     f"**{geo([r['bare_dev'] for r in small]):.1f} µs** vs 原生 "
                     f"**{geo([r['nat_dev'] for r in small]):.1f} µs** vs 自研 "
                     f"**{geo([r['ours_dev'] for r in small]):.1f} µs** —— 固定开销在 CANN 调用"
                     f"路径本身，不是 torch 层")
    if e2e_w:
        worst = min(rows, key=lambda r: r["e2e_ratio"] if r["e2e_ratio"] == r["e2e_ratio"] else 9e9)
        best = max(rows, key=lambda r: r["e2e_ratio"] if r["e2e_ratio"] == r["e2e_ratio"] else -1)
        L.append(f"- E2E 最快点 **{best['e2e_ratio']:.2f}x** @ n={best['n']}/b={best['batch']}，"
                 f"最慢点 **{worst['e2e_ratio']:.2f}x** @ n={worst['n']}/b={worst['batch']}")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(txt)
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump({"rows": rows,
               "meta": {"ns": ns, "bs": bs, "app": a.app or None,
                        "shapes": pairs, "reps": a.reps, "rounds": rounds,
                        "bare_run": bool(run_bare),
                        "generated": time.strftime("%Y-%m-%d %H:%M:%S")}},
              open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(txt)
    print(f"\n-> {a.out}\n-> {a.json}", file=sys.stderr)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
