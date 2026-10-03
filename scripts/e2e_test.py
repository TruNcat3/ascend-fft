#!/usr/bin/env python3
"""端到端对比测试：H2D + 变换 + D2H，自研 kfft_fwd vs CANN 原生复数 FFT。

口径（两侧逐段对应，详见 docs/实验对比.md §4）：

  自研   :  aclrtMemcpyAsync(H2D) -> kfft_fwd -> aclrtMemcpyAsync(D2H)   （fft_check AB_E2E=1）
  CANN 原生:  xd.copy_(x_cpu)      -> torch.fft.fft(xd) -> y.cpu()        （bench_native_npu.py --e2e）

两侧都在计时区外完成：进程/框架启动、设备上下文建立、输入生成。
  * 自研 `boot_us`  = aclInit + SetDevice + CreateStream（进程级一次性）
    `plan_us`  = 旋转因子/索引生成 + 显存分配 + 首次上传 + kernel 二进制装载
  * 原生 `first_us` = 该 shape 的冷调用（CANN plan 尚未构建，含 H2D/变换/D2H）；
    进程级冷启动已在 bench_native_npu.py 里用一次极小变换提前消耗掉。

用法：
  python3 scripts/e2e_test.py [--ns ...] [--bs ...] [--reps 10] [--rounds 3]
                              [--out results/e2e.md] [--json results/e2e.json]

`--rounds K`：整套跑 K 遍、逐点取 min-of-means（与 matrix_test.py 同口径，
宿主负载 20~30 下单次 mean 会被离群点污染）。输出 markdown 表（stdout/stdout 文件）
与 JSON（供 scripts/plot_results.py 画图）。
"""
import argparse, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/local/python3.11.15/bin/python3"


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
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    rounds = max(1, a.rounds)

    ours, nat, boot, plan = {}, {}, {}, {}
    nat_ok = True

    # ---- 原生：一个进程跑完全网格（进程级冷启动只消耗一次）----
    print(f"[1/2] CANN 原生 E2E  ({len(ns)}x{len(bs)} x{rounds}) ...", file=sys.stderr)
    for _ in range(rounds):
        out = sh(f"{PY} scripts/bench_native_npu.py --ns {a.ns} --bs {a.bs} "
                 f"--reps {max(a.reps, 20)} --e2e")
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

    # ---- 自研：逐点一个进程（boot/plan 是进程级一次性，天然逐点独立）----
    print(f"[2/2] 自研 kfft_fwd E2E ({len(ns)}x{len(bs)}, reps={a.reps} x{rounds}) ...",
          file=sys.stderr)
    for n in ns:
        for b in bs:
            cmd = (f"AB_E2E={a.reps} ./build/fft_check {n} {b} {a.reps}")
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
    L.append("# 端到端对比：H2D + 变换 + D2H（自研 vs CANN 原生）\n")
    L.append(f"> 硬件 Ascend910_9382（48 AIV）；reps={a.reps}；`--rounds {rounds}` "
             f"逐点 min-of-means（与 `matrix_test.py` 同口径）。\n")
    L.append("> **口径**：`自研` = H2D → `kfft_fwd` → D2H；`原生` = "
             "`xd.copy_(x_cpu)` → `torch.fft.fft(xd)` → `y.cpu()`。两侧计时区外均已剔除 "
             "进程/框架启动与输入生成。**E2E 比值 > 1 表示自研更快**；"
             "只算 kernel 的 `device 比值` 见 `matrix_test_a7.md`。\n")
    L.append("")
    L.append("| n | batch | 自研 device | 原生 device | device 比值 | **自研 E2E** | **原生 E2E** "
             "| **E2E 比值** | 自研 plan 一次 | 原生 冷首调 | maxRel | 正确性 |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")

    rows = []
    for n in ns:
        for b in bs:
            o = ours.get((n, b), {})
            d = nat.get((n, b), {})
            od = o.get("dev_mean", float("nan"))
            nd = d.get("dev_mean", float("nan"))
            oe = o.get("e2e_mean", float("nan"))
            ne = d.get("e2e_mean", float("nan"))
            dr, er = ratio(od, nd), ratio(oe, ne)
            rel = o.get("maxRel", float("nan"))
            ok = "PASS" if (o.get("ok") and nat_ok) else "FAIL"
            L.append(f"| {n} | {b} | {fm(od)} | {fm(nd)} | {fm(dr)}x | **{fm(oe)}** "
                     f"| **{fm(ne)}** | **{fm(er)}x** | {fm(plan.get((n, b)))} "
                     f"| {fm(d.get('first_us', float('nan')))} | {rel:.3e} | {ok} |")
            rows.append({
                "n": n, "batch": b,
                "ours_dev": od, "nat_dev": nd, "dev_ratio": dr,
                "ours_e2e": oe, "nat_e2e": ne, "e2e_ratio": er,
                "ours_plan_us": plan.get((n, b), float("nan")),
                "ours_boot_us": boot.get((n, b), float("nan")),
                "nat_first_us": d.get("first_us", float("nan")),
                "maxRel": rel, "ok": bool(o.get("ok") and nat_ok),
                "xfer_share": (1 - od / oe) if (oe == oe and od == od and oe > 0) else float("nan"),
            })

    dev_w = [r for r in rows if r["dev_ratio"] == r["dev_ratio"]]
    e2e_w = [r for r in rows if r["e2e_ratio"] == r["e2e_ratio"]]
    L.append("")
    L.append("## 汇总\n")
    L.append(f"- 正确性：**{sum(1 for r in rows if r['ok'])}/{len(rows)} PASS**（maxRel ≤ 1e-4）")
    if dev_w:
        L.append(f"- **device-only**：{sum(1 for r in dev_w if r['dev_ratio'] > 1)}/{len(dev_w)} 点自研更快，"
                 f"几何均值 {pow(__import__('math').prod(r['dev_ratio'] for r in dev_w), 1/len(dev_w)):.2f}x")
    if e2e_w:
        L.append(f"- **端到端 E2E**：{sum(1 for r in e2e_w if r['e2e_ratio'] > 1)}/{len(e2e_w)} 点自研更快，"
                 f"几何均值 {pow(__import__('math').prod(r['e2e_ratio'] for r in e2e_w), 1/len(e2e_w)):.2f}x")
    worst = min(rows, key=lambda r: r["e2e_ratio"] if r["e2e_ratio"] == r["e2e_ratio"] else 9e9)
    best = max(rows, key=lambda r: r["e2e_ratio"] if r["e2e_ratio"] == r["e2e_ratio"] else -1)
    L.append(f"- E2E 最快点 **{best['e2e_ratio']:.2f}x** @ n={best['n']}/b={best['batch']}，"
             f"最慢点 **{worst['e2e_ratio']:.2f}x** @ n={worst['n']}/b={worst['batch']}")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(txt)
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump({"rows": rows,
               "meta": {"ns": ns, "bs": bs, "reps": a.reps, "rounds": rounds,
                        "generated": time.strftime("%Y-%m-%d %H:%M:%S")}},
              open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(txt)
    print(f"\n-> {a.out}\n-> {a.json}", file=sys.stderr)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
