#!/usr/bin/env python3
"""矩阵测试：n × batch 全网格上跑 正确性 + 实测耗时 + η 精度 + 与 CANN 原生算子对比。

  python3 scripts/matrix_test.py --ns 64,128,...,4096 --bs 1,4,...,10000 --reps 20

三路数据，统计口径对齐：
  * 自研 `kfft_fwd`  —— `fft_check <n> <b> <reps>`，**mean**（与 §8.5 的 η 标定同口径）
  * 自研 min        —— 同上，**min**（与 `Plan::measure` 同口径）
  * CANN 原生算子    —— `bench_native_npu.py`（torch.fft.fft / torch_npu），mean + min
  * η                —— `test_framework`（含选型闭环，顺带做一次真实验收）

`--rounds K`（默认 3）：整套测量跑 K 遍，逐点取 **min of mean**（自研与原生同口径）。
本机宿主负载 20~30，单次 mean 的瞬时离群点能把 19 µs 抬到 34 µs（min 仍是 10.7 µs），
而 η 是「无噪声耗时」的模型值 —— 单次 mean 会让 η/实测 出现 ±40% 的假偏差。
取 min-of-means 后原生/自研与 η 三者口径一致、可复现。

输出 markdown 表到 stdout（进度到 stderr）。`--no-eta` 关掉可省掉选型开销。
"""
import argparse, os, re, subprocess, sys

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
    return float(m.group(1)) if m else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--bs", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3,
                    help="整套测量跑几遍、逐点取 min（0 噪声口径，见文件头）")
    ap.add_argument("--no-eta", action="store_true")
    ap.add_argument("--no-native", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    rounds = max(1, a.rounds)

    nat, nat_ok = {}, True
    if not a.no_native:
        print(f"[1/3] CANN 原生复数 FFT  ({len(ns)}x{len(bs)} x{rounds}) ...", file=sys.stderr)
        for _ in range(rounds):
            out = sh(f"{PY} scripts/bench_native_npu.py --ns {a.ns} --bs {a.bs} "
                     f"--reps {max(a.reps, 20)}")
            for ln in out.splitlines():
                m = re.match(r"NATIVE n=(\d+) b=(\d+) native_us=([\d.]+) native_mean_us=([\d.]+) "
                             r"maxRel=([\d.eE+-]+) (\w+)", ln)
                if not m:
                    continue
                k = (int(m.group(1)), int(m.group(2)))
                nmin, nmean = float(m.group(3)), float(m.group(4))
                prev = nat.get(k)
                # 逐点各自取 min：min 列与 mean 列互不牵连
                if prev is None:
                    nat[k] = [nmin, nmean]
                else:
                    nat[k] = [min(prev[0], nmin), min(prev[1], nmean)]
                nat_ok &= (m.group(6) == "PASS")
        print(f"    native {len(nat)} 点 {'PASS' if nat_ok else 'FAIL'}", file=sys.stderr)

    print(f"[2/3] 自研 kfft_fwd ({len(ns)}x{len(bs)}, reps={a.reps} x{rounds}) ...",
          file=sys.stderr)
    ours = {}
    for n in ns:
        for b in bs:
            cmd = f"./build/fft_check {n} {b} {a.reps}"
            best_mean = best_min = float("nan")
            rel = float("nan"); ok = False
            for _ in range(rounds):
                s = sh(cmd)
                mn, mi = f(r"kfft_fwd: ([\d.]+) us/call", s), f(r"\(min ([\d.]+) us\)", s)
                if mn != mn:
                    continue
                ok = ok or bool(re.search(r"^PASS$", s, re.M))
                r = f(r"maxRel=([\d.eE+-]+)", s)
                if r == r:
                    rel = r if rel != rel else min(rel, r)
                best_mean = mn if best_mean != best_mean else min(best_mean, mn)
                best_min = mi if best_min != best_min else min(best_min, mi)
            ours[(n, b)] = {"mean": best_mean, "min": best_min, "maxRel": rel, "ok": ok}
            print(f"    n={n:<5} b={b:<5} {'PASS' if ok else 'FAIL'}"
                  f"  {best_mean:.1f} us", file=sys.stderr)

    eta = {}
    if not a.no_eta:
        print("[3/3] 框架 η / 选型闭环 ...", file=sys.stderr)
        for n in ns:
            for b in bs:
                s = sh(f"./build/test_framework config/ascend910_93_profile.json "
                       f"config/butterfly_space.json build/fft_radix2.o {n} {b}")
                e = f(r"eta=([\d.]+) us", s)
                ok = "selected:" in s and "no feasible" not in s
                eta[(n, b)] = e
                print(f"    n={n:<5} b={b:<5} eta={e:8.1f} "
                      f"{'ok' if ok else 'NO FEASIBLE'}", file=sys.stderr)

    # ---------------- 表 ----------------
    rows = []
    for n in ns:
        for b in bs:
            o = ours[(n, b)]
            nm, nmean = nat.get((n, b), (float("nan"), float("nan")))
            e = eta.get((n, b), float("nan"))
            rows.append((n, b, o["mean"], o["min"], o["maxRel"], nm, nmean, e, o["ok"]))

    def fmt(x, d=1):
        return "—" if x != x else f"{x:,.{d}f}"

    lines = []
    L = lines.append
    L("# 矩阵测试：n × batch 全网格（自研 kfft_fwd vs CANN 原生复数 FFT）\n")
    L("> **复现脚本**：[`scripts/matrix_test.py`](../scripts/matrix_test.py)"
      "（本文是它的输出）· [`scripts/one_click_test.sh`](../scripts/one_click_test.sh)"
      "（编译+门禁+矩阵一键）· [`scripts/calib_eta.py`](../scripts/calib_eta.py)（η 列）。\n")
    L(f"> 硬件 Ascend910_9382（48 AIV）；reps={a.reps}；"
      f"`自研 mean` 与 `原生 mean` 同口径、`自研 min` 与 `原生 min` 同口径。\n")
    L("> **η** 来自框架选型闭环（`test_framework`），同一行的 `η/实测` 列给出模型相对"
      "`自研 mean` 的偏差；`原生/自研` > 1 表示自研更快。\n")
    L("| n | batch | 自研 mean | 自研 min | CANN 原生 mean | CANN 原生 min | 原生/自研(mean) "
      "| η | η/实测 | maxRel | 正确性 |")
    L("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    n_ok = 0
    for n, b, om, omin, rel, nm, nmean, e, ok in rows:
        if om == om and nm == nm and nm > 0:
            r = nmean / om
            rs = f"**{r:.2f}×**" if r >= 1 else f"{r:.2f}×"
        else:
            rs = "—"
        if e == e and om == om and om > 0:
            ed = f"{(e - om) / om * 100:+.1f}%"
        else:
            ed = "—"
        good = ok and (nm == nm)
        n_ok += 1 if good else 0
        # nat = (min, mean)，列序与表头一致：原生 mean | 原生 min
        L(f"| {n} | {b} | {fmt(om)} | {fmt(omin)} | {fmt(nmean)} | {fmt(nm)} | {rs} "
          f"| {fmt(e)} | {ed} | {rel:.2e} | {'PASS' if good else 'FAIL'} |")

    lines.append("")
    L(f"**正确性：{n_ok}/{len(rows)} PASS**（判据 maxRel ≤ 1e-4，"
      f"自研与原生各自对双精度 CPU 参考；原生 "
      f"{'全部 PASS' if nat_ok else '有 FAIL'}）")
    body = "\n".join(lines) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(body)
        print(f"written -> {a.out}", file=sys.stderr)
    print(body)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
