#!/usr/bin/env python3
"""汇总一份（或多份）msprof profile：算子耗时、管线占用率、设备任务时长。

数据源（由 scripts/profile_test.sh 采集）：
  <dir>/**/mindstudio_profiler_output/op_summary_*.csv   每 op 耗时 + 占空比列
  <dir>/**/mindstudio_profiler_output/api_statistic_*.csv host 侧 API 耗时（可选）
  <dir>/**/device_0/sqlite/ascend_task.db                 AscendTask.duration（纳秒）

  python3 scripts/sum_prof.py results/profiles/p_b4k
  python3 scripts/sum_prof.py results/profiles/p_a6_before results/profiles/p_a6_after
  python3 scripts/sum_prof.py results/profiles/p_nat --per 13 --drop 13
  python3 scripts/sum_prof.py results/profiles/* --csv      # 机器可读

输出的 aiv_vec_ratio / aiv_scalar_ratio / aiv_mte2_ratio / aiv_mte3_ratio /
cube_utilization(%) 就是 docs/trace与profile诊断-*.md 与 docs/性能优化-*.md 里
那些占用率表格的出处。
"""
import argparse
import csv
import glob
import os
import sqlite3
import sys
from collections import Counter, defaultdict

RATIO_COLS = [
    ("aiv_vec_ratio", "vec"), ("aiv_scalar_ratio", "sclr"),
    ("aiv_mte2_ratio", "mte2"), ("aiv_mte3_ratio", "mte3"),
    ("aic_scalar_ratio", "asc"), ("aic_mte2_ratio", "amte2"),
    ("cube_utilization(%)", "cube%"),
]


def find_one(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return sorted(hits)[0] if hits else None


def detect_period(names):
    """Op Name 序列的最小周期 = 每次迭代的算子个数。检测不到返回 1。"""
    for p in range(1, len(names) // 2 + 1):
        if all(names[i] == names[i % p] for i in range(len(names))):
            return p
    return 1


def load(d):
    d = os.path.abspath(d)
    csvp = find_one(d, "op_summary*.csv")
    if not csvp:
        return None
    rows = list(csv.DictReader(open(csvp, newline="", encoding="utf-8")))
    for r in rows:
        for k in ("Task Start Time(us)", "Task Duration(us)"):
            if k in r and r[k]:
                r[k] = float(str(r[k]).strip())
    rows.sort(key=lambda r: r.get("Task Start Time(us)", 0.0))
    return {"dir": d, "csv": csvp, "rows": rows,
            "db": find_one(d, "ascend_task.db"),
            "api": find_one(d, "api_statistic*.csv")}


def fnum(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return 0.0


def summarize(p, drop, per):
    rows = [r for r in p["rows"] if r.get("Task Duration(us)", 0) > 0]
    names = [r["Op Name"] for r in rows]
    if per <= 0:
        per = detect_period(names[drop:]) if len(rows) > drop else 1
    tail = rows[drop:]
    usable = len(tail) // per * per
    tail = tail[:usable]
    if not tail:
        return None

    cnt = Counter(r["Op Name"] for r in tail)
    dur = defaultdict(list)
    for r in tail:
        dur[r["Op Name"]].append(r.get("Task Duration(us)", 0.0))
    niter = len(tail) // per
    span = (max(r.get("Task Start Time(us)", 0) + r.get("Task Duration(us)", 0) for r in tail)
            - min(r.get("Task Start Time(us)", 0) for r in tail))

    dev = None
    if p["db"]:
        try:
            c = sqlite3.connect(p["db"])
            ds = sorted((fnum(r[0]) / 1000.0 for r in c.execute(
                "select duration from AscendTask where duration > 1000")),
                reverse=True)
            c.close()
            if ds:
                # 一次迭代往往拆成多个 task（原生尤其如此），按「每迭代取最大的
                # niter 个 task」对齐 op_summary 的迭代口径，避免被琐碎 task 拉低均值。
                top = ds[:max(1, niter)]
                dev = (len(ds), sum(top) / len(top), top[-1], ds[0])
        except sqlite3.Error:
            dev = None

    return {"p": p, "per": per, "drop": drop, "niter": niter, "rows": tail,
            "cnt": cnt, "dur": dur, "span_us": span / max(1, niter),
            "dev_sum_us": sum(sum(v) for v in dur.values()) / max(1, niter),
            "dev": dev}


def op_rows(s):
    for name, ds in sorted(s["dur"].items(), key=lambda kv: -sum(kv[1])):
        yield name, s["cnt"][name], sum(ds) / s["niter"], min(ds), max(ds)


def ratios(s, name):
    sel = [r for r in s["rows"] if r["Op Name"] == name]
    out = []
    for col, short in RATIO_COLS:
        vals = [fnum(r.get(col)) for r in sel if r.get(col) not in (None, "", "N/A")]
        vals = [v for v in vals if v > 0]
        out.append((short, sum(vals) / len(vals) if vals else None))
    return out


def print_one(s, as_csv=False):
    p = s["p"]
    if as_csv:
        for name, c, mean, lo, hi in op_rows(s):
            rr = ",".join(f"{k}={v:.3f}" for k, v in ratios(s, name) if v is not None)
            print(f"{name},{c},{mean:.3f},{lo:.3f},{hi:.3f},{rr}")
        return
    print(f"=== {os.path.relpath(p['dir'])} ===")
    print(f"  op_summary : {os.path.relpath(p['csv'])}  rows={len(p['rows'])} "
          f"dropped={s['drop']} per_iter={s['per']} meas_iter={s['niter']}")
    for name, c, mean, lo, hi in op_rows(s):
        rs = "  ".join(f"{k}={v:.3f}" for k, v in ratios(s, name) if v is not None)
        print(f"   {c:3d} x {name:<44s} mean={mean:8.3f}  min={lo:8.3f} max={hi:8.3f}")
        if rs:
            print(f"       {rs}")
    print(f"   device sum/iter = {s['dev_sum_us']:.2f} us   "
          f"wall span/iter = {s['span_us']:.2f} us")
    if s["dev"]:
        n, mean, lo, hi = s["dev"]
        print(f"   AscendTask: 全部 {n} 个 task，按迭代取最大 {s['niter']} 个 → "
              f"mean={mean:.2f} us  min={lo:.2f} us  max={hi:.2f} us"
              f"（duration 字段单位为 ns，已 /1000）")
    if p["api"]:
        print(f"   api_statistic: {os.path.relpath(p['api'])}")
    print()


def print_compare(ss, as_csv=False):
    if as_csv:
        print("dir,op,iter,count,mean_us,min_us,max_us")
        for s in ss:
            for name, c, mean, lo, hi in op_rows(s):
                print(f"{os.path.relpath(s['p']['dir'])},{name},{c},"
                      f"{mean:.3f},{lo:.3f},{hi:.3f}")
        return
    print(f"{'profile':<34s} {'op':<30s} {'iter':>4s} {'mean':>9s} {'min':>9s} "
          f"{'wall/it':>9s} {'vec':>6s} {'sclr':>6s}")
    print("-" * 118)
    for s in ss:
        d = os.path.relpath(s["p"]["dir"])
        for name, c, mean, lo, hi in op_rows(s):
            rr = dict(ratios(s, name))
            print(f"{d[:34]:<34s} {name[:30]:<30s} {s['niter']:4d} "
                  f"{mean:9.3f} {lo:9.3f} {s['span_us']:9.3f} "
                  f"{(rr['vec'] or 0):6.3f} {(rr['sclr'] or 0):6.3f}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="msprof 输出目录（可多个，便于前后对照）")
    ap.add_argument("--drop", type=int, default=3,
                    help="丢弃开头几个热身迭代（默认 3）")
    ap.add_argument("--per", type=int, default=0,
                    help="每次迭代的算子个数；默认按 Op Name 序列自动检测")
    ap.add_argument("--csv", action="store_true", help="机器可读输出")
    a = ap.parse_args()

    loaded = [load(d) for d in a.dirs]
    bad = [d for d, p in zip(a.dirs, loaded) if p is None]
    for d in bad:
        print(f"warning: {d} 下找不到 op_summary*.csv（跳过）", file=sys.stderr)
    ps = [p for p in loaded if p]
    if not ps:
        return 1

    ss = [s for s in (summarize(p, a.drop, a.per) for p in ps) if s]
    if not ss:
        print("没有可用的 op 行（检查 --drop / --per）", file=sys.stderr)
        return 1

    if len(ss) == 1 or a.csv:
        for s in ss:
            print_one(s, a.csv)
    else:
        print_compare(ss, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
