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
import argparse, math, os, random, re, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abenv  # noqa: E402  与 scripts/env.sh 共用同一套路径探测
PY = abenv.python_bin()
PROFILE = os.environ.get("AB_PROFILE", os.path.join(ROOT, "config/ascend910_93_profile.json"))


def sh(cmd, env=None, cwd=ROOT, timeout=3600):
    e = dict(os.environ); e.update(env or {})
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, env=e,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, "", f"timeout after {exc.timeout}s")


def output(result):
    return result.stdout + result.stderr


def valid_measurement(mean, minimum, error):
    return (all(math.isfinite(value) for value in (mean, minimum, error))
            and mean > 0 and minimum > 0 and 0 <= error <= 1e-4)


def f(pat, s, d=float("nan")):
    m = re.search(pat, s)
    return float(m.group(1)) if m else d


def runner_orders(rounds, native_enabled, seed):
    rng = random.Random(seed)
    orders = []
    for _ in range(rounds):
        order = ["self"] if not native_enabled else ["native", "self"]
        rng.shuffle(order)
        orders.append(order)
    return orders


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="64,128,256,512,1024,2048,4096")
    ap.add_argument("--bs", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3,
                    help="整套测量跑几遍、逐点取 min（0 噪声口径，见文件头）")
    ap.add_argument("--no-eta", action="store_true")
    ap.add_argument("--no-native", action="store_true")
    ap.add_argument("--order-seed", type=int, default=0,
                    help="可复现地随机交错 native/self 轮次，降低时序漂移偏差")
    from datetime import datetime, timezone
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-matrix")
    ap.add_argument("--out", default=f"results/runs/{run_name}/matrix.md")
    ap.add_argument("--csv", default=None, help="结构化矩阵；默认与 --out 同目录")
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    if a.rounds < 1 or a.reps < 1 or not ns or not bs or any(x < 1 for x in ns + bs):
        ap.error("lengths, batches, reps and rounds must be positive")
    if len(set(ns)) != len(ns) or len(set(bs)) != len(bs):
        ap.error("lengths and batches must not contain duplicates")
    rounds = a.rounds
    expected = {(n, b) for n in ns for b in bs}
    trials = []
    native_status = {key: True for key in expected}
    native_errors = {}

    nat, nat_ok = {}, True
    ours = {key: {"mean": float("nan"), "min": float("nan"),
                  "maxRel": float("nan"), "ok": True} for key in expected}
    orders = runner_orders(rounds, not a.no_native, a.order_seed)
    print(f"[1/3 + 2/3] paired runner rounds seed={a.order_seed}: {orders}", file=sys.stderr)
    for trial, order in enumerate(orders, 1):
        for runner in order:
            if runner == "native":
                result = sh(f"{PY} scripts/bench_native_npu.py --ns {a.ns} --bs {a.bs} "
                            f"--reps {max(a.reps, 20)}")
                records = {key: [] for key in expected}
                for line in output(result).splitlines():
                    match = re.match(
                        r"NATIVE n=(\d+) b=(\d+) native_us=([\d.]+) native_mean_us=([\d.]+) "
                        r"maxRel=([\d.eE+-]+) (\w+)", line)
                    if match:
                        key = (int(match.group(1)), int(match.group(2)))
                        if key in records:
                            records[key].append(match)
                for key, matches in sorted(records.items()):
                    if len(matches) != 1:
                        native_status[key] = nat_ok = False
                        trials.append({"runner": "native", "trial": trial, "n": key[0],
                                       "batch": key[1], "returncode": result.returncode,
                                       "correct": False,
                                       "failure": f"expected one record, found {len(matches)}"})
                        continue
                    match = matches[0]
                    nmin, nmean = float(match.group(3)), float(match.group(4))
                    error = float(match.group(5))
                    good = (result.returncode == 0 and match.group(6) == "PASS"
                            and valid_measurement(nmean, nmin, error))
                    native_status[key] &= good
                    nat_ok &= good
                    native_errors[key] = max(native_errors.get(key, error), error)
                    trials.append({"runner": "native", "trial": trial, "n": key[0],
                                   "batch": key[1], "returncode": result.returncode,
                                   "mean_us": nmean, "min_us": nmin, "max_rel": error,
                                   "correct": good,
                                   "failure": "" if good else "command, PASS marker, timing or error check failed"})
                    previous = nat.get(key)
                    nat[key] = ([nmin, nmean] if previous is None else
                                [min(previous[0], nmin), min(previous[1], nmean)])
            else:
                for n in ns:
                    for b in bs:
                        current = ours[(n, b)]
                        result = sh(f"./build/fft_check {n} {b} {a.reps}")
                        text = output(result)
                        mean = f(r"kfft_fwd: ([\d.]+) us/call", text)
                        minimum = f(r"\(min ([\d.]+) us\)", text)
                        error = f(r"maxRel=([\d.eE+-]+)", text)
                        good = (result.returncode == 0
                                and bool(re.search(r"^PASS$", text, re.M))
                                and not bool(re.search(r"^FAIL\b", text, re.M))
                                and valid_measurement(mean, minimum, error))
                        current["ok"] &= good
                        trials.append({"runner": "self", "trial": trial, "n": n,
                                       "batch": b, "returncode": result.returncode,
                                       "mean_us": mean, "min_us": minimum,
                                       "max_rel": error, "correct": good,
                                       "failure": "" if good else "command, PASS marker, timing or error check failed"})
                        if math.isfinite(error):
                            current["maxRel"] = (error if not math.isfinite(current["maxRel"])
                                                 else max(current["maxRel"], error))
                        if math.isfinite(mean):
                            current["mean"] = (mean if not math.isfinite(current["mean"])
                                               else min(current["mean"], mean))
                        if math.isfinite(minimum):
                            current["min"] = (minimum if not math.isfinite(current["min"])
                                              else min(current["min"], minimum))
    if not a.no_native:
        print(f"    native {len(nat)} 点 {'PASS' if nat_ok else 'FAIL'}", file=sys.stderr)
    for n in ns:
        for b in bs:
            current = ours[(n, b)]
            print(f"    n={n:<5} b={b:<5} {'PASS' if current['ok'] else 'FAIL'}"
                  f"  {current['mean']:.1f} us", file=sys.stderr)

    eta = {}
    if not a.no_eta:
        print("[3/3] 框架 η / 选型闭环 ...", file=sys.stderr)
        for n in ns:
            for b in bs:
                s = output(sh(f"./build/test_framework {shlex.quote(PROFILE)} "
                              f"config/butterfly_space.json build/fft_radix2.o {n} {b}"))
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
            good = o["ok"] and (a.no_native or native_status[(n, b)])
            rows.append((n, b, o["mean"], o["min"], o["maxRel"], nm, nmean, e, good))

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
    L(f"> 每点 {rounds} trials；所有启用 runner 的每轮必须成功、PASS 且完整覆盖；"
      f"runner 按 seed={a.order_seed} 逐轮随机交错；耗时保留 min-of-means，"
      "maxRel 取所有自研轮次最大值。\n")
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
        good = ok
        n_ok += 1 if good else 0
        # nat = (min, mean)，列序与表头一致：原生 mean | 原生 min
        L(f"| {n} | {b} | {fmt(om)} | {fmt(omin)} | {fmt(nmean)} | {fmt(nm)} | {rs} "
          f"| {fmt(e)} | {ed} | {rel:.2e} | {'PASS' if good else 'FAIL'} |")

    lines.append("")
    L(f"**正确性：{n_ok}/{len(rows)} PASS**（判据 maxRel ≤ 1e-4，"
      f"自研与原生各自对双精度 CPU 参考；原生 "
      f"{'未启用' if a.no_native else ('全部 PASS' if nat_ok else '有 FAIL')}）")
    failures = [record for record in trials if not record["correct"]]
    L(f"**逐轮验收：{len(trials) - len(failures)}/{len(trials)} PASS；失败 {len(failures)} 轮。**")
    body = "\n".join(lines) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(body)
        print(f"written -> {a.out}", file=sys.stderr)
        import csv
        import json
        destination = a.csv or os.path.join(os.path.dirname(a.out), "matrix.csv")
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        with open(destination, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["n", "batch", "ours_us", "ours_min_us", "max_rel", "native_min_us", "native_us", "eta_us", "correct"])
            writer.writerows(rows)
        with open(os.path.join(os.path.dirname(a.out), "trials.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["runner", "trial", "n", "batch", "returncode",
                                                       "mean_us", "min_us", "max_rel", "correct", "failure"])
            writer.writeheader()
            writer.writerows(trials)
        with open(os.path.join(os.path.dirname(a.out), "summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"operator": "c2c", "precision": "fp32", "direction": "forward",
                       "timing": "device-only", "reps": a.reps, "rounds": rounds,
                       "runner_order_seed": a.order_seed, "runner_orders": orders,
                       "ns": ns, "batches": bs, "correctness_threshold": 1e-4,
                       "correct_points": n_ok, "total_points": len(rows),
                       "native_enabled": not a.no_native, "total_trials": len(trials),
                       "failed_trials": len(failures),
                       "failures": [{key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                                     for key, value in record.items()} for record in failures],
                       "native_max_rel": [{"n": n, "batch": b, "max_rel": error}
                                          for (n, b), error in sorted(native_errors.items())
                                          if math.isfinite(error)]},
                      handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(body)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
