#!/usr/bin/env python3
"""Regenerate docs/benchmarks/long-fft-p0-report.md from the archived evidence.

Every number in the report is recomputed here from committed archives only:
  results/evidence/long-fft-device-boundary/acceptance.json  device chain (§2-§6)
  results/evidence/long-fft-acceptance/acceptance.json        host chain (§5)
  results/evidence/long-fft-baseline/baseline.json            torch_npu baseline (§7)
  results/evidence/long-fft-p0/msprof-task-time.json          msprof §4 (extracted once
      from the local profile run via scripts/sum_prof.py; profiles stay uncommitted)
  results/evidence/long-fft-device-boundary-fused/acceptance.json
      PR-B R1 fused chain (§9); section rendered only when this archive
      exists, so pre-PR-B renders stay byte-identical

  python3 scripts/gen_p0_report.py            # rewrite the report
  python3 scripts/gen_p0_report.py --check    # fail on drift (lock chain)

tests/test_p0_report.py runs the same comparison in CI.
"""
import argparse
import json
import math
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "evidence"
DEVICE = EVIDENCE / "long-fft-device-boundary" / "acceptance.json"
HOST = EVIDENCE / "long-fft-acceptance" / "acceptance.json"
BASELINE = EVIDENCE / "long-fft-baseline" / "baseline.json"
MSPROF = EVIDENCE / "long-fft-p0" / "msprof-task-time.json"
FUSED = EVIDENCE / "long-fft-device-boundary-fused" / "acceptance.json"
OUT = ROOT / "docs" / "benchmarks" / "long-fft-p0-report.md"

SEGS = ("transpose_in", "fft1", "twiddle", "transpose_boundary", "fft2",
        "transpose_out")
# PR-B R1: fused chains report five spans (twiddle merged into the boundary
# transpose), so archive-derived segment iteration must follow the impl.
FUSED_SEGS = tuple(k for k in SEGS if k != "twiddle")
PRIORITY = ((8192, 1), (16384, 47), (32768, 47), (65536, 47))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _med(values):
    return statistics.median(values)


def _cv_pct(values):
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if mean else 0.0


def _r1(value):
    """Round to 1 decimal, half-up (prose share sums need exact decimals)."""
    return Decimal(str(value)).quantize(Decimal("0.1"),
                                        rounding=ROUND_HALF_UP)


def _pct0(value):
    """Integer percent, half-up."""
    return int(Decimal(str(value)).quantize(Decimal("1"),
                                            rounding=ROUND_HALF_UP))


def _chain_rows(doc, segs=SEGS):
    rows = []
    for p in doc["points"]:
        raw = p["trials"]["raw"]
        chains = [r["scopes"]["device_chain"] for r in raw]
        e2es = [r["e2e_us"] for r in raw]
        rows.append({
            "n": p["n"], "b": p["b"],
            "segs": {k: _med([r["segments"][k] for r in raw])
                     for k in segs},
            "chain": _med(chains), "chain_cv": _cv_pct(chains),
            "e2e": _med(e2es), "e2e_cv": _cv_pct(e2es),
            "h2d": _med([r["scopes"]["h2d"] for r in raw]),
            "d2h": _med([r["scopes"]["d2h"] for r in raw]),
        })
    return rows


def _host_e2e(doc):
    return {(p["n"], p["b"]): _med([r["e2e_us"] for r in p["trials"]["raw"]])
            for p in doc["points"]}


def _shares(row, segs=SEGS):
    """Segment share of chain (%), 1-decimal half-up Decimals."""
    return {k: _r1(row["segs"][k] / row["chain"] * 100.0) for k in segs}


def _shape_key(n, b):
    return f"{n}x{b}"


def render():
    dev_doc = _load(DEVICE)
    host_doc = _load(HOST)
    base = _load(BASELINE)
    msp = _load(MSPROF)
    fused_doc = _load(FUSED) if FUSED.is_file() else None

    dev = _chain_rows(dev_doc)
    dev_map = {(r["n"], r["b"]): r for r in dev}
    host = _host_e2e(host_doc)
    shares = {k: _shares(v) for k, v in dev_map.items()}
    bshapes = base["shapes"]
    bman = base["manifest"]

    # ---- §1 provenance -------------------------------------------------
    bin_sha = dev_doc["manifest"]["binary_sha256"][:12]
    torch_npu = bman["torch_npu"]
    native_reps = bman["native_reps"]
    profile_dir = msp["profiles_dir"]
    dev_git = dev_doc["manifest"]["git_sha"][:8]
    host_git = host_doc["manifest"]["git_sha"][:8]
    base_git = bman["git_sha"][:8]

    # ---- §4 msprof vs event -------------------------------------------
    def event_cols(n, b):
        r = dev_map[(n, b)]
        return (r["segs"]["transpose_in"] + r["segs"]["transpose_boundary"]
                + r["segs"]["transpose_out"],
                r["segs"]["fft1"] + r["segs"]["fft2"],
                r["segs"]["twiddle"])

    ms_rows = []
    diffs_b47 = []
    diff_8192 = None
    for n, b in PRIORITY:
        m = msp["shapes"][_shape_key(n, b)]
        ev = event_cols(n, b)
        ms = (m["kfft_lt_tr"], m["kfft_fwd"], m["kfft_lt_tw"])
        rel = [abs(ms[i] - ev[i]) / ev[i] * 100.0 for i in range(3)]
        if b == 47:
            diffs_b47.extend(rel)
        else:
            diff_8192 = rel[0]
        ms_rows.append((n, b, ms, ev))
    agree_b47 = max(diffs_b47)
    npass = sum(1 for r in bshapes
                if r.get("native_pass") and r.get("native_e2e_pass"))

    # ---- §5 host/device E2E -------------------------------------------
    e2e_rows = []
    for r in dev:
        k = (r["n"], r["b"])
        ratio = host[k] / r["e2e"]
        hd = r["h2d"] + r["d2h"]
        e2e_rows.append({"n": r["n"], "b": r["b"], "host": host[k],
                         "dev": r["e2e"], "ratio": ratio, "hd": hd,
                         "share": hd / r["e2e"] * 100.0})
    improved = [x for x in e2e_rows if x["ratio"] > 1.01]
    flat = [x for x in e2e_rows if 0.99 <= x["ratio"] <= 1.01]
    worse = [x for x in e2e_rows if x["ratio"] < 0.99]
    shares_all = [x["share"] for x in e2e_rows]
    shares_b47 = [x["share"] for x in e2e_rows if x["b"] == 47]

    # ---- §6 CV ranges --------------------------------------------------
    chain_cvs = [(r["n"], r["b"], r["chain_cv"]) for r in dev]
    e2e_cvs = [(r["n"], r["b"], r["e2e_cv"]) for r in dev]
    cv_all = [x[2] for x in chain_cvs]
    cv_b47 = [x[2] for x in chain_cvs if x[1] == 47]
    cv_small = [x[2] for x in chain_cvs if x[1] in (1, 3)]
    e2e_all = [x[2] for x in e2e_cvs]
    top3 = sorted(e2e_cvs, key=lambda x: -x[2])[:3]

    # ---- §7 baseline ---------------------------------------------------
    dev_sp = [(r["n"], r["b"], r["speedup_device_native_over_ours"])
              for r in bshapes]
    e2e_sp = [(r["n"], r["b"], r["speedup_e2e_native_over_ours"])
              for r in bshapes]
    dev_lo, dev_hi = min(x[2] for x in dev_sp), max(x[2] for x in dev_sp)
    e2e_lo, e2e_hi = min(x[2] for x in e2e_sp), max(x[2] for x in e2e_sp)
    faster = sorted((x for x in dev_sp if x[2] > 1.0),
                    key=lambda x: x[2], reverse=True)
    b47_e2e = [x[2] for x in e2e_sp if x[1] == 47]
    torch_ver = bman["torch"].split("+")[0]

    # ---- §3/§8 share prose --------------------------------------------
    r8192x1 = dev_map[(8192, 1)]
    sh8192x1 = shares[(8192, 1)]
    tr8192x1 = sum(sh8192x1[k] for k in
                   ("transpose_in", "transpose_boundary", "transpose_out"))
    fft8192x1 = sh8192x1["fft1"] + sh8192x1["fft2"]
    tw_b47 = [shares[(n, 47)]["twiddle"] for n in (8192, 16384, 32768, 65536)]
    tr_b47 = [sum(shares[(n, 47)][k] for k in
                  ("transpose_in", "transpose_boundary", "transpose_out"))
              for n in (8192, 16384, 32768, 65536)]
    sh65536x47 = shares[(65536, 47)]
    sh65536_vals = [sh65536x47[k] for k in SEGS]

    ratio_8192x1 = next(x["ratio"] for x in e2e_rows
                        if (x["n"], x["b"]) == (8192, 1))

    # ================= render ==========================================
    L = []
    a = L.append
    a("# P0 报告：长 FFT device 链六段归因与同尺寸 torch_npu 基线")
    a("")
    a("> PR #2 性能评论「阶段 2 · P0」的执行报告（第 1 轮）。本页由"
      " `python3 scripts/gen_p0_report.py` 从仓库归档直读生成，复算入口见"
      "文末「复现」；不引用任何未归档读数。")
    a("")
    a("## 1. 范围与方法")
    a("")
    a("| 项 | 口径 |")
    a("|---|---|")
    a("| 六段事件计时 | `fft_check.cpp` 在 device 边界内核发射间插入事件对，"
      "`segments:` 行与产生 `device_chain` 最小值的同一次 launch 绑定，"
      "`sum(segments) == device_chain`（telescope 恒等）；host/短路径 "
      "`segments: NA`。段数随 impl 分支：separate 六段（twiddle 与段边界"
      "转置两次发射）、fused 五段（twiddle 并入段边界转置），device 链"
      "另打 `boundary_impl:` 自报。`tests/test_scopes.py` + "
      "`tests/test_collect_evidence.py` 锁定 |")
    a(f"| 自研协议 | 每形状 {dev_doc['trials_per_shape']} 独立 trial × "
      f"{dev_doc['trials_per_shape']} reps；host `boundary=2`、device "
      f"`boundary=0`；同一 binary `sha256={bin_sha}` |")
    a(f"| 基线协议 | `torch.fft.fft`（torch_npu {torch_npu} op-plugin，复→复）"
      f"同网格，device-only 与 E2E 各 {native_reps} reps 取 min；"
      "E2E 为 pinned 口径 |")
    a(f"| 逐 kernel 校验 | msprof `--task-time`（`profile_test.sh --only "
      "lfft8k1,lfft16k47,lfft32k47,lfft65k47`），提取值归档于 "
      f"`{MSPROF.relative_to(ROOT)}`，原始 profile 在 "
      f"`{profile_dir}/`（本地，不入库） |")
    a("")
    bound = (f"归档绑定（全部清洁提交）：device `git={dev_git}`、host "
             f"`git={host_git}`、baseline `git={base_git}`")
    if fused_doc is not None:
        bound += f"、fused `git={fused_doc['manifest']['git_sha'][:8]}`"
    a(bound + "。")
    a("")
    a("## 2. 六段归因（separate impl，全网格，5-trial 中位数，µs）")
    a("")
    a("| N | B | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | "
      "transpose-out | device_chain | chain CV% | E2E | E2E CV% | h2d | d2h |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in dev:
        s = r["segs"]
        a(f"| {r['n']} | {r['b']} | {s['transpose_in']:.1f} | "
          f"{s['fft1']:.1f} | {s['twiddle']:.1f} | "
          f"{s['transpose_boundary']:.1f} | {s['fft2']:.1f} | "
          f"{s['transpose_out']:.1f} | {r['chain']:.1f} | "
          f"{r['chain_cv']:.2f} | {r['e2e']:.1f} | {r['e2e_cv']:.2f} | "
          f"{r['h2d']:.1f} | {r['d2h']:.1f} |")
    a("")
    a("数据源：`results/evidence/long-fft-device-boundary/acceptance.json` "
      "的 `points[].trials.raw[].segments/scopes`（`AB_LONG_BOUNDARY_IMPL"
      "=separate`，六发射六段"
      + ("；fused 五段对照见 §9" if fused_doc is not None else "") + "）。")
    a("")
    a("## 3. 优先形状的段占比与解读")
    a("")
    a("| 形状 | transpose-in | FFT1 | twiddle | transpose-boundary | FFT2 | "
      "transpose-out |")
    a("|---|---|---|---|---|---|---|")
    for n, b in PRIORITY:
        sh = shares[(n, b)]
        a(f"| {_shape_key(n, b)} | {sh['transpose_in']}% | {sh['fft1']}% | "
          f"{sh['twiddle']}% | {sh['transpose_boundary']}% | "
          f"{sh['fft2']}% | {sh['transpose_out']}% |")
    a("")
    a(f"- **`8192×1`（{ratio_8192x1:.2f}× 回退点）**：三次转置合计占 chain 的 "
      f"{tr8192x1}%，FFT 仅 {fft8192x1}%、"
      f"twiddle {sh8192x1['twiddle']}% —— 六次发射 + 三次全张量转置的固定成本"
      "无法由 batch=1 摊销，与评论「固定成本摊不销」的判断一致；"
      "优先级应放在减少转置成本而非 FFT 核心。")
    a(f"- **batch=47 形状：独立 twiddle 是最大单段**"
      f"（16384×47 {shares[(16384, 47)]['twiddle']}%、"
      f"32768×47 {shares[(32768, 47)]['twiddle']}%、"
      f"65536×47 {sh65536x47['twiddle']}%），超过任一段 FFT —— "
      "它对中间态做一次完整 GM 读 + 写外加两次 Gather 重排，"
      "是 P1 融合（twiddle+transpose-boundary）的首选目标。")
    a(f"- `65536×47` 六段均衡（各 {min(sh65536_vals)}%–{max(sh65536_vals)}%），"
      "与带宽受限假设一致（待 MTE/GM/Vector 计数器验证，task-time/event "
      "只能定位昂贵段、不能证明瓶颈类型）；融合后可省去中间态一整次 GM "
      "写+读。")
    a("")
    a("## 4. msprof task-time 交叉验证（µs/launch，4 优先形状）")
    a("")
    a("| 形状 | kfft_lt_tr（3 次合计） | 事件三转置合计 | kfft_fwd（2 次合计） "
      "| 事件两 FFT 合计 | kfft_lt_tw | 事件 twiddle |")
    a("|---|---|---|---|---|---|---|")
    for n, b, ms, ev in ms_rows:
        a(f"| {_shape_key(n, b)} | {ms[0]:.1f} | {ev[0]:.1f} | "
          f"{ms[1]:.1f} | {ev[1]:.1f} | {ms[2]:.1f} | {ev[2]:.1f} |")
    a("")
    a(f"msprof 列数据源：`{MSPROF.relative_to(ROOT)}`（`sum_prof.py` 从本地 "
      "profile 提取的均值口径）；事件列为 device 归档六段中位数之和。")
    a("")
    a(f"两套独立测量（host 侧事件 vs msprof device task-time）在 b=47 形状"
      f"逐 op 吻合 {agree_b47:.1f}% 以内；`8192×1` 的三转置项差 "
      f"{diff_8192:.1f}%（profiler 汇入了 E2E 发射，事件只取 chain 最小值的"
      "单次 launch，小形状 launch 间散布大）。六段口径可以放心用于后续候选"
      "排名；profiler 继续用于解释原因（PipeUtilization 等），不替代真实 "
      "event 计时。")
    a("")
    a("## 5. E2E 分解与传输口径")
    a("")
    a("| 形状 | host E2E | device E2E | host/device | device h2d+d2h | "
      "占 device E2E |")
    a("|---|---|---|---|---|---|")
    for x in e2e_rows:
        a(f"| {_shape_key(x['n'], x['b'])} | {x['host']:.1f} | "
          f"{x['dev']:.1f} | {x['ratio']:.2f}× | {x['hd']:.1f} | "
          f"{_pct0(x['share'])}% |")
    a("")
    a(f"数据源：`{HOST.relative_to(ROOT)}`（host 链 E2E 中位数）与 "
      f"`{DEVICE.relative_to(ROOT)}`（device 链 E2E/h2d/d2h 中位数）。")
    a("")
    def clause(rows, kind):
        parts = []
        for x in rows:
            tail = "，第 3 节固定成本归因" if (
                kind == "回退" and (x["n"], x["b"]) == (8192, 1)) else ""
            if kind == "持平":
                parts.append(f"`{x['n']}×{x['b']}` 持平"
                             f"（{x['ratio']:.2f}×）")
            else:
                parts.append(f"`{x['n']}×{x['b']}` 例外回退"
                             f"（{x['ratio']:.2f}×{tail}）")
        return "、".join(parts)
    tail_txt = []
    if flat:
        tail_txt.append(clause(flat, "持平"))
    if worse:
        tail_txt.append(clause(worse, "回退"))
    a(f"- 设备段边界链在 {len(improved)}/12 个形状上降低 E2E"
      f"（最高 {max(x['ratio'] for x in improved):.2f}×）"
      + (("，" + "，".join(tail_txt)) if tail_txt else "，无持平/回退形状")
      + "。")
    a(f"- **自研长链 E2E 的 {_pct0(min(shares_all))}%–"
      f"{_pct0(max(shares_all))}% 是 H2D+D2H**"
      f"（b=47 达 {_pct0(min(shares_b47))}%–{_pct0(max(shares_b47))}%），"
      "且当前长路径从 `std::vector` 发起"
      "（`fft_check.cpp` 的 pinned 分配仅覆盖短路径），为 pageable 口径；"
      "下一节基线的 E2E 差距必须先扣除此项再谈 kernel。")
    a("- host 链的 E2E 主体是宿主标量段（四步转置/twiddle/重排的 CPU 循环），"
      "不是 kernel 慢，比较口径不同。")
    a("")
    a("## 6. 波动性")
    a("")
    a(f"- device `device_chain` CV：**{min(cv_all):.2f}%–"
      f"{max(cv_all):.2f}%**（b=47 形状 {min(cv_b47):.2f}%–"
      f"{max(cv_b47):.2f}%，b≤3 形状 {min(cv_small):.2f}%–"
      f"{max(cv_small):.2f}%，短链单次 launch 的时钟分辨率噪声占比更高）。")
    a(f"- device E2E CV：**{min(e2e_all):.2f}%–{max(e2e_all):.2f}%**"
      + "（" + "、".join(f"`{n}×{b}` {cv:.2f}%"
                         for n, b, cv in top3)
      + "），显著高于同形状 chain CV —— 波动来自传输、host dispatch 与环境，"
        "不是 FFT 算术核心；评论的判断成立。")
    a("- 下一步按评论要求做交错运行 + 环境记录（温度/频率/共租户）后再定"
      "是否可接受。")
    a("")
    a("## 7. 同尺寸 torch_npu 基线")
    a("")
    a(f"完整 12 行对照见生成页 [长 FFT 同尺寸基线]"
      "(../generated/long-fft-baseline.md)"
      f"（`results/evidence/long-fft-baseline/baseline.json`，{npass}/12 "
      f"native PASS，torch {torch_ver} / torch_npu {torch_npu} / "
      f"CANN {bman['cann']}，native {native_reps} reps min）。要点：")
    a("")
    if faster:
        who = "、".join(f"`{n}×{b}` 为 {v:.2f}×" for n, b, v in faster)
        lead = (f"仅 {who}，其余 native 领先" if len(faster) == 1
                else f"{who}，其余 native 领先")
    else:
        lead = "本组无自研领先形状"
    a(f"- **device-only：`native_min / ours_median` = {dev_lo:.3f}×–"
      f"{dev_hi:.3f}×**（>1 表示自研更快：{lead}）。"
      "native 是单次融合变换不物化本实现的三转置段边界，形态不同；"
      "3.4× 的 device-vs-host 结论**不能**外推为对外部库优势。")
    a(f"- **E2E：同一比值 {e2e_lo:.3f}×–{e2e_hi:.3f}×**，b=47 最差"
      f"（{min(b47_e2e):.3f}×–{max(b47_e2e):.3f}×）——其中含自研 pageable "
      "传输口径 vs native 的差距（第 5 节），与 kernel 差距必须分列。")
    a("- 两口径、两协议均已归档，后续任何候选以同表复测对比。")
    a("")
    a("## 8. 结论 → P1 优先级")
    a("")
    a(f"1. **融合 twiddle + transpose-boundary**（`fft_long.cpp`，"
      "`kfft_lt_tr` 签名已带 `tw`）：目标消掉最大单段（"
      f"{_pct0(min(tw_b47))}%–{_pct0(max(tw_b47))}%）+ 中间态一整次 GM 写读；"
      "验收 = 六段计时 + GM 字节 + A/B/A，六内核路径保留 incumbent"
      + ("（PR-B 已落地 separate/fused 双路径，对照见 §9）"
         if fused_doc is not None else "") + "。")
    a(f"2. **转置成本**（`8192×1` 占 {tr8192x1}%、b=47 合计 "
      f"{_pct0(min(tr_b47))}%–{_pct0(max(tr_b47))}%）：tile/LT_H×LT_W "
      "参数化、双缓冲（当前 3×32KB UB，`long_fft_ub.h` 模型已就位）、"
      "窄作用域 pipe barrier。")
    a("3. **长链 pinned 传输**：修正 E2E 口径，使与 native 的 E2E 对比同口径。")
    a("4. **与 native 的形态差距**：P1 融合做完后重新对照；不达标候选不进"
      "默认配置。")
    a("5. **波动治理**：交错运行 + 环境记录（评论退出条件）。")
    a("")
    cond1 = ("不再慢于 host（满足）" if ratio_8192x1 >= 1.0
             else "仍慢于 host（未满足，已有归因）")
    mark5 = "**满足**" if npass == 12 else "**未满足**"
    cond4 = ("twiddle 融合已做（PR-B separate/fused 双路径，§9）"
             if fused_doc is not None else "twiddle 融合未做（P1 首项）")
    a(f"评论「下一轮退出条件」现状：① 8192×1 {cond1}；"
      "② 六段时间可解释 chain（**满足**，§2/§4）；"
      f"③ 高 CV 未治理（待环境记录）；④ {cond4}；"
      f"⑤ torch_npu 同语义矩阵（{mark5}，{npass}/12 §7）。")
    a("")
    if fused_doc is not None:
        fu = _chain_rows(fused_doc, segs=FUSED_SEGS)
        fu_map = {(r["n"], r["b"]): r for r in fu}
        a("## 9. PR-B R1：twiddle 并入边界转置（separate vs fused 对照）")
        a("")
        a("`AB_LONG_BOUNDARY_IMPL` 双路径：separate = 六次发射六段"
          "（twiddle 独立成段），fused = 五次发射五段（`kfft_lt_tr` 在段"
          "边界转置的读入 tile 内复乘 twiddle，`tw` 参数直接消费）。"
          f"归档 `{FUSED.relative_to(ROOT)}`：`boundary={fused_doc['boundary']}`、"
          f"`impl={fused_doc.get('impl')}`，每 trial 自报 "
          "`boundary_impl: fused` 且段数契约由 collect 逐 trial 校验"
          "（`segments` 五段 telescope 进 `device_chain`）。"
          "链与 E2E 均为 5-trial 中位数：")
        a("")
        a("![Long FFT boundary fusion detail]"
          "(../figures/boundary_fusion_detail.svg)")
        a("")
        a("这张局部图只说明边界组织的差异，不把“少一次发射”直接等同于"
          "端到端加速；是否晋升仍由同协议的 paired event、正确性和波动门槛决定。")
        a("")
        a("| 形状 | separate chain | fused chain | Δ chain | separate E2E "
          "| fused E2E | Δ E2E |")
        a("|---|---|---|---|---|---|---|")
        dchain, de2e = [], []
        for r in dev:
            f = fu_map.get((r["n"], r["b"]))
            if f is None:
                continue
            d = (f["chain"] - r["chain"]) / r["chain"] * 100.0
            de = (f["e2e"] - r["e2e"]) / r["e2e"] * 100.0
            dchain.append(d)
            de2e.append(de)
            a(f"| {_shape_key(r['n'], r['b'])} | {r['chain']:.1f} | "
              f"{f['chain']:.1f} | {d:+.1f}% | {r['e2e']:.1f} | "
              f"{f['e2e']:.1f} | {de:+.1f}% |")
        a("")
        fused_faster = sum(1 for d in dchain if d < 0)
        a(f"- fused 在 {fused_faster}/{len(dchain)} 个形状上缩短 device_chain"
          f"（Δ 中位数 {statistics.median(dchain):+.1f}%，"
          f"最差 {max(dchain):+.1f}%）；E2E 因 H2D+D2H 占比被稀释"
          f"（ΔE2E 中位数 {statistics.median(de2e):+.1f}%）。")
        a("- 本表是全网格初测（非交错序）；晋升门槛"
          "（≥4/5 配对不慢且中位数改善 ≥5%，任一点回退 >3% 则保留 "
          "separate fallback）以 4 优先点 `8192×1 / 16384×47 / 32768×47 / "
          "65536×47` 的 separate/fused 交错 ≥5 trial 归因为准。")
        a("")
    a("## 10. 复现")
    a("")
    a("```bash")
    a("bash scripts/build.sh")
    a("python3 scripts/collect_long_fft_evidence.py              # host, 含 segments")
    a("python3 scripts/collect_long_fft_evidence.py --boundary device")
    a("python3 scripts/collect_long_fft_evidence.py --boundary device-fused "
      "# PR-B §9")
    a("python3 scripts/collect_long_fft_evidence.py --verify "
      "results/evidence/long-fft-acceptance/acceptance.json")
    a("python3 scripts/bench_long_baseline.py                    # torch_npu 基线")
    a("python3 scripts/bench_long_baseline.py --check            # md ↔ JSON")
    a("python3 scripts/summarize_long_fft_evidence.py --check")
    a("python3 scripts/gen_p0_report.py --check                  # 本页 ↔ 归档")
    a("scripts/profile_test.sh --only lfft8k1,lfft16k47,lfft32k47,lfft65k47")
    a("python3 scripts/sum_prof.py --csv results/profiles/<dir>/lfft*  # §4 出处")
    a("python3 -m unittest discover -s tests -q")
    a("```")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the committed report matches the archives")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        if not OUT.is_file():
            print(f"missing {OUT}", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} drifted from the archives; rerun without --check",
                  file=sys.stderr)
            return 1
        print("p0 report matches the committed archives")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
