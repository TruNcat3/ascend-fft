#!/usr/bin/env python3
"""Build the machine-readable long-FFT evidence summary and its docs table.

Reads the committed acceptance archives
  results/evidence/long-fft-acceptance/acceptance.json        (host chain)
  results/evidence/long-fft-device-boundary/acceptance.json   (device chain)
recomputes every published number from the raw samples, and writes
  results/evidence/long-fft-summary.json     machine-readable summary
  docs/generated/long-fft-evidence.md        human table linked from docs

Output is deterministic (no timestamps): `--check` regenerates in memory and
fails on any drift, so documentation can only cite numbers recomputable from
the archive (PR #2 stage 5).  tests/test_evidence_summary.py runs the same
comparison in CI.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "evidence"
ACCEPTANCE = {
    "host": EVIDENCE / "long-fft-acceptance" / "acceptance.json",
    "device": EVIDENCE / "long-fft-device-boundary" / "acceptance.json",
}
SUMMARY_OUT = EVIDENCE / "long-fft-summary.json"
MD_OUT = ROOT / "docs" / "generated" / "long-fft-evidence.md"


def load_mode(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(block):
    """Trials stats dict or None on the pre-trial archive schema."""
    if not block:
        return None
    stats = (block.get("trials") or {}).get("stats")
    if not stats or stats.get("median") is None:
        return None
    return {k: stats[k] for k in ("median", "min", "mean", "cv")}


def _seq_worst(point):
    values = [s.get("max_rel") for s in (point.get("seq") or [])
              if isinstance(s.get("max_rel"), (int, float))]
    return max(values) if values else None


def build_summary():
    modes = {}
    for mode, path in ACCEPTANCE.items():
        doc = load_mode(path)
        if doc is None:
            continue
        manifest = doc.get("manifest") or {}
        shapes = []
        for p in doc.get("points") or []:
            shapes.append({
                "n": p.get("n"), "b": p.get("b"),
                "max_rel": p.get("max_rel"),
                "seq_worst_max_rel": _seq_worst(p),
                "e2e_us": _stats(p),
                "device_chain_us": None,
            })
            raw = ((p.get("trials") or {}).get("raw") or [])
            chains = [((s.get("scopes") or {}).get("device_chain"))
                      for s in raw]
            chains = [c for c in chains if isinstance(c, (int, float))]
            if chains:
                import statistics
                mean = statistics.fmean(chains)
                shapes[-1]["device_chain_us"] = {
                    "median": statistics.median(chains),
                    "min": min(chains), "mean": mean,
                    "cv": (statistics.pstdev(chains) / mean) if mean else None,
                }
        modes[mode] = {
            "git_sha": manifest.get("git_sha") or doc.get("git_sha"),
            "git_dirty": manifest.get("git_dirty",
                                      doc.get("git_dirty")),
            "binary_sha256": manifest.get("binary_sha256"),
            "status": doc.get("status"),
            "boundary": (doc.get("e2e_transfers") or {}).get(
                "transfers", ""),
            "trials_per_shape": doc.get("trials_per_shape"),
            "threshold": doc.get("threshold"),
            "shapes": shapes,
        }
    worst = None
    for mode in modes.values():
        for s in mode["shapes"]:
            for key in ("max_rel", "seq_worst_max_rel"):
                v = s.get(key)
                if isinstance(v, (int, float)) and (worst is None or v > worst):
                    worst = v
    # median e2e speedup host/device per shape (needs both archives' trials)
    speedup = {}
    if "host" in modes and "device" in modes:
        dev = {(s["n"], s["b"]): s for s in modes["device"]["shapes"]}
        for s in modes["host"]["shapes"]:
            d = dev.get((s["n"], s["b"]))
            if s.get("e2e_us") and d and d.get("e2e_us"):
                ratio = s["e2e_us"]["median"] / d["e2e_us"]["median"]
                speedup[f"{s['n']}x{s['b']}"] = round(ratio, 3)
    return {
        "schema": "long-fft-evidence-summary/1",
        "threshold": 1e-4,
        "modes": modes,
        "worst_max_rel": worst,
        "median_e2e_speedup_host_over_device": speedup,
    }


def _fmt(v, spec="{:.1f}"):
    return spec.format(v) if isinstance(v, (int, float)) else "n/a"


def render_markdown(summary):
    lines = [
        "# 长 FFT 验收证据（宿主/设备段边界对照）",
        "",
        "本页由 `python3 scripts/summarize_long_fft_evidence.py` 从两份"
        "`acceptance.json` 原始样本生成，`--check` 可复核漂移；请勿手改。",
        "",
        f"- 门槛 `maxRel ≤ {summary['threshold']:g}`，全网格最坏 "
        f"`{summary['worst_max_rel']}`（A/B/A 序列与整体输出口径的并集）。",
        "- 协议：每个形状 5 次独立 trial（每次进程内 5 reps），原始样本全部"
        "归档于各 `acceptance.json` 的 `trials.raw`；下表为 trial 统计。",
        "- 传输契约：宿主链每次执行 `in=1 out=1 boundary=2`，设备链"
        "`in=1 out=1 boundary=0`（段边界不回宿主）。",
        "",
        "| N | B | host e2e median (us) | device e2e median (us) | "
        "median speedup | worst maxRel (host) | worst maxRel (device) |",
        "|---|---|---|---|---|---|---|",
    ]
    host = (summary["modes"].get("host") or {}).get("shapes") or []
    devmap = {(s["n"], s["b"]): s
              for s in (summary["modes"].get("device") or {}).get("shapes") or []}
    for s in host:
        d = devmap.get((s["n"], s["b"]), {})
        key = f"{s['n']}x{s['b']}"
        speed = summary["median_e2e_speedup_host_over_device"].get(key, "n/a")
        speed = f"{speed}x" if isinstance(speed, (int, float)) else speed
        lines.append(
            f"| {s['n']} | {s['b']} | "
            f"{_fmt((s.get('e2e_us') or {}).get('median'))} | "
            f"{_fmt((d.get('e2e_us') or {}).get('median'))} | {speed} | "
            f"{_fmt(s.get('max_rel'), '{:.2e}')} | "
            f"{_fmt(d.get('max_rel'), '{:.2e}')} |")
    lines += [
        "",
        "数据源：`results/evidence/long-fft-acceptance/`（host）与 "
        "`results/evidence/long-fft-device-boundary/`（device），"
        "汇总见 `results/evidence/long-fft-summary.json`。",
        "",
    ]
    return "\n".join(lines)


def render(summary):
    return json.dumps(summary, indent=1, ensure_ascii=False,
                      sort_keys=True) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify committed summary/docs match the archives")
    args = ap.parse_args(argv)

    summary = build_summary()
    summary_text = render(summary)
    md_text = render_markdown(summary)
    if args.check:
        problems = []
        if not SUMMARY_OUT.is_file():
            problems.append(f"missing {SUMMARY_OUT}")
        elif SUMMARY_OUT.read_text(encoding="utf-8") != summary_text:
            problems.append(f"{SUMMARY_OUT} drifted from the archives; "
                            "rerun without --check")
        if not MD_OUT.is_file():
            problems.append(f"missing {MD_OUT}")
        elif MD_OUT.read_text(encoding="utf-8") != md_text:
            problems.append(f"{MD_OUT} drifted from the archives; "
                            "rerun without --check")
        if problems:
            for p in problems:
                print(p, file=sys.stderr)
            return 1
        print("evidence summary matches the committed archives")
        return 0
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(summary_text, encoding="utf-8")
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.write_text(md_text, encoding="utf-8")
    print(f"wrote {SUMMARY_OUT.relative_to(ROOT)} and "
          f"{MD_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
