#!/usr/bin/env python3
"""P0 同尺寸 torch_npu 长 FFT 基线与自研链对照（PR #2 性能评论 阶段 2）。

基线口径 = `scripts/bench_native_npu.py`（torch_npu op-plugin 的
`torch.fft.fft`，复->复，与自研同语义）：
  - device-only：`native_us`（min）/ `native_mean_us`（mean），torch.npu
    同步夹紧的墙钟；
  - E2E（pinned，与 fft_check `AB_E2E_HOST` 默认一致）：`e2e_us`（min）/
    `e2e_mean_us`（mean）+ 冷调用 `first_us`。
自研列取 `results/evidence/long-fft-device-boundary/acceptance.json`
的 5-trial 归档：`device_chain` 中位数（六段之和）与 E2E 中位数。

输出（确定性渲染，测量本身不可复算，归档即事实）：
  results/evidence/long-fft-baseline/baseline.json   机器可读归档
  docs/generated/long-fft-baseline.md                由 JSON 渲染的对照表
`--check` 只校验 md 与 JSON 一致、网格完整，不重新测量。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import abenv  # noqa: E402
from collect_long_fft_evidence import (  # noqa: E402
    NS, BS, git_state, detect_cann_version, detect_driver)

EVIDENCE = ROOT / "results" / "evidence" / "long-fft-device-boundary"
ACCEPTANCE = EVIDENCE / "acceptance.json"
OUT_JSON = ROOT / "results" / "evidence" / "long-fft-baseline" / "baseline.json"
OUT_MD = ROOT / "docs" / "generated" / "long-fft-baseline.md"
NATIVE_LINE = re.compile(
    r"^NATIVE n=(\d+) b=(\d+) native_us=([\d.]+) native_mean_us=([\d.]+) "
    r"maxRel=([0-9.eE+-]+) (PASS|FAIL)$", re.M)
NATIVE_E2E_LINE = re.compile(
    r"^NATIVE_E2E n=(\d+) b=(\d+) e2e_us=([\d.]+) e2e_mean_us=([\d.]+) "
    r"first_us=([\d.]+) maxRel=([0-9.eE+-]+) (PASS|FAIL)$", re.M)


def parse_native_output(text):
    """bench_native_npu stdout -> {(n, b): {...}}; raises on missing shapes."""
    shapes = {}
    for m in NATIVE_LINE.finditer(text):
        n, b = int(m.group(1)), int(m.group(2))
        shapes[(n, b)] = {
            "native_device_min_us": float(m.group(3)),
            "native_device_mean_us": float(m.group(4)),
            "native_max_rel": float(m.group(5)),
            "native_pass": m.group(6) == "PASS",
        }
    for m in NATIVE_E2E_LINE.finditer(text):
        n, b = int(m.group(1)), int(m.group(2))
        shapes.setdefault((n, b), {}).update({
            "native_e2e_min_us": float(m.group(3)),
            "native_e2e_mean_us": float(m.group(4)),
            "native_first_us": float(m.group(5)),
            "native_e2e_max_rel": float(m.group(6)),
            "native_e2e_pass": m.group(7) == "PASS",
        })
    missing = [(n, b) for n in NS for b in BS
               if (n, b) not in shapes
               or "native_e2e_min_us" not in shapes[(n, b)]]
    if missing:
        raise ValueError("native bench incomplete, missing %r" % (missing,))
    return shapes


def run_native(reps):
    """Invoke bench_native_npu on the long grid with --e2e; return parsed."""
    py = abenv.python_bin()
    cmd = [py, "scripts/bench_native_npu.py",
           "--ns", ",".join(str(n) for n in NS),
           "--bs", ",".join(str(b) for b in BS),
           "--reps", str(reps), "--e2e"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=3600)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:])
        raise SystemExit("native bench failed rc=%d" % proc.returncode)
    return parse_native_output(proc.stdout), cmd


def load_ours():
    """Device-boundary archive -> {(n, b): ours columns}."""
    if not ACCEPTANCE.is_file():
        raise SystemExit("%s missing; run collect_long_fft_evidence.py "
                         "--boundary device first" % ACCEPTANCE)
    doc = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    out = {}
    for p in doc.get("points") or []:
        chains = []
        for s in (p.get("trials") or {}).get("raw") or []:
            v = ((s.get("scopes") or {}).get("device_chain"))
            if isinstance(v, (int, float)):
                chains.append(v)
        stats = (p.get("trials") or {}).get("stats") or {}
        out[(p["n"], p["b"])] = {
            "ours_chain_median_us": _median(chains),
            "ours_e2e_median_us": stats.get("median"),
            "ours_e2e_min_us": stats.get("min"),
            "ours_max_rel": p.get("max_rel"),
        }
    missing = [(n, b) for n in NS for b in BS if (n, b) not in out]
    if missing:
        raise ValueError("ours archive incomplete, missing %r" % (missing,))
    return out, doc.get("manifest") or {}


def _median(values):
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _ratio(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) \
            or not b:
        return None
    return round(a / b, 3)


def build_document(native, ours, ours_manifest, native_cmd, reps,
                   allow_dirty=False):
    """Pure assembly of the machine-readable archive."""
    sha, dirty = git_state()
    if dirty and not allow_dirty:
        raise SystemExit(
            "refusing to publish baseline from a dirty tree (%s); commit "
            "first or pass --allow-dirty" % sha[:8])
    versions = subprocess.run(
        [abenv.python_bin(), "-c",
         "import torch, torch_npu; "
         "print(torch.__version__, torch_npu.__version__)"],
        capture_output=True, text=True, timeout=120)
    torch_v, torch_npu_v = (versions.stdout.split() + ["?", "?"])[:2]
    rows = []
    for n in NS:
        for b in BS:
            nat = native[(n, b)]
            our = ours[(n, b)]
            row = {"n": n, "b": b, **nat, **our}
            row["speedup_device_native_over_ours"] = _ratio(
                nat.get("native_device_min_us"), our.get("ours_chain_median_us"))
            row["speedup_e2e_native_over_ours"] = _ratio(
                nat.get("native_e2e_min_us"), our.get("ours_e2e_median_us"))
            rows.append(row)
    return {
        "schema": "long-fft-baseline/1",
        "threshold": 1e-4,
        "manifest": {
            "git_sha": sha,
            "git_dirty": dirty,
            "command": " ".join(native_cmd),
            "native_reps": reps,
            "torch": torch_v,
            "torch_npu": torch_npu_v,
            "cann": detect_cann_version(),
            "driver": detect_driver(),
            "ours_git_sha": ours_manifest.get("git_sha"),
            "ours_binary_sha256": ours_manifest.get("binary_sha256"),
            "ours_protocol": "5 trials x 5 reps, "
                             "results/evidence/long-fft-device-boundary",
        },
        "grid": {"ns": list(NS), "bs": list(BS)},
        "shapes": rows,
    }


def render_markdown(doc):
    """Deterministic docs table; no timestamps, fixed float formats."""
    lines = [
        "# 长 FFT 同尺寸基线：torch_npu vs 自研 device 边界链",
        "",
        "本页由 `python3 scripts/bench_long_baseline.py --check` 从归档 "
        "`results/evidence/long-fft-baseline/baseline.json` 渲染校验；请勿手改。",
        "",
        "- 基线：`torch.fft.fft`（torch_npu op-plugin，复->复）同网格同语义，"
        "device-only 取 `bench_native_npu.py` 的 min，E2E 取 pinned 口径 min"
        "（与 fft_check `AB_E2E_HOST` 默认一致）。",
        "- 自研：device 边界链 5-trial 归档的 `device_chain` / E2E 中位数"
        "（`segments:` 六段之和即 chain）。",
        "- `speedup = native_min / ours_median`，>1 表示自研更快；"
        "误差门槛 `maxRel ≤ %g`。" % doc["threshold"],
        "",
        "| N | B | native dev min (us) | ours chain median (us) | dev speedup "
        "| native E2E min (us) | ours E2E median (us) | E2E speedup "
        "| ours maxRel | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in doc["shapes"]:
        def f(v, spec="{:.1f}"):
            return spec.format(v) if isinstance(v, (int, float)) else "n/a"
        ok = r.get("native_pass") and r.get("native_e2e_pass") \
            and r.get("ours_max_rel") is not None \
            and r["ours_max_rel"] <= doc["threshold"]
        dev_sp = r.get("speedup_device_native_over_ours")
        e2e_sp = r.get("speedup_e2e_native_over_ours")
        lines.append(
            "| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["n"], r["b"],
                f(r.get("native_device_min_us")),
                f(r.get("ours_chain_median_us")),
                ("%.2fx" % dev_sp) if isinstance(dev_sp, (int, float)) else "n/a",
                f(r.get("native_e2e_min_us")),
                f(r.get("ours_e2e_median_us")),
                ("%.2fx" % e2e_sp) if isinstance(e2e_sp, (int, float)) else "n/a",
                f(r.get("ours_max_rel"), "{:.2e}"),
                "PASS" if ok else "CHECK"))
    m = doc["manifest"]
    lines += [
        "",
        "数据源：`%s`（基线，git `%s`，torch %s / torch_npu %s，CANN %s）与 "
        "`%s`（自研，git `%s`，binary `%s`）。"
        % ("results/evidence/long-fft-baseline/baseline.json",
           (m.get("git_sha") or "")[:8], m.get("torch"), m.get("torch_npu"),
           m.get("cann"),
           "results/evidence/long-fft-device-boundary/acceptance.json",
           (m.get("ours_git_sha") or "")[:8],
           (m.get("ours_binary_sha256") or "")[:12]),
        "",
    ]
    return "\n".join(lines)


def _render_json(doc):
    return json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def check():
    problems = []
    if not OUT_JSON.is_file():
        problems.append("missing %s" % OUT_JSON)
        return problems, None
    doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if doc.get("schema") != "long-fft-baseline/1":
        problems.append("unexpected schema %r" % doc.get("schema"))
    keys = [(s["n"], s["b"]) for s in doc.get("shapes") or []]
    want = [(n, b) for n in NS for b in BS]
    if keys != want:
        problems.append("grid mismatch: %r" % (keys,))
    if not OUT_MD.is_file():
        problems.append("missing %s" % OUT_MD)
    else:
        want_md = render_markdown(doc)
        if OUT_MD.read_text(encoding="utf-8") != want_md:
            problems.append("%s drifted from the archive JSON; rerun "
                            "without --check" % OUT_MD)
    return problems, doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the docs table matches the archived JSON")
    ap.add_argument("--reps", type=int, default=20,
                    help="native bench reps per shape (default 20)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="record the dirty flag instead of refusing")
    args = ap.parse_args(argv)

    if args.check:
        problems, doc = check()
        for p in problems:
            print(p, file=sys.stderr)
        if not problems:
            passed = sum(1 for s in doc["shapes"]
                         if s.get("native_pass") and s.get("native_e2e_pass"))
            print("baseline table matches the archive JSON "
                  "(%d/%d native shapes PASS)" % (passed, len(doc["shapes"])))
        return 1 if problems else 0

    native, cmd = run_native(args.reps)
    ours, ours_manifest = load_ours()
    doc = build_document(native, ours, ours_manifest, cmd, args.reps,
                         allow_dirty=args.allow_dirty)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(_render_json(doc), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(doc), encoding="utf-8")
    bad = [r for r in doc["shapes"]
           if not (r.get("native_pass") and r.get("native_e2e_pass"))]
    print("wrote %s and %s (%d shapes, %d native FAIL)"
          % (OUT_JSON.relative_to(ROOT), OUT_MD.relative_to(ROOT),
             len(doc["shapes"]), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
