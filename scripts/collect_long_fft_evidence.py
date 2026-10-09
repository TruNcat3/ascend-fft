#!/usr/bin/env python3
"""Collect the long-FFT dynamic-input acceptance evidence (P1-A / G1 envelope).

Runs the committed `fft_check` binary over the G1 grid
N={8192,16384,32768,65536} x B={1,3,47} with an A/B/A input sequence from raw
float32 files (file contents change between calls), plus a short-path A/B/A
control and the E2E single-input/single-output transfer assertion.  Every
shape also gets 5 independent metric trials (one process invocation each,
raw samples retained, median/min/mean/CV reported).

Hard acceptance (PR #2 stage 4, verify_* functions are unit-tested in
tests/test_collect_evidence.py):
  - every long shape exits rc == 0;
  - exactly three seq entries in `impulse, random-seeded, impulse` order,
    each PASS, finite, <= threshold, never STALE-OUTPUT;
  - the stale-input summary line exists and ends with `-> PASS`;
  - the 12 (N, batch) keys are complete and unique;
  - transfer/boundary counters match the selected path (host boundary=2,
    device boundary=0), both per shape and on the dedicated E2E run;
  - manifest: clean git tree is required (or --allow-dirty records a patch
    digest), sha256(build/fft_check), build recipe, CANN/SoC/driver,
    effective AB_* environment and the exact commands.

  python3 scripts/collect_long_fft_evidence.py               # 宿主中介链（默认）
  python3 scripts/collect_long_fft_evidence.py --boundary device
      # addendum §3 device-materialized 段边界（AB_BOUNDARY=device）：
      # 同一网格 + A/B/A，段边界不回宿主 => 期望 E2E boundary=0，
      # 证据写入 results/evidence/long-fft-device-boundary/。
  --allow-dirty   记录 patch digest 后继续（默认脏树直接拒绝发布证据）
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scopes  # noqa: E402  (shared scopes-line parser/validator)

NS = (8192, 16384, 32768, 65536)
BS = (1, 3, 47)
THRESHOLD = 1e-4
EXPECTED_SEQ = ("impulse", "random-seeded", "impulse")
TRIALS = 5
# 全部执行带 AB_E2E：逐点断言段边界传输契约——
# host 模式 = 宿主中介链 boundary=2（行为锁定）；device 模式 = 不回宿主 boundary=0。
BASE_ENV = {"AB_E2E": "1"}
BOUNDARY_BY_MODE = {"host": "boundary=2", "device": "boundary=0"}
OUT_BY_MODE = {"host": "long-fft-acceptance",
               "device": "long-fft-device-boundary"}


def run(args, env=None):
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          env=env, timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def parse_point(out):
    m = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    seq = re.findall(
        r"seq\[\d+\]=(\S+) maxRel=([\d.eE+-]+) (PASS|FAIL)( STALE-OUTPUT)?",
        out)
    seq_line = re.search(r"^seq: .*$", out, re.M)
    transfers = re.search(r"^(E2E transfers: .*)$", out, re.M)
    return {
        "max_rel": float(m.group(2)) if m else None,
        "max_abs": float(m.group(1)) if m else None,
        "seq": [{"input": s[0], "max_rel": float(s[1]), "pass": s[2] == "PASS",
                 "stale": bool(s[3])} for s in seq],
        "seq_summary": seq_line.group(0) if seq_line else "",
        "e2e_transfers": transfers.group(1) if transfers else "",
        "pass": bool(re.search(r"^PASS$", out, re.M)),
    }


def parse_trial(out):
    """One independent metric invocation -> raw trial sample."""
    e2e = re.search(r"^E2E n=.*e2e_us=([\d.eE+-]+) e2e_min_us=([\d.eE+-]+)",
                    out, re.M)
    sample = {
        "e2e_us": float(e2e.group(1)) if e2e else None,
        "e2e_min_us": float(e2e.group(2)) if e2e else None,
    }
    line = next((l for l in out.splitlines() if l.startswith("scopes:")), "")
    sample["scopes"] = scopes.parse_scopes(line) if line else None
    seg_line = next((l for l in out.splitlines() if l.startswith("segments:")),
                    "")
    sample["segments"] = (scopes.parse_segments(seg_line)
                          if seg_line else None)
    return sample


def compute_stats(values):
    vals = [v for v in values if _finite(v)]
    if not vals:
        return {"median": None, "min": None, "mean": None, "cv": None,
                "samples": list(values)}
    mean = statistics.fmean(vals)
    cv = (statistics.pstdev(vals) / mean) if mean else None
    return {"median": statistics.median(vals), "min": min(vals),
            "mean": mean, "cv": cv, "samples": list(values)}


# ---------- hard acceptance (pure; unit-tested) --------------------------
def verify_point(p, expect_boundary, threshold=THRESHOLD, trials=True):
    """Explicit problems for one long shape; [] == accepted."""
    problems = []
    if p.get("rc") != 0:
        problems.append(f"rc={p.get('rc')} (want 0)")
    if not p.get("pass"):
        problems.append("overall PASS line missing")
    if p.get("max_rel") is None:
        problems.append("max_rel missing")
    elif not _finite(p["max_rel"]):
        problems.append(f"max_rel not finite: {p['max_rel']!r}")
    elif p["max_rel"] > threshold:
        problems.append(f"max_rel {p['max_rel']} > threshold {threshold}")
    seq = p.get("seq") or []
    if len(seq) != 3:
        problems.append(f"seq has {len(seq)} entries, want exactly 3")
    else:
        got = tuple(s["input"] for s in seq)
        if got != EXPECTED_SEQ:
            problems.append(f"seq order {got} != {EXPECTED_SEQ}")
    for i, s in enumerate(seq):
        if not s.get("pass"):
            problems.append(f"seq[{i}] {s.get('input')} did not PASS")
        if s.get("stale"):
            problems.append(f"seq[{i}] reported STALE-OUTPUT")
        if s.get("max_rel") is None or not _finite(s["max_rel"]):
            problems.append(f"seq[{i}] max_rel not finite: {s.get('max_rel')!r}")
        elif s["max_rel"] > threshold:
            problems.append(f"seq[{i}] max_rel {s['max_rel']} > {threshold}")
    summary = p.get("seq_summary") or ""
    if not summary:
        problems.append("stale-input summary line missing")
    elif not summary.rstrip().endswith("-> PASS"):
        problems.append(f"stale-input summary not successful: {summary!r}")
    transfers = p.get("e2e_transfers") or ""
    if not transfers:
        problems.append("E2E transfers line missing")
    else:
        if "in=1 out=1" not in transfers:
            problems.append(f"logical transfer counts wrong: {transfers!r}")
        if expect_boundary and expect_boundary not in transfers:
            problems.append(f"want {expect_boundary} in {transfers!r}")
    if expect_boundary and not p.get("boundary_ok"):
        problems.append("per-shape boundary_ok is not true")
    if trials:
        t = p.get("trials")
        if not t:
            problems.append("trials block missing")
        else:
            if t.get("count") != TRIALS:
                problems.append(f"trials count {t.get('count')} != {TRIALS}")
            stats = t.get("stats") or {}
            samples = stats.get("samples") or []
            if len(samples) != TRIALS:
                problems.append(
                    f"raw trial samples {len(samples)} != {TRIALS}")
            for i, v in enumerate(samples):
                if not _finite(v) or v <= 0:
                    problems.append(f"trial[{i}] e2e_us invalid: {v!r}")
            for key in ("median", "min", "mean"):
                if not _finite(stats.get(key)):
                    problems.append(f"trial stats missing {key}")
            # P0 six-segment instrumentation: device chains must report the
            # full decomposition (telescoping into device_chain); host chains
            # must keep the explicit NA so the two never get mixed up.
            for i, s in enumerate(t.get("raw") or []):
                seg = (s or {}).get("segments", "unset")
                if expect_boundary == "boundary=0":
                    fields = (s or {}).get("scopes")
                    if not fields or seg in (None, "unset"):
                        problems.append(
                            f"trial[{i}] device chain missing six segments")
                    else:
                        problems += [
                            f"trial[{i}] {x}" for x in
                            scopes.validate_segments(seg, fields,
                                                     "long_device")]
                elif expect_boundary and seg not in (None,):
                    problems.append(
                        f"trial[{i}] host chain must report segments=NA, "
                        f"got {seg!r}")
    return problems


def verify_control(control, threshold=THRESHOLD):
    """Short-path A/B/A control: same dynamic-input contract, no boundary."""
    fake = dict(control)
    fake.setdefault("e2e_transfers", "E2E transfers: in=1 out=1")
    fake["boundary_ok"] = True
    return verify_point(fake, None, threshold=threshold, trials=False)


def verify_e2e(e2e, expect_boundary):
    problems = []
    if e2e.get("rc") != 0:
        problems.append(f"e2e rc={e2e.get('rc')} (want 0)")
    transfers = e2e.get("transfers") or ""
    if "in=1 out=1" not in transfers:
        problems.append(f"e2e logical transfer counts wrong: {transfers!r}")
    if expect_boundary and expect_boundary not in transfers:
        problems.append(f"e2e want {expect_boundary} in {transfers!r}")
    if not e2e.get("line"):
        problems.append("e2E summary line missing")
    return problems


def verify_grid(points, ns=NS, bs=BS):
    problems = []
    want = {(n, b) for n in ns for b in bs}
    got = [(p.get("n"), p.get("b")) for p in points]
    if len(got) != len(set(got)):
        problems.append(f"duplicate (N,batch) keys: {sorted(got)}")
    if set(got) != want:
        problems.append(f"grid mismatch: missing={sorted(want - set(got))} "
                        f"extra={sorted(set(got) - want)}")
    return problems


# ---------- manifest -----------------------------------------------------
def git_state(root=ROOT):
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True).stdout.strip())
    return sha, dirty


def patch_digest(root=ROOT):
    """sha256 over tracked diff + untracked file list (dirty-tree fallback)."""
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=root,
                          capture_output=True).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root,
        capture_output=True, text=True).stdout
    return hashlib.sha256(diff + b"\nuntracked:\n" +
                          untracked.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_cann_version():
    env = os.environ.get("AB_CANN_VERSION")
    if env:
        return env
    target = os.path.realpath("/usr/local/Ascend/ascend-toolkit/latest")
    name = os.path.basename(target)
    if name.startswith("cann-"):
        return name[len("cann-"):]
    if os.path.isdir(target):
        return name
    return "unknown"


def detect_driver():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    m = re.search(r"Driver Version\s*[:=]\s*(\S+)", out)
    return m.group(1) if m else "unknown"


def build_manifest(mode, allow_dirty):
    sha, dirty = git_state()
    man = {
        "git_sha": sha,
        "git_dirty": dirty,
        "binary": "build/fft_check",
        "binary_sha256": sha256_file(ROOT / "build" / "fft_check"),
        "build": {
            "script": "bash scripts/build.sh",
            "recipe": "scripts/env.sh: ab_ccec (ccec -c --cce-aicore-only "
                      "--npu-soc=$AB_SOC --asc-aicore-lang -O2 -std=c++17 "
                      "-Iinclude + AB_INC) / ab_cxx (g++ -O2 -std=c++17 "
                      "-I$AB_CANN/include -Iinclude -lascendcl)",
        },
        "soc": os.environ.get("AB_SOC", "Ascend910_9382"),
        "cann_version": detect_cann_version(),
        "driver_version": detect_driver(),
        "python": sys.version.split()[0],
        "env": {k: v for k, v in sorted(os.environ.items())
                if k.startswith("AB_")},
        "boundary": mode,
        "allow_dirty": bool(allow_dirty),
    }
    if dirty:
        man["git_patch_sha256"] = patch_digest()
    return man


# ---------- collection ---------------------------------------------------
def mode_env(mode, **extra):
    env = dict(os.environ)
    env.update(BASE_ENV)
    if mode == "device":
        env["AB_BOUNDARY"] = "device"
    env.update(extra)
    return env


def collect(mode, allow_dirty=False):
    expect_boundary = BOUNDARY_BY_MODE[mode]
    binary = ROOT / "build" / "fft_check"
    if not binary.is_file():
        return None, ["build/fft_check missing; run scripts/build.sh check"]
    manifest = build_manifest(mode, allow_dirty)
    points, transcripts = [], []
    commands = []
    for n in NS:
        for b in BS:
            seq_cmd = ["./build/fft_check", str(n), str(b), "3"]
            if not commands:
                commands.append({"kind": "aba_seq", "argv": seq_cmd,
                                 "env": [
                                     "AB_INPUT_SEQ=impulse,random-seeded,impulse",
                                     "AB_E2E=1"] +
                                    (["AB_BOUNDARY=device"]
                                     if mode == "device" else []),
                                 "reps": 3, "note": "first shape"})
            rc, out = run(seq_cmd, env=mode_env(
                mode, AB_INPUT_SEQ="impulse,random-seeded,impulse"))
            point = {"n": n, "b": b, "rc": rc, **parse_point(out)}
            point["boundary_ok"] = (expect_boundary in
                                    point["e2e_transfers"])
            # 5 independent metric trials: one invocation each, raw kept
            trial_cmd = ["./build/fft_check", str(n), str(b), "5"]
            samples = []
            for _ in range(TRIALS):
                rct, outt = run(trial_cmd, env=mode_env(mode, AB_E2E="5"))
                if rct != 0 or not commands or \
                        commands[-1]["kind"] != "trial":
                    commands.append(
                        {"kind": "trial", "argv": trial_cmd,
                         "env": ["AB_E2E=5"] + (["AB_BOUNDARY=device"]
                                                if mode == "device" else []),
                         "reps": 5, "note": "repeated per shape"})
                sample = parse_trial(outt)
                sample["rc"] = rct
                samples.append(sample)
            stats = compute_stats([s["e2e_us"] for s in samples])
            point["trials"] = {"count": TRIALS, "raw": samples,
                               "stats": stats}
            points.append(point)
            transcripts.append(
                f"===== long A/B/A n={n} b={b} rc={rc} =====\n{out}")
    rc, out = run(["./build/fft_check", "4096", "3", "3"],
                  env=mode_env(mode,
                               AB_INPUT_SEQ="impulse,random-seeded,impulse"))
    control = {"n": 4096, "b": 3, "rc": rc, **parse_point(out),
               "path": "short"}
    transcripts.append(f"===== short A/B/A control rc={rc} =====\n{out}")

    eenv = mode_env(mode, AB_E2E="3")
    e2e_cmd = ["./build/fft_check", "8192", "1", "3"]
    rc, out = run(e2e_cmd, env=eenv)
    e2e = {"rc": rc,
           "transfers": (re.search(r"^(E2E transfers: .*)$", out, re.M)
                         .group(1)
                         if re.search(r"^E2E transfers: .*$", out, re.M)
                         else ""),
           "line": (re.search(r"^(E2E n=.*)$", out, re.M).group(1)
                    if re.search(r"^E2E n=.*$", out, re.M) else "")}
    commands.append({"kind": "e2e", "argv": e2e_cmd,
                     "env": ["AB_E2E=3"] + (["AB_BOUNDARY=device"]
                                            if mode == "device" else []),
                     "reps": 3})
    transcripts.append(f"===== E2E transfer assertion rc={rc} =====\n{out}")

    problems = []
    problems += [f"grid: {p}" for p in verify_grid(points)]
    for p in points:
        problems += [f"point n={p['n']} b={p['b']}: {x}"
                     for x in verify_point(p, expect_boundary)]
    problems += [f"control: {x}" for x in verify_control(control)]
    problems += [f"e2e: {x}" for x in verify_e2e(e2e, expect_boundary)]

    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "commands": commands,
        "command": "python3 scripts/collect_long_fft_evidence.py"
                   + (" --boundary device" if mode == "device" else ""),
        "binary": "build/fft_check (AB_INPUT_SEQ named input modes)",
        "boundary": mode,
        "threshold": THRESHOLD,
        "grid": {"ns": list(NS), "bs": list(BS)},
        "trials_per_shape": TRIALS,
        "points": points,
        "short_control": control,
        "e2e_transfers": e2e,
        "problems": problems,
        "status": "pass" if not problems else "fail",
    }
    return (document, transcripts), problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary", choices=("host", "device"), default="host")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="record a patch digest instead of refusing on a "
                         "dirty tree")
    args = ap.parse_args(argv)

    sha, dirty = git_state()
    if dirty and not args.allow_dirty:
        print("refusing to publish evidence from a dirty tree "
              f"({sha[:8]}; commit first or pass --allow-dirty)",
              file=sys.stderr)
        return 2

    result, problems = collect(args.boundary, allow_dirty=args.allow_dirty)
    if result is None:
        for p in problems:
            print(p, file=sys.stderr)
        return 2
    document, transcripts = result
    out_dir = ROOT / "results" / "evidence" / OUT_BY_MODE[args.boundary]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs.txt").write_text("\n".join(transcripts),
                                         encoding="utf-8")
    (out_dir / "acceptance.json").write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    points = document["points"]
    print(f"{sum(1 for p in points if p['pass'])}/{len(points)} long points, "
          f"control={'PASS' if document['short_control']['pass'] else 'FAIL'}, "
          f"trials={TRIALS}x{len(points)}, "
          f"e2e={document['e2e_transfers']['transfers']!r} -> "
          f"{document['status']} -> {out_dir}")
    if problems:
        for p in problems:
            print(f"  problem: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
