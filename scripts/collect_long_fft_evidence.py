#!/usr/bin/env python3
"""Collect the long-FFT dynamic-input acceptance evidence (P1-A / G1 envelope).

Runs the committed `fft_check` binary over the G1 grid
N={8192,16384,32768,65536} x B={1,3,47} with an A/B/A input sequence from raw
float32 files (file contents change between calls), plus a short-path A/B/A
control and the E2E single-input/single-output transfer assertion.  Every
shape also gets 5 independent metric trials (one process invocation each,
raw samples retained, median/min/mean/CV reported).

Hard acceptance (PR #2 stage 4 / R0.2 contract, verify_* functions are
unit-tested in tests/test_collect_evidence.py):
  - every long shape exits rc == 0;
  - exactly three seq entries in `impulse, random-seeded, impulse` order,
    each PASS, finite, <= threshold, never STALE-OUTPUT;
  - the stale-input summary line exists and ends with `-> PASS`;
  - the 12 (N, batch) keys are complete and unique;
  - transfer/boundary counters match the selected path (host boundary=2,
    device boundary=0), both per shape and on the dedicated E2E run;
  - trials: len(raw) is authoritative (count is derived, not trusted),
    every raw rc==0, PASS/maxRel captured and under threshold, all timing
    fields finite, host and device scopes validated via scopes.validate_scopes,
    and stats are recomputed from raw (the cached block is output-only);
  - PR-B impl contract: device chains report the impl-matched segment
    decomposition (six spans separate / five fused, telescoping into
    device_chain) plus a matching `boundary_impl:` self-report; host
    chains report segments=NA and no boundary_impl line;
  - manifest: clean git tree is required (or --allow-dirty records a patch
    digest), sha256(build/fft_check), sha256 of the runtime kernel objects
    (fft_radix2.o / fft_long.o / fft_real.o), sha256 of config/*.json, build
    recipe, CANN/SoC/driver (+ driver source), effective AB_* environment and
    the exact commands;
  - a driver version that cannot be resolved from any source marks the
    document `status="incomplete"` (non-zero exit unless --allow-incomplete)
    instead of silently recording "unknown";
  - verify_document() re-runs the whole contract on a stored acceptance.json
    (archive-level gate used by CI).

  python3 scripts/collect_long_fft_evidence.py               # 宿主中介链（默认）
  python3 scripts/collect_long_fft_evidence.py --boundary device
      # addendum §3 device-materialized 段边界（AB_BOUNDARY=device）：
      # 同一网格 + A/B/A，段边界不回宿主 => 期望 E2E boundary=0，
      # 证据写入 results/evidence/long-fft-device-boundary/。
  python3 scripts/collect_long_fft_evidence.py --boundary device-fused
      # PR-B R1：device 链 + AB_LONG_BOUNDARY_IMPL=fused（twiddle 并入段
      # 边界转置，5 发射 5 段），证据写入
      # results/evidence/long-fft-device-boundary-fused/（不覆盖旧档）。
  --allow-dirty       记录 patch digest 后继续（默认脏树直接拒绝发布证据）
  --allow-incomplete  driver/哈希不全时仍发布（status 保持 incomplete）
  --verify PATH       对已存 acceptance.json 跑归档级校验后退出
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
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
# host 模式 = 宿主中介链 boundary=2（行为锁定）；device/device-fused 模式 =
# 不回宿主 boundary=0，fused 额外锁定 AB_LONG_BOUNDARY_IMPL（mode_env 统一置）。
BASE_ENV = {"AB_E2E": "1"}
BOUNDARY_BY_MODE = {"host": "boundary=2", "device": "boundary=0",
                    "device-fused": "boundary=0"}
OUT_BY_MODE = {"host": "long-fft-acceptance",
               "device": "long-fft-device-boundary",
               "device-fused": "long-fft-device-boundary-fused"}
# PR-B R1：每种采集模式对应的段边界发射形态（fused 只在 device-fused 出现）。
IMPL_BY_MODE = {"host": "separate", "device": "separate",
                "device-fused": "fused"}


def run(args, env=None):
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          env=env, timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


def env_monitor_snapshot():
    """Best-effort host + NPU state snapshot (PR-B review R1.1).

    Records what the machine actually exposes: loadavg, uptime, and
    npu-smi power / temperature / AICore% / HBM per NPU and die.
    Unavailable fields are the string 'unavailable' -- never estimated.
    This is a variance-governance side channel for timing runs; profiler
    runs stay separate from timing runs by design."""
    snap = {"utc": datetime.now(timezone.utc).isoformat()}
    try:
        l1, l5, l15 = os.getloadavg()
        snap["loadavg"] = {"1m": round(l1, 2), "5m": round(l5, 2),
                           "15m": round(l15, 2)}
    except (OSError, AttributeError):
        snap["loadavg"] = "unavailable"
    try:
        with open("/proc/uptime", encoding="ascii") as fh:
            snap["uptime_s"] = float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        snap["uptime_s"] = "unavailable"
    smi = shutil.which("npu-smi")
    if not smi:
        snap["npu_smi"] = "unavailable"
        return snap
    try:
        proc = subprocess.run([smi, "info"], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        snap["npu_smi"] = "unavailable"
        return snap
    if proc.returncode != 0:
        snap["npu_smi"] = "unavailable"
        return snap
    npus, chips = [], []
    for line in proc.stdout.splitlines():
        if not line.startswith("|"):
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if not parts:
            continue
        head = parts[0].split()
        if not head or not head[0].isdigit():
            continue          # header rows ("NPU Name ...", "Chip Phy ...")
        npu_id = int(head[0])
        if len(parts) >= 3 and parts[1] in ("OK", "-"):
            # NPU row: "idx NAME" | health | "power(W) temp(C) hugepages"
            fields = parts[2].split()
            power = fields[0] if fields else ""
            temp = fields[1] if len(fields) > 1 else ""
            npus.append({
                "npu": npu_id,
                "power_w": (float(power) if power not in ("", "-")
                            else "unavailable"),
                "temp_c": (int(temp) if temp not in ("", "-")
                           else "unavailable"),
            })
        elif len(parts) >= 3 and ":" in parts[1]:
            # chip row: "idx PHY" | bus-id | "aicore% memory" | hbm
            aicore = parts[2].split()[0] if parts[2] else ""
            chips.append({
                "npu": npu_id,
                "phy": (int(head[1]) if len(head) > 1
                        and head[1].isdigit() else -1),
                "aicore_pct": (float(aicore) if aicore not in ("", "-")
                               else "unavailable"),
                "hbm_usage": parts[3] if len(parts) > 3 else "unavailable",
            })
    snap["npu_smi"] = {"npus": npus, "chips": chips,
                       "frequency": "unavailable"}
    return snap


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
    """One independent metric invocation -> raw trial sample.

    Besides timing/scopes, captures the correctness lines the same run
    prints (`n=... maxRel=...` + terminal PASS) so per-trial accuracy is
    part of the raw archive instead of being assumed.
    """
    e2e = re.search(r"^E2E n=.*e2e_us=([\d.eE+-]+) e2e_min_us=([\d.eE+-]+)",
                    out, re.M)
    corr = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    sample = {
        "e2e_us": float(e2e.group(1)) if e2e else None,
        "e2e_min_us": float(e2e.group(2)) if e2e else None,
        "max_abs": float(corr.group(1)) if corr else None,
        "max_rel": float(corr.group(2)) if corr else None,
        "pass": bool(re.search(r"^PASS$", out, re.M)),
    }
    line = next((l for l in out.splitlines() if l.startswith("scopes:")), "")
    sample["scopes"] = scopes.parse_scopes(line) if line else None
    seg_line = next((l for l in out.splitlines() if l.startswith("segments:")),
                    "")
    sample["segments"] = (scopes.parse_segments(seg_line)
                          if seg_line else None)
    # PR-B：device 链自报发射形态（5/6 段归属的运行时佐证）；宿主/短路径无此行。
    bi_line = next((l for l in out.splitlines()
                    if l.startswith("boundary_impl:")), "")
    if bi_line:
        scopes.parse_boundary_impl(bi_line)   # malformed => ValueError
        sample["boundary_impl"] = bi_line.split(":", 1)[1].strip()
    else:
        sample["boundary_impl"] = None
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
def verify_point(p, expect_boundary, threshold=THRESHOLD, trials=True,
                 impl="separate"):
    """Explicit problems for one long shape; [] == accepted.

    `impl` (PR-B) selects the segment contract: separate device chains
    must report six spans, fused chains five (no twiddle), and every raw
    device trial must self-report the matching `boundary_impl:` line.
    """
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
            raw = t.get("raw") or []
            if len(raw) != TRIALS:
                problems.append(f"trials.raw has {len(raw)} entries, "
                                f"want {TRIALS}")
            if t.get("count") != len(raw):
                problems.append(f"trials count {t.get('count')} != "
                                f"len(raw) {len(raw)}")
            elif t.get("count") != TRIALS:
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
            # Per-raw contract (R0.2): rc, correctness, finite timings and
            # validated scopes on every archived sample — never assumed.
            scope_mode = ("long_device" if expect_boundary == "boundary=0"
                          else "long_host")
            for i, s in enumerate(raw):
                s = s or {}
                if s.get("rc") != 0:
                    problems.append(f"trial[{i}] rc={s.get('rc')} (want 0)")
                for field in ("e2e_us", "e2e_min_us"):
                    v = s.get(field)
                    if not _finite(v) or v <= 0:
                        problems.append(
                            f"trial[{i}] raw {field} invalid: {v!r}")
                if not s.get("pass"):
                    problems.append(f"trial[{i}] PASS line missing")
                mr = s.get("max_rel")
                if mr is None:
                    problems.append(f"trial[{i}] max_rel missing")
                elif not _finite(mr):
                    problems.append(f"trial[{i}] max_rel not finite: {mr!r}")
                elif mr > threshold:
                    problems.append(f"trial[{i}] max_rel {mr} > "
                                    f"threshold {threshold}")
                fields = s.get("scopes")
                if not fields:
                    problems.append(f"trial[{i}] scopes missing")
                else:
                    problems += [f"trial[{i}] {x}" for x in
                                 scopes.validate_scopes(fields, scope_mode)]
                    for name, v in fields.items():
                        if isinstance(v, (int, float)) and not _finite(v):
                            problems.append(
                                f"trial[{i}] scope {name} not finite: {v!r}")
                # PR-B segment instrumentation: device chains must report
                # the impl-matched decomposition (six spans separate, five
                # fused, telescoping into device_chain) plus the matching
                # boundary_impl self-report; host chains keep explicit NA
                # and no boundary_impl line.
                seg = s.get("segments", "unset")
                bi = s.get("boundary_impl")
                if expect_boundary == "boundary=0":
                    want = "six" if impl == "separate" else "five"
                    if not fields or seg in (None, "unset"):
                        problems.append(
                            f"trial[{i}] device chain missing {want} "
                            f"segments ({impl})")
                    else:
                        problems += [
                            f"trial[{i}] {x}" for x in
                            scopes.validate_segments(seg, fields,
                                                     "long_device",
                                                     impl=impl)]
                    # attest only when the trial actually carries the key:
                    # parse_trial always records it for new collections
                    # (None when the line was absent), while pre-PR-B
                    # archives predate it entirely.
                    if "boundary_impl" in s and bi != impl:
                        problems.append(
                            f"trial[{i}] boundary_impl {bi!r} != {impl!r}")
                else:
                    if expect_boundary and seg not in (None,):
                        problems.append(
                            f"trial[{i}] host chain must report "
                            f"segments=NA, got {seg!r}")
                    if bi is not None:
                        problems.append(
                            f"trial[{i}] host chain must not report "
                            f"boundary_impl, got {bi!r}")
            # Stats are an output cache: recompute from raw so a tampered
            # median/min/mean/cv can never drift from the archived samples.
            recomputed = compute_stats([(s or {}).get("e2e_us")
                                        for s in raw])
            for key in ("median", "min", "mean", "cv"):
                a, b = stats.get(key), recomputed.get(key)
                if a is None and b is None:
                    continue
                if (a is None or b is None
                        or not math.isclose(a, b, rel_tol=1e-9,
                                            abs_tol=1e-9)):
                    problems.append(
                        f"stats {key}={a!r} != recomputed {b!r}")
            if samples != recomputed.get("samples"):
                problems.append("stats samples != raw e2e_us list")
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


# ---------- archive-level gate (pure; unit-tested) ------------------------
def verify_document(doc, threshold=None):
    """Re-run the whole contract on a stored acceptance.json.

    Returns [] only for a fully attested, passing archive (status "pass"
    with every stored verdict recomputed identically). Tampered stats/raw
    samples, malformed manifests and provenance gaps all yield explicit
    problems; used as the CI gate on the committed archives.
    """
    if isinstance(doc, (str, Path)):
        doc = json.loads(Path(doc).read_text(encoding="utf-8"))
    missing = [k for k in ("manifest", "points", "grid", "boundary",
                           "trials_per_shape", "short_control",
                           "e2e_transfers", "status", "env_monitor")
               if k not in doc]
    if missing:
        return [f"missing top-level key {k!r}" for k in missing]

    def _snap_problem(snap, where):
        if not isinstance(snap, dict) or not isinstance(snap.get("utc"),
                                                        str):
            return [f"{where}: env_monitor snapshot missing utc"]
        probs = []
        if "npu_smi" not in snap:
            probs.append(f"{where}: env_monitor snapshot missing npu_smi")
        if "loadavg" not in snap:
            probs.append(f"{where}: env_monitor snapshot missing loadavg")
        return probs

    problems = []
    for where, snap in (("env_monitor.before",
                         (doc.get("env_monitor") or {}).get("before")),
                        ("env_monitor.after",
                         (doc.get("env_monitor") or {}).get("after"))):
        problems += _snap_problem(snap, where)

    boundary = doc["boundary"]
    expect = BOUNDARY_BY_MODE.get(boundary)
    if expect is None:
        return [f"unknown boundary {boundary!r}"]
    # PR-B：impl 归属（无 impl 键的旧档 = separate，天然向后兼容）；档头
    # boundary 与 impl 必须互相印证，防止单方面改键绕过段数契约。
    impl = doc.get("impl", "separate")
    if impl not in scopes.IMPLS:
        return [f"unknown impl {impl!r} (expected one of {scopes.IMPLS})"]
    want_impl = IMPL_BY_MODE.get(boundary)
    if want_impl is not None and impl != want_impl:
        problems.append(f"impl {impl!r} != {want_impl!r} for "
                        f"boundary {boundary!r}")
    thr = doc.get("threshold", THRESHOLD) if threshold is None else threshold
    grid = doc.get("grid") or {}
    problems += [f"grid: {x}" for x in
                 verify_grid(doc["points"], tuple(grid.get("ns") or ()),
                             tuple(grid.get("bs") or ()))]
    for p in doc["points"]:
        pem = p.get("env_monitor") or {}
        for w in ("before", "after"):
            problems += _snap_problem(
                pem.get(w), f"point n={p.get('n')} b={p.get('b')} "
                            f"env_monitor.{w}")
        problems += [f"point n={p.get('n')} b={p.get('b')}: {x}"
                     for x in verify_point(p, expect, threshold=thr,
                                           impl=impl)]
    problems += [f"control: {x}" for x in
                 verify_control(doc["short_control"], threshold=thr)]
    problems += [f"e2e: {x}" for x in
                 verify_e2e(doc["e2e_transfers"], expect)]
    problems += [f"manifest: {x}" for x in
                 verify_manifest(doc["manifest"], boundary)]
    incomplete = manifest_incomplete(doc["manifest"])
    status = ("fail" if problems else
              "incomplete" if incomplete else "pass")
    out = list(problems)
    if status == "incomplete":
        out += [f"incomplete: {x}" for x in incomplete]
    # Self-consistency: stored verdicts must equal the recomputation.
    if doc.get("status") != status:
        out.append(f"status {doc.get('status')!r} != recomputed {status!r}")
    if list(doc.get("problems") or []) != problems:
        out.append("stored problems list != recomputation "
                   f"({len(doc.get('problems') or [])} vs {len(problems)})")
    if list(doc.get("incomplete") or []) != incomplete:
        out.append("stored incomplete list != recomputation: "
                   f"{doc.get('incomplete')!r} vs {incomplete!r}")
    return out


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


def driver_info():
    """(version, source) of the NPU driver, multi-source.

    Order: AB_DRIVER_VERSION env -> /usr/local/Ascend/driver/version.info
    (`Version=`) -> `npu-smi info` regex. source is None only when every
    source fails: callers must treat that as `status="incomplete"` evidence
    rather than silently publishing "unknown".
    """
    env = os.environ.get("AB_DRIVER_VERSION")
    if env:
        return env, "env:AB_DRIVER_VERSION"
    info = Path("/usr/local/Ascend/driver/version.info")
    try:
        for line in info.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version="):
                v = line.split("=", 1)[1].strip()
                if v:
                    return v, str(info)
    except OSError:
        pass
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    m = re.search(r"Driver Version\s*[:=]\s*(\S+)", out)
    if m:
        return m.group(1), "npu-smi"
    return "unknown", None


def detect_driver():
    return driver_info()[0]


def kernel_object_paths():
    """Runtime-loaded device objects as (name -> relative path)."""
    return {
        "fft_radix2.o": os.environ.get("AB_FFT_O", "build/fft_radix2.o"),
        "fft_long.o": os.environ.get("AB_LONG_O", "build/fft_long.o"),
        "fft_real.o": os.environ.get("AB_REAL_O", "build/fft_real.o"),
    }


def manifest_incomplete(man):
    """Evidence-completeness reasons (content-level, not structure).

    Non-empty => the document must be `status="incomplete"`: the numbers may
    be valid but provenance cannot be fully attested (R0.2).
    """
    reasons = []
    if "driver_source" not in man:
        reasons.append("driver_source key missing from manifest")
    elif not man.get("driver_source"):
        reasons.append("driver version unrecognized "
                       "(AB_DRIVER_VERSION/version.info/npu-smi all failed)")
    kos = man.get("kernel_objects")
    if not isinstance(kos, dict):
        reasons.append("kernel_objects missing")
    else:
        for name in sorted(set(kernel_object_paths()) - set(kos)):
            reasons.append(f"kernel object {name} not hashed")
        for name, v in sorted(kos.items()):
            if v is None:
                reasons.append(f"kernel object {name} hash unavailable")
    if not man.get("config_sha256"):
        reasons.append("config_sha256 empty")
    return reasons


def verify_manifest(man, boundary):
    """Structural manifest contract; [] == well-formed (content completeness
    is manifest_incomplete()'s job)."""
    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", str(man.get("git_sha") or "")):
        problems.append(f"git_sha malformed: {man.get('git_sha')!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(man.get("binary_sha256") or "")):
        problems.append(f"binary_sha256 malformed: "
                        f"{man.get('binary_sha256')!r}")
    kos = man.get("kernel_objects")
    if not isinstance(kos, dict) or set(kos) != set(kernel_object_paths()):
        problems.append(f"kernel_objects keys wrong: "
                        f"{sorted(kos) if isinstance(kos, dict) else kos!r}")
    else:
        for name, v in sorted(kos.items()):
            if v is not None and not re.fullmatch(r"[0-9a-f]{64}", str(v)):
                problems.append(f"kernel_objects[{name}] malformed: {v!r}")
    cfg = man.get("config_sha256")
    if not isinstance(cfg, dict) or not cfg:
        problems.append(f"config_sha256 missing/empty: {cfg!r}")
    else:
        for name, v in sorted(cfg.items()):
            if not re.fullmatch(r"[0-9a-f]{64}", str(v)):
                problems.append(f"config_sha256[{name}] malformed: {v!r}")
    if "driver_source" not in man:
        problems.append("driver_source key missing from manifest")
    if "driver_version" not in man:
        problems.append("driver_version key missing from manifest")
    if not man.get("cann_version"):
        problems.append("cann_version missing")
    if man.get("boundary") != boundary:
        problems.append(f"manifest boundary {man.get('boundary')!r} != "
                        f"document {boundary!r}")
    if not man.get("build", {}).get("recipe"):
        problems.append("build recipe missing")
    if not isinstance(man.get("env"), dict):
        problems.append("env block missing")
    return problems


def build_manifest(mode, allow_dirty):
    sha, dirty = git_state()
    driver_version, driver_source = driver_info()
    man = {
        "git_sha": sha,
        "git_dirty": dirty,
        "binary": "build/fft_check",
        "binary_sha256": sha256_file(ROOT / "build" / "fft_check"),
        "kernel_objects": {name: sha256_file(ROOT / rel)
                           for name, rel in kernel_object_paths().items()},
        "config_sha256": {p.name: sha256_file(p)
                          for p in sorted((ROOT / "config").glob("*.json"))},
        "build": {
            "script": "bash scripts/build.sh",
            "recipe": "scripts/env.sh: ab_ccec (ccec -c --cce-aicore-only "
                      "--npu-soc=$AB_SOC --asc-aicore-lang -O2 -std=c++17 "
                      "-Iinclude + AB_INC) / ab_cxx (g++ -O2 -std=c++17 "
                      "-I$AB_CANN/include -Iinclude -lascendcl)",
        },
        "soc": os.environ.get("AB_SOC", "Ascend910_9382"),
        "cann_version": detect_cann_version(),
        "driver_version": driver_version,
        "driver_source": driver_source,
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
def env_tags(mode):
    """Env contract recorded next to each archived command argv."""
    tags = []
    if mode in ("device", "device-fused"):
        tags.append("AB_BOUNDARY=device")
    tags.append(f"AB_LONG_BOUNDARY_IMPL={IMPL_BY_MODE[mode]}")
    # R2-A experiment overrides (ambient passthrough): record them so an
    # archive never claims the default tile/blocks when a sweep set them.
    for k in ("AB_LT_TILE", "AB_LT_BLOCKS"):
        v = os.environ.get(k)
        if v:
            tags.append(f"{k}={v}")
    return tags


def control_env(mode, **extra):
    """Env for the short-path control run (every mode).

    The impl switch only exists for the long device chain; fft_check
    rejects fused+short as a misconfig, and the control never runs that
    chain -- pin the default for device-fused rather than stripping.
    """
    pin = ({"AB_LONG_BOUNDARY_IMPL": "separate"}
           if mode == "device-fused" else {})
    return mode_env(mode, AB_INPUT_SEQ="impulse,random-seeded,impulse",
                    **pin, **extra)


def mode_env(mode, **extra):
    env = dict(os.environ)
    env.update(BASE_ENV)
    if mode in ("device", "device-fused"):
        env["AB_BOUNDARY"] = "device"
    # pin the impl for every mode: host runs must not inherit a stray
    # AB_LONG_BOUNDARY_IMPL=fused from the ambient environment (fft_check
    # rejects fused+host), and each device mode gets its contracted shape.
    env["AB_LONG_BOUNDARY_IMPL"] = IMPL_BY_MODE[mode]
    env.update(extra)
    return env


def collect(mode, allow_dirty=False):
    expect_boundary = BOUNDARY_BY_MODE[mode]
    impl = IMPL_BY_MODE[mode]
    binary = ROOT / "build" / "fft_check"
    if not binary.is_file():
        return None, ["build/fft_check missing; run scripts/build.sh check"]
    manifest = build_manifest(mode, allow_dirty)
    points, transcripts = [], []
    commands = []
    env_before = env_monitor_snapshot()
    for n in NS:
        for b in BS:
            point_env_before = env_monitor_snapshot()
            seq_cmd = ["./build/fft_check", str(n), str(b), "3"]
            if not commands:
                commands.append({"kind": "aba_seq", "argv": seq_cmd,
                                 "env": [
                                     "AB_INPUT_SEQ=impulse,random-seeded,impulse",
                                     "AB_E2E=1"] + env_tags(mode),
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
                         "env": ["AB_E2E=5"] + env_tags(mode),
                         "reps": 5, "note": "repeated per shape"})
                sample = parse_trial(outt)
                sample["rc"] = rct
                samples.append(sample)
            stats = compute_stats([s["e2e_us"] for s in samples])
            point["trials"] = {"count": TRIALS, "raw": samples,
                               "stats": stats}
            point["env_monitor"] = {"before": point_env_before,
                                    "after": env_monitor_snapshot()}
            points.append(point)
            transcripts.append(
                f"===== long A/B/A n={n} b={b} rc={rc} =====\n{out}")
    rc, out = run(["./build/fft_check", "4096", "3", "3"],
                  env=control_env(mode))
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
                     "env": ["AB_E2E=3"] + env_tags(mode),
                     "reps": 3})
    transcripts.append(f"===== E2E transfer assertion rc={rc} =====\n{out}")

    problems = []
    problems += [f"grid: {p}" for p in verify_grid(points)]
    for p in points:
        problems += [f"point n={p['n']} b={p['b']}: {x}"
                     for x in verify_point(p, expect_boundary, impl=impl)]
    problems += [f"control: {x}" for x in verify_control(control)]
    problems += [f"e2e: {x}" for x in verify_e2e(e2e, expect_boundary)]
    incomplete = manifest_incomplete(manifest)

    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "commands": commands,
        "command": "python3 scripts/collect_long_fft_evidence.py"
                   + (f" --boundary {mode}" if mode != "host" else ""),
        "binary": "build/fft_check (AB_INPUT_SEQ named input modes)",
        "boundary": mode,
        "impl": impl,
        "threshold": THRESHOLD,
        "grid": {"ns": list(NS), "bs": list(BS)},
        "trials_per_shape": TRIALS,
        # R1.1 variance governance: host/NPU state around the whole
        # collection (per-point before/after lives on each point).
        "env_monitor": {"before": env_before,
                        "after": env_monitor_snapshot()},
        "points": points,
        "short_control": control,
        "e2e_transfers": e2e,
        "problems": problems,
        "incomplete": incomplete,
        "status": ("fail" if problems else
                   "incomplete" if incomplete else "pass"),
    }
    return (document, transcripts), problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boundary",
                    choices=("host", "device", "device-fused"),
                    default="host")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="record a patch digest instead of refusing on a "
                         "dirty tree")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="publish and exit 0 even when provenance is "
                         "incomplete (status stays 'incomplete')")
    ap.add_argument("--verify", metavar="PATH",
                    help="verify a stored acceptance.json "
                         "(archive-level gate) and exit")
    args = ap.parse_args(argv)

    if args.verify:
        probs = verify_document(Path(args.verify))
        if probs:
            for x in probs:
                print(x, file=sys.stderr)
            print(f"{args.verify}: {len(probs)} problem(s)",
                  file=sys.stderr)
            return 1
        print(f"{args.verify}: archive accepted")
        return 0

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
    if document["status"] == "incomplete":
        for r in document["incomplete"]:
            print(f"  incomplete: {r}", file=sys.stderr)
        if not args.allow_incomplete:
            print("evidence incomplete; rerun with --allow-incomplete to "
                  "publish with status=incomplete anyway", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
