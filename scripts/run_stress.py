#!/usr/bin/env python3
"""Budgeted stress runner for the `stress` profile in config/test_matrix.json.

The generic correctness runner has no memory model, so this dedicated runner owns the
budgeted layer: shapes derive from the declared input-buffer tiers, host/device
headroom is reserved before any allocation is attempted, a canonical shape soaks
`repeat_execution` plan reuses while sampling process RSS, process-level lifecycle
cycles plus a forced-OOM recovery probe prove failures stay recoverable, and every
refused row records its budget numbers so skips remain explainable.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_test_profile import DEFAULT_CONFIG, command_for, load_config

MIB = 1024 ** 2
GIB = 1024 ** 3
OFFSET_LIMIT = 1 << 32
DEVICE_FACTOR = 2.5
DEVICE_FIXED = 64 * MIB
HOST_FACTOR = 2.2
HOST_FIXED = 256 * MIB
QUICK_REPS = 3
RSS_SAMPLE_S = 0.5
RSS_MIN_WALL_S = 20.0
RSS_MIN_SAMPLES = 8
FIELDS = ("stage", "tier_bytes", "direction", "n", "batch", "reps", "input",
          "input_bytes", "host_avail", "device_free", "device_total",
          "host_needed", "device_needed", "decision", "reason", "returncode",
          "wall_ms", "max_rel", "rss_growth_mb", "correct", "failure")

OOM_PROBE = """
import sys
import torch
import torch_npu

torch.npu.set_device(0)
chunks = []
probe = None
c = None
failed = False
while not failed and len(chunks) < 160:
    try:
        c = torch.empty(1 << 30, dtype=torch.uint8, device="npu")
        c.zero_()
        chunks.append(c)
    except RuntimeError:
        failed = True
if len(chunks) >= 160:
    print("NO_OOM")
    sys.exit(1)
if not chunks:
    print("FIRST_ALLOC_FAILED")
    sys.exit(1)
chunks.clear()
c = None
torch.npu.empty_cache()
probe = torch.zeros(1 << 20, dtype=torch.uint8, device="npu")
del probe
print("RECOVERED")
"""


def read_host_available(path="/proc/meminfo"):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable missing from meminfo")


def parse_hbm_table(text, chip=0):
    for line in text.splitlines():
        match = re.match(r"\|\s*\d+\s+(\d+)\s+\|", line)
        if not match or int(match.group(1)) != chip:
            continue
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if pairs:
            used_mb, total_mb = pairs[-1]
            return int(used_mb) * MIB, int(total_mb) * MIB
    raise ValueError(f"cannot parse HBM row for chip {chip} from npu-smi table")


def read_hbm(chip=0):
    result = subprocess.run(["npu-smi", "info"], cwd=ROOT, capture_output=True,
                            text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"npu-smi info failed: {result.stderr.strip()}")
    return parse_hbm_table(result.stdout, chip)


def input_bytes_per_elem(direction):
    return 4 if direction == "r2c" else 8


def derive_batch(n, tier_bytes, per_elem):
    if n < 1 or tier_bytes < 1 or per_elem < 1:
        return 0
    return tier_bytes // (n * per_elem)


def budget_case(input_bytes, host_avail, device_free, device_total):
    device_headroom = max(4 * GIB, int(0.05 * device_total))
    host_headroom = max(2 * GIB, int(0.02 * host_avail))
    device_needed = int(input_bytes * DEVICE_FACTOR) + DEVICE_FIXED
    host_needed = int(input_bytes * HOST_FACTOR) + HOST_FIXED
    budget = {"decision": "run", "reason": "", "host_needed": host_needed,
              "device_needed": device_needed, "device_headroom": device_headroom,
              "host_headroom": host_headroom}
    if input_bytes > OFFSET_LIMIT - MIB:
        budget.update(decision="skip",
                      reason=f"offset-limit: input {input_bytes} B has no uint32 "
                             f"byte-offset headroom below {OFFSET_LIMIT} B")
        return budget
    if device_needed > device_free - device_headroom:
        budget.update(decision="skip",
                      reason=f"device budget: need {device_needed} B > free "
                             f"{device_free} B - headroom {device_headroom} B")
        return budget
    if host_needed > host_avail - host_headroom:
        budget.update(decision="skip",
                      reason=f"host budget: need {host_needed} B > available "
                             f"{host_avail} B - headroom {host_headroom} B")
        return budget
    return budget


def case_timeout_s(reps):
    return float(max(300, min(1800, 60 + reps // 10)))


def plan(profile, host_avail, device_free, device_total):
    c2c_ns = [int(n) for n in profile.get("c2c", {}).get("ns", [])]
    real = profile.get("real", {})
    tiers = [int(tier) for tier in profile.get("input_buffer_bytes", [])]
    soak_reps = int(profile.get("repeat_execution", 10000))
    soak_n = 1024 if 1024 in c2c_ns else (c2c_ns[len(c2c_ns) // 2] if c2c_ns else 0)
    rows = []

    def add(tier, stage, direction, n, reps, per_elem, input_mode):
        batch = derive_batch(n, tier, per_elem)
        input_bytes = n * batch * per_elem if batch else 0
        if batch:
            budget = budget_case(input_bytes, host_avail, device_free, device_total)
        else:
            budget = {"decision": "skip",
                      "reason": f"tier {tier} B cannot hold one n={n} input row",
                      "host_needed": "", "device_needed": "", "device_headroom": "",
                      "host_headroom": ""}
        row = {"stage": stage, "tier_bytes": tier, "direction": direction, "n": n,
               "batch": batch, "reps": reps, "input": input_mode,
               "input_bytes": input_bytes, "host_avail": host_avail,
               "device_free": device_free, "device_total": device_total,
               "host_needed": budget["host_needed"],
               "device_needed": budget["device_needed"],
               "decision": budget["decision"], "reason": budget["reason"],
               "returncode": "", "wall_ms": "", "max_rel": "",
               "rss_growth_mb": "", "correct": "", "failure": ""}
        rows.append(row)

    for tier in tiers:
        for n in c2c_ns:
            stage = "soak" if n == soak_n else "case"
            reps = soak_reps if stage == "soak" else QUICK_REPS
            add(tier, stage, "c2c", n, reps, 8, "random-seeded")
        for n in real.get("r2c_ns", []):
            add(tier, "case", "r2c", int(n), QUICK_REPS, 4, "random-seeded")
        for n in real.get("c2r_ns", []):
            add(tier, "case", "c2r", int(n), QUICK_REPS, 8, "half-spectrum")
    return rows


def read_rss_bytes(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"VmRSS:\s+(\d+) kB", text)
    return int(match.group(1)) * 1024 if match else None


def rss_growth_bytes(samples):
    if len(samples) < RSS_MIN_SAMPLES:
        return None
    baseline = samples[len(samples) // 4][1]
    tail = samples[-max(4, len(samples) // 10):]
    return max(0, max(rss for _, rss in tail) - baseline)


def rss_limit_bytes(peak_bytes):
    return max(96 * MIB, int(0.08 * peak_bytes))


def run_with_rss(command, env, timeout_s):
    started = time.perf_counter()
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    samples = []
    timed_out = False
    while proc.poll() is None:
        if time.perf_counter() - started > timeout_s:
            timed_out = True
            proc.kill()
            break
        rss = read_rss_bytes(proc.pid)
        if rss is not None:
            samples.append((time.perf_counter() - started, rss))
        time.sleep(RSS_SAMPLE_S)
    output = ""
    if proc.stdout is not None:
        output = proc.stdout.read()
        proc.stdout.close()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    wall_ms = (time.perf_counter() - started) * 1000.0
    return proc.returncode, output, samples, timed_out, wall_ms


def execute_case(row):
    case = {"category": "stress", "direction": row["direction"], "n": row["n"],
            "batch": row["batch"], "input": row["input"], "seed": 0}
    env_delta, command = command_for(case, row["reps"])
    env = dict(os.environ, **env_delta)
    timeout_s = case_timeout_s(row["reps"])
    returncode, output, samples, timed_out, wall_ms = run_with_rss(command, env,
                                                                   timeout_s)
    match = re.search(r"maxRel=([\d.eE+-]+)", output)
    error = float(match.group(1)) if match else float("nan")
    passed = (not timed_out and returncode == 0 and bool(re.search(r"^PASS$", output, re.M))
              and 0 <= error <= 1e-4)
    failure = ""
    if timed_out:
        failure = f"timeout after {timeout_s:.0f} seconds"
    elif not passed:
        failure = "command, PASS marker or maxRel check failed"
    growth = ""
    if row["stage"] == "soak" and wall_ms >= RSS_MIN_WALL_S * 1000.0:
        measured = rss_growth_bytes(samples)
        if measured is not None:
            limit = rss_limit_bytes(max(rss for _, rss in samples))
            growth = round(measured / MIB, 3)
            if measured > limit:
                passed = False
                failure = (f"rss grew {measured} B over baseline limit {limit} B "
                           f"during soak")
    row.update(returncode=returncode, wall_ms=round(wall_ms, 1),
               max_rel=("" if error != error else error), rss_growth_mb=growth,
               correct=passed, failure=failure)
    if not passed and not failure:
        row["failure"] = "command, PASS marker or maxRel check failed"
    return row


def run_lifecycle(row, cycles):
    case = {"category": "stress", "direction": row["direction"], "n": row["n"],
            "batch": row["batch"], "input": row["input"], "seed": 0}
    env_delta, command = command_for(case, 1)
    env = dict(os.environ, **env_delta)
    started = time.perf_counter()
    failed_cycles = []
    for cycle in range(1, cycles + 1):
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, timeout=120)
        except subprocess.TimeoutExpired:
            failed_cycles.append(cycle)
            continue
        output = result.stdout + result.stderr
        if result.returncode != 0 or not re.search(r"^PASS$", output, re.M):
            failed_cycles.append(cycle)
    wall_ms = (time.perf_counter() - started) * 1000.0
    passed = not failed_cycles
    row.update(reps=cycles, returncode=0 if passed else 1, wall_ms=round(wall_ms, 1),
               correct=passed,
               failure="" if passed else f"lifecycle cycles failed: {failed_cycles[:8]}")
    return row


def run_probe(timeout_s=300.0):
    row = {"stage": "probe", "tier_bytes": 0, "direction": "c2c", "n": 0, "batch": 0,
           "reps": 0, "input": "random-seeded", "input_bytes": 0, "host_avail": "",
           "device_free": "", "device_total": "", "host_needed": "",
           "device_needed": "", "decision": "run", "reason": "", "returncode": "",
           "wall_ms": "", "max_rel": "", "rss_growth_mb": "", "correct": "",
           "failure": ""}
    started = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, "-c", OOM_PROBE], cwd=ROOT,
                                capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        row.update(returncode=124, wall_ms=round((time.perf_counter() - started) * 1000, 1),
                   correct=False, failure=f"oom probe timeout after {timeout_s:.0f} s")
        return row
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    verdict = stdout_lines[-1] if stdout_lines else "NO_OUTPUT"
    passed = result.returncode == 0 and verdict == "RECOVERED"
    failure = ""
    if not passed:
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        failure = f"oom probe verdict {verdict!r} rc={result.returncode} stderr={stderr_tail}"
    row.update(returncode=result.returncode,
               wall_ms=round((time.perf_counter() - started) * 1000, 1),
               correct=passed, failure=failure)
    return row


def print_plan(rows, soak_reps, host_avail, device_free, device_total):
    print(f"plan rows={len(rows)} soak_reps={soak_reps} "
          f"host_avail={host_avail} device_free={device_free} "
          f"device_total={device_total}")
    for row in rows:
        print(f"  tier={row['tier_bytes']:<12} {row['stage']:<6} {row['direction']:<4} "
              f"N={row['n']:<5} B={row['batch']:<8} reps={row['reps']:<6} "
              f"input={row['input_bytes']:<11} {row['decision']:<5} {row['reason']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default="")
    parser.add_argument("--build", action="store_true", help="run scripts/build.sh all first")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--soak-reps", type=int, default=None,
                        help="override profile repeat_execution (debug only)")
    parser.add_argument("--lifecycle-cycles", type=int, default=None,
                        help="override lifecycle cycles (debug only)")
    args = parser.parse_args(argv)

    document = load_config(args.config)
    if "stress" not in document["profiles"]:
        parser.error("config has no stress profile")
    profile = dict(document["profiles"]["stress"])
    if args.soak_reps is not None:
        if args.soak_reps < 1:
            parser.error("--soak-reps must be positive")
        profile["repeat_execution"] = args.soak_reps
    cycles = args.lifecycle_cycles if args.lifecycle_cycles is not None else 20
    if cycles < 1:
        parser.error("--lifecycle-cycles must be positive")
    soak_reps = int(profile.get("repeat_execution", 10000))

    try:
        host_avail = read_host_available()
        hbm_before, device_total = read_hbm()
        device_free = device_total - hbm_before
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"cannot budget without memory numbers: {exc}", file=sys.stderr)
        return 2

    rows = plan(profile, host_avail, device_free, device_total)
    if args.dry_run:
        print_plan(rows, soak_reps, host_avail, device_free, device_total)
        return 0
    if args.build:
        subprocess.run([str(ROOT / "scripts" / "build.sh"), "all"], cwd=ROOT, check=True)
    if not (ROOT / "build" / "fft_check").is_file():
        parser.error("build/fft_check is missing; rerun with --build")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else ROOT / "results" / "runs" / f"{run_stamp}-stress"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    host_before = host_avail
    lifecycle_template = None
    executed = []
    total = len(rows) + 2
    index = 0
    for row in rows:
        index += 1
        print(f"[{index}/{total}] {row['stage']} tier={row['tier_bytes']} "
              f"{row['direction']} N={row['n']} B={row['batch']} reps={row['reps']} "
              f"-> {row['decision']}"
              + (f" ({row['reason']})" if row["reason"] else ""), flush=True)
        if row["decision"] != "run":
            continue
        if row["stage"] == "case" and lifecycle_template is None and row["direction"] == "c2c" \
                and row["tier_bytes"] == min(profile.get("input_buffer_bytes", [0])):
            lifecycle_template = row
        execute_case(row)
        executed.append(row)
        if not row["correct"]:
            print(row["failure"], file=sys.stderr)

    lifecycle_row = dict.fromkeys(FIELDS, "")
    lifecycle_row.update(stage="lifecycle", decision="run",
                         tier_bytes=min(profile.get("input_buffer_bytes", [0])),
                         direction="c2c", input="random-seeded")
    index += 1
    print(f"[{index}/{total}] lifecycle cycles={cycles}", flush=True)
    if lifecycle_template is None:
        lifecycle_row.update(correct=False, returncode=1,
                             failure="no budgeted c2c shape available for lifecycle")
        lifecycle_ok = False
    else:
        lifecycle_row.update(n=lifecycle_template["n"], batch=lifecycle_template["batch"],
                             input=lifecycle_template["input"],
                             tier_bytes=lifecycle_template["tier_bytes"])
        run_lifecycle(lifecycle_row, cycles)
        lifecycle_ok = lifecycle_row["correct"]
        executed.append(lifecycle_row)
        if not lifecycle_ok:
            print(lifecycle_row["failure"], file=sys.stderr)

    index += 1
    print(f"[{index}/{total}] oom recovery probe", flush=True)
    probe_row = run_probe()
    executed.append(probe_row)
    if not probe_row["correct"]:
        print(probe_row["failure"], file=sys.stderr)

    gate_failures = []
    if not lifecycle_ok and lifecycle_template is None:
        gate_failures.append(lifecycle_row["failure"])
    if not probe_row["correct"]:
        gate_failures.append(probe_row["failure"])
    try:
        time.sleep(3.0)
        hbm_after, device_total_after = read_hbm()
        host_after = read_host_available()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        hbm_after, device_total_after, host_after = -1, device_total, -1
        gate_failures.append(f"cannot re-read memory numbers after run: {exc}")
    hbm_allowance = max(2 * GIB, int(0.05 * device_total))
    hbm_growth = hbm_after - hbm_before if hbm_after >= 0 else -1
    if hbm_after >= 0 and hbm_growth > hbm_allowance:
        gate_failures.append(f"device HBM grew {hbm_growth} B over allowance "
                             f"{hbm_allowance} B after all processes exited")

    tiers = [int(tier) for tier in profile.get("input_buffer_bytes", [])]
    for tier in tiers:
        if not any(row["tier_bytes"] == tier for row in executed):
            gate_failures.append(f"tier {tier} produced no executable case")

    failures = [row for row in executed if not row["correct"]]
    skipped = [row for row in rows if row["decision"] != "run"]
    passed = len(executed) - len(failures)
    status = "pass" if not failures and not gate_failures else "fail"

    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(lifecycle_row)
        writer.writerow(probe_row)

    summary = {
        "schema_version": 1,
        "profile": "stress",
        "tiers": tiers,
        "repeat_execution": soak_reps,
        "lifecycle_cycles": cycles,
        "planned_rows": len(rows),
        "executed": len(executed),
        "skipped": len(skipped),
        "passed": passed,
        "failed": len(failures),
        "gate_failures": gate_failures,
        "hbm_before_mb": hbm_before // MIB,
        "hbm_after_mb": hbm_after // MIB,
        "hbm_growth_mb": hbm_growth // MIB,
        "hbm_allowance_mb": hbm_allowance // MIB,
        "host_avail_before_mb": host_before // MIB,
        "host_avail_after_mb": host_after // MIB,
        "oom_probe": "recovered" if probe_row["correct"] else probe_row["failure"],
        "correctness_threshold": 1e-4,
        "status": status,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8")
    print(f"{passed}/{len(executed)} PASS, {len(skipped)} budget-skipped, "
          f"gates={len(gate_failures)} -> {out}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
