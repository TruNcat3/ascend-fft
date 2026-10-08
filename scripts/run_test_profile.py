#!/usr/bin/env python3
"""Run a predeclared correctness profile from config/test_matrix.json.

This runner intentionally does not benchmark external libraries. Performance publication remains
owned by matrix_test.py, bench_r2c_c2r.py and e2e_test.py so correctness and timing policies cannot
be confused.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "test_matrix.json"
DEFAULT_COMPONENTS = ("c2c", "real", "numeric", "applications")


def load_config(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("profiles"), dict):
        raise ValueError("unsupported test-matrix schema")
    return document


def c2c_points(profile):
    spec = profile.get("c2c", {})
    ns = spec.get("ns", [])
    points = {(int(n), int(batch)) for n in ns for batch in spec.get("batches", [])}
    for total in spec.get("fixed_total_points", []):
        for n in ns:
            if total % n == 0:
                points.add((int(n), int(total // n)))
    return sorted(points)


def _case(category, direction, n, batch, input_mode, seed=0):
    return {
        "category": category,
        "direction": direction,
        "n": int(n),
        "batch": int(batch),
        "input": input_mode,
        "seed": int(seed),
    }


def expand_cases(profile, components=DEFAULT_COMPONENTS):
    components = set(components)
    cases = []
    if "c2c" in components:
        cases.extend(_case("c2c", "c2c", n, batch, "sin")
                     for n, batch in c2c_points(profile))

    if "real" in components:
        real = profile.get("real", {})
        for n in real.get("r2c_ns", []):
            cases.extend(_case("real", "r2c", n, batch, "sin")
                         for batch in real.get("batches", []))
        for n in real.get("c2r_ns", []):
            cases.extend(_case("real", "c2r", n, batch, "half-spectrum")
                         for batch in real.get("batches", []))

    if "numeric" in components:
        seeds = profile.get("random_seeds", [0])
        for n, batch in profile.get("numeric_cases", []):
            for pattern in profile.get("numeric_patterns", []):
                pattern_seeds = seeds if pattern == "random-seeded" else [0]
                cases.extend(_case("numeric", "c2c", n, batch, pattern, seed)
                             for seed in pattern_seeds)

    if "applications" in components:
        modes = {"ofdm": "ofdm", "radar-range": "radar",
                 "dl-frequency-layer": "dl", "stft": "sin"}
        for app in profile.get("applications", []):
            transform = app.get("transform")
            if transform not in ("c2c", "r2c") or app.get("id") not in modes:
                continue
            for n in app.get("ns", []):
                cases.extend(_case("application", transform, n, batch, modes[app["id"]])
                             for batch in app.get("batches", []))

    unique = {}
    for case in cases:
        key = tuple(case[field] for field in ("category", "direction", "n", "batch", "input", "seed"))
        unique[key] = case
    return [unique[key] for key in sorted(unique)]


def command_for(case, reps):
    env = {"AB_DIR": case["direction"]}
    if case["input"] != "half-spectrum":
        env["AB_INPUT"] = case["input"]
    if case["seed"]:
        env["AB_SEED"] = str(case["seed"])
    command = [str(ROOT / "build" / "fft_check"), str(case["n"]),
               str(case["batch"]), str(reps)]
    return env, command


def shell_line(case, reps):
    env, command = command_for(case, reps)
    prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items()))
    return f"{prefix} {' '.join(command)}"


def run_case(case, reps, trial):
    env_delta, command = command_for(case, reps)
    env = dict(os.environ, **env_delta)
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                timeout=3600)
    except subprocess.TimeoutExpired as exc:
        wall_ms = (time.perf_counter() - started) * 1000.0
        return dict(case, trial=trial, returncode=124, max_rel=math.nan,
                    wall_ms=wall_ms, correct=False, failure="timeout after 3600 seconds",
                    output_tail=str(exc))
    wall_ms = (time.perf_counter() - started) * 1000.0
    output = result.stdout + result.stderr
    match = re.search(r"maxRel=([\d.eE+-]+)", output)
    error = float(match.group(1)) if match else math.nan
    passed = (result.returncode == 0 and bool(re.search(r"^PASS$", output, re.M))
              and math.isfinite(error) and 0 <= error <= 1e-4)
    return dict(case, trial=trial, returncode=result.returncode, max_rel=error,
                wall_ms=wall_ms, correct=passed,
                failure="" if passed else "command, PASS marker or maxRel check failed",
                output_tail="\n".join(output.splitlines()[-12:]) if not passed else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="smoke")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS),
                        help="comma-separated: c2c,real,numeric,applications")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--build", action="store_true", help="run scripts/build.sh all first")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    document = load_config(args.config)
    profiles = document["profiles"]
    if args.list:
        for name, profile in profiles.items():
            print(f"{name:20} runnable={str(bool(profile.get('runnable'))).lower():5} "
                  f"{profile.get('description', '')}")
        return 0
    if args.profile not in profiles:
        parser.error(f"unknown profile {args.profile!r}; use --list")
    profile = profiles[args.profile]
    if not profile.get("runnable"):
        parser.error(f"profile {args.profile!r} is an acceptance target, not runnable")
    components = tuple(item.strip() for item in args.components.split(",") if item.strip())
    unknown = set(components) - set(DEFAULT_COMPONENTS)
    if unknown:
        parser.error(f"unknown components: {','.join(sorted(unknown))}")
    if args.reps < 1 or (args.trials is not None and args.trials < 1):
        parser.error("reps and trials must be positive")
    trials = args.trials or (3 if args.profile in ("regression", "publication") else 1)
    cases = expand_cases(profile, components)

    print(f"profile={args.profile} cases={len(cases)} trials={trials} reps={args.reps}")
    if args.dry_run:
        for case in cases:
            print(shell_line(case, args.reps))
        return 0
    if args.build:
        subprocess.run([str(ROOT / "scripts" / "build.sh"), "all"], cwd=ROOT, check=True)
    if not (ROOT / "build" / "fft_check").is_file():
        parser.error("build/fft_check is missing; rerun with --build")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else ROOT / "results" / "runs" / f"{run_stamp}-{args.profile}"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    total = len(cases) * trials
    index = 0
    for trial in range(1, trials + 1):
        for case in cases:
            index += 1
            print(f"[{index}/{total}] {case['category']} {case['direction']} "
                  f"N={case['n']} B={case['batch']} input={case['input']} trial={trial}",
                  flush=True)
            row = run_case(case, args.reps, trial)
            rows.append(row)
            if not row["correct"]:
                print(row["output_tail"], file=sys.stderr)

    fields = ("category", "direction", "n", "batch", "input", "seed", "trial",
              "returncode", "max_rel", "wall_ms", "correct", "failure", "output_tail")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if not row["correct"]]
    summary = {
        "schema_version": 1,
        "profile": args.profile,
        "components": components,
        "cases": len(cases),
        "trials": trials,
        "executions": len(rows),
        "passed": len(rows) - len(failures),
        "failed": len(failures),
        "correctness_threshold": 1e-4,
        "status": "pass" if not failures else "fail",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['passed']}/{summary['executions']} PASS -> {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
