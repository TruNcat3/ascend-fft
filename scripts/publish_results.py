#!/usr/bin/env python3
"""Record experiment provenance and explicitly publish validated result snapshots."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FILES = ("matrix.md", "matrix.csv", "summary.json", "sixway.md", "e2e.json",
         "e2e_app.json", "r2c_c2r.json")


def canonical(value):
    def normalize(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: normalize(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [normalize(entry) for entry in item]
        return item
    return (json.dumps(normalize(value), indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def git(*arguments):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def record_run(directory, experiment, command, exit_code):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "soc": os.environ.get("AB_SOC", "unknown"),
        "cann_path": os.environ.get("AB_CANN", "unknown"),
        "cann_version": os.environ.get("AB_CANN_VERSION", "unknown"),
        "hardware_profile_sha256": {
            str(item.relative_to(ROOT)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted((ROOT / "config").glob("*profile*.json"))},
        "experiments": [],
    }
    entry = {"name": experiment, "command": command,
             "status": "running" if exit_code is None else "complete" if exit_code == 0 else "failed"}
    if exit_code is not None:
        entry["exit_code"] = exit_code
    if manifest["experiments"] and manifest["experiments"][-1]["name"] == experiment and manifest["experiments"][-1]["status"] == "running":
        manifest["experiments"][-1] = entry
    else:
        manifest["experiments"].append(entry)
    path.write_bytes(canonical(manifest))


def parse_matrix(text):
    rows = []
    for line in text.splitlines():
        fields = [part.strip().replace("**", "") for part in line.split("|")]
        if len(fields) != 13 or not fields[1].isdigit() or not fields[2].isdigit():
            continue
        if fields[11] != "PASS":
            raise ValueError("matrix contains a failed correctness point")
        def number(index):
            value = float(fields[index].replace(",", "").replace("×", ""))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("matrix latency must be positive and finite")
            return value
        rows.append({"n": int(fields[1]), "batch": int(fields[2]),
                     "ours_us": number(3), "native_us": number(5),
                     "speedup": number(5) / number(3)})
    if not rows:
        raise ValueError("matrix.md contains no supported correctness-checked rows")
    if len({(row["n"], row["batch"]) for row in rows}) != len(rows):
        raise ValueError("matrix.md contains duplicate shapes")
    return rows


def validate_json(name, value):
    if name == "summary.json":
        if value.get("correct_points") != value.get("total_points") or not value.get("total_points"):
            raise ValueError("summary.json contains incomplete correctness coverage")
        return
    rows = value if isinstance(value, list) else value.get("rows", [])
    if not rows:
        raise ValueError(f"{name} contains no result rows")
    for row in rows:
        flags = [row[key] for key in ("ok", "ours_ok", "correct", "native_ok", "nat_ok", "torch_ok") if key in row and row[key] is not None]
        bare = row.get("bare_dev")
        if isinstance(bare, (int, float)) and math.isfinite(bare) and bare > 0:
            flags.append(row.get("bare_ok", False))
        if not flags or not all(flag is True or flag == 1 for flag in flags):
            raise ValueError(f"{name} has missing or failed correctness flags")


def render_matrix(rows):
    lines = ["# Published C2C Matrix", "", "> Device-only latency in microseconds. Speedup > 1 favors Ascend-FFT.", "",
             "| N | Batch | Ascend-FFT (us) | CANN native (us) | Speedup |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['n']} | {row['batch']} | {row['ours_us']:.3f} | {row['native_us']:.3f} | {row['speedup']:.3f}x |")
    return ("\n".join(lines) + "\n").encode()


def build_artifacts(run):
    source = run / "manifest.json"
    if not source.is_file():
        raise ValueError("run requires manifest.json; use repro.sh or record provenance explicitly")
    manifest = json.loads(source.read_text())
    experiments = manifest.get("experiments", [])
    if not experiments or any(item.get("status") != "complete" for item in experiments):
        raise ValueError("only completed, successful runs may be published")
    artifacts = {}
    for name in FILES:
        path = run / name
        if path.exists():
            if path.suffix == ".json":
                value = json.loads(path.read_text())
                validate_json(name, value)
                artifacts[name] = canonical(value)
            else:
                artifacts[name] = path.read_bytes()
    if "matrix.md" not in artifacts:
        raise ValueError("publication requires matrix.md; combine experiments with AB_RUN_DIR")
    rows = parse_matrix(artifacts["matrix.md"].decode("utf-8"))
    summary = json.loads(artifacts.get("summary.json", b"{}"))
    if summary and summary["total_points"] != len(rows):
        raise ValueError("matrix.md and summary.json coverage disagree")
    if "matrix.csv" not in artifacts:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=["n", "batch", "ours_us", "native_us", "speedup"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        artifacts["matrix.csv"] = output.getvalue().encode()
    geometric = math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
    manifest["published_artifacts"] = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(artifacts.items())}
    artifacts["manifest.json"] = canonical(manifest)
    report = ("# Published Benchmark Summary\n\n"
              f"Source commit: `{manifest.get('commit', 'unknown')}`. "
              f"SoC: `{manifest.get('soc', 'unknown')}`.\n\n"
              f"C2C device-only: **{len(rows)} checked points**, "
              f"geometric mean speedup **{geometric:.3f}x** against CANN native.\n\n"
              "This mean covers only the published C2C matrix, not end-to-end or real transforms.\n\n"
              "[Full matrix](matrix.md). Protocol and provenance are in the published manifest.\n")
    comparison = report.encode()
    if "sixway.md" in artifacts:
        comparison += ("\n## Matched Multi-Baseline Detail\n\n").encode()
        comparison += artifacts["sixway.md"]
    generated = {"matrix.md": render_matrix(rows), "comparison.md": comparison}
    return artifacts, generated


def synchronize(directory, files, check):
    stale = []
    for name, data in sorted(files.items()):
        path = directory / name
        if check:
            if not path.exists() or path.read_bytes() != data:
                stale.append(str(path))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--snapshot", default="ascend910_9382-cann9.0.0")
    parser.add_argument("--published-root", type=Path, default=ROOT / "results/published")
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "docs/generated")
    parser.add_argument("--check", action="store_true", help="verify artifacts without writing")
    parser.add_argument("--record-run", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--command")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    if args.record_run:
        if not args.experiment or not args.command or args.check:
            parser.error("--record-run requires --experiment and --command and cannot use --check")
        record_run(args.record_run, args.experiment, args.command, args.exit_code)
        return 0
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.snapshot):
        parser.error("--snapshot must be a single safe directory name")
    destination = args.published_root / args.snapshot
    run = args.run or (destination if args.check else None)
    if run is None:
        parser.error("publication requires --run")
    try:
        artifacts, generated = build_artifacts(run)
        if not args.check and destination.exists():
            leftovers = [name for name in FILES if (destination / name).exists() and name not in artifacts]
            if leftovers:
                raise ValueError("snapshot contains additional artifacts; use a new --snapshot or supply them in the run: " + ", ".join(leftovers))
        stale = synchronize(destination, artifacts, args.check)
        stale += synchronize(args.generated_dir, generated, args.check)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"publication error: {error}", file=sys.stderr)
        return 1
    if stale:
        print("out-of-date artifacts:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    print("Published artifacts verified" if args.check else f"Published snapshot: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
