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
         "e2e_app.json", "r2c_c2r.json", "r2c_c2r.raw_trials.json",
         "r2c_c2r.raw_trials.csv", "protocol.json", "figures.json")
# 原始实验产物（进快照即为不可变原始证据）；matrix.csv/comparison.md 属派生。
RAW_INPUTS = ("matrix.md", "summary.json", "sixway.md", "e2e.json",
              "e2e_app.json", "r2c_c2r.json", "r2c_c2r.raw_trials.json",
              "r2c_c2r.raw_trials.csv", "protocol.json", "figures.json")
DERIVED_GENERATED = ("matrix.csv", "matrix.md", "comparison.md")


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


def record_run(directory, experiment, command, exit_code):
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 2,
        "created_utc": now,
        "started_utc": now,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "soc": os.environ.get("AB_SOC", "unknown"),
        "cann_path": os.environ.get(
            "AB_CANN", "/usr/local/Ascend/ascend-toolkit/latest"),
        "cann_version": detect_cann_version(),
        "versions": {
            "python": sys.version.split()[0],
            "cann": detect_cann_version(),
        },
        "hardware_id": os.environ.get("AB_SOC", "unknown"),
        "npu_smi": subprocess.run(["npu-smi", "info"], capture_output=True,
                                  text=True, timeout=30).stdout[:600],
        "hardware_profile_sha256": {
            str(item.relative_to(ROOT)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted((ROOT / "config").glob("*profile*.json"))},
        "experiments": [],
    }
    entry = {"name": experiment, "command": command,
             "started_utc": now,
             "status": "running" if exit_code is None else "complete" if exit_code == 0 else "failed"}
    if exit_code is not None:
        entry["exit_code"] = exit_code
        entry["ended_utc"] = now
    if manifest["experiments"] and manifest["experiments"][-1]["name"] == experiment and manifest["experiments"][-1]["status"] == "running":
        entry["started_utc"] = manifest["experiments"][-1].get("started_utc", now)
        manifest["experiments"][-1] = entry
    else:
        manifest["experiments"].append(entry)
    manifest["updated_utc"] = now
    manifest["ended_utc"] = now
    manifest.setdefault("started_utc", now)
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
    if name in ("protocol.json", "figures.json"):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        return
    if name.endswith(".raw_trials.json"):
        trials = value.get("raw_trials", [])
        if not trials:
            raise ValueError(f"{name} contains no raw trial rows")
        valid = {"ok", "error", "timeout", "unsupported"}
        for rec in trials:
            if rec.get("status") not in valid:
                raise ValueError(f"{name} has unknown trial status {rec.get('status')!r}")
            if rec.get("status") == "ok" and not (rec.get("us") and rec["us"] > 0):
                raise ValueError(f"{name} admits a non-positive ok trial timing")
        return
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
    # 契约字段（P2-D）：原始/派生哈希、协议、图形→输入映射全部由 run 文件确定性导出，
    # 校验重算与发布写入得到同一字节流；npu_smi/版本等易变 provenance 由 record_run 固化。
    # 旧 schema(v1) 快照保持字节不变——不就地变更 legacy。
    if manifest.get("schema_version", 1) >= 2 or "raw_artifact_hashes" in manifest:
        raw_hashes = {name: hashlib.sha256(artifacts[name]).hexdigest()
                      for name in RAW_INPUTS if name in artifacts}
        derived_hashes = {name: hashlib.sha256(data).hexdigest()
                          for name, data in artifacts.items()
                          if name not in RAW_INPUTS and name != "manifest.json"}
        derived_hashes.update({f"generated/{name}": hashlib.sha256(data).hexdigest()
                               for name, data in generated.items()})
        if "protocol.json" in artifacts:
            manifest["protocol"] = json.loads(artifacts["protocol.json"])
        if "figures.json" in artifacts:
            manifest["figures"] = json.loads(artifacts["figures.json"])
        manifest["raw_artifact_hashes"] = raw_hashes
        manifest["derived_artifact_hashes"] = derived_hashes
    manifest["published_artifacts"] = {name: hashlib.sha256(data).hexdigest()
                                       for name, data in sorted(artifacts.items())}
    artifacts["manifest.json"] = canonical(manifest)
    return artifacts, generated


def verify_immutable(destination, artifacts):
    """P2-D：--check 附加的不可变性判据（原始哈希、图形输入、commit 与工作树漂移）。"""
    problems = []
    manifest = json.loads(artifacts["manifest.json"])
    published = manifest.get("published_artifacts", {})
    for name, digest in sorted(published.items()):
        path = destination / name
        if not path.is_file():
            problems.append(f"missing published artifact: {path}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"artifact hash drift: {path}")
    for figure, input_name in sorted(manifest.get("figures", {}).items()):
        if input_name not in published:
            problems.append(f"figure input drift: {figure} -> {input_name} "
                            f"is not a hash-anchored published artifact")
        elif not (destination / input_name).is_file():
            problems.append(f"figure input missing: {input_name} (for {figure})")
    commit = manifest.get("commit", "")
    if commit and commit != "unknown":
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                                cwd=ROOT)
        if result.returncode != 0:
            problems.append(f"manifest commit drift: {commit} is not an "
                            f"ancestor of HEAD")
    try:
        rel = destination.resolve().relative_to(ROOT.resolve())
    except ValueError:
        rel = None
    if rel is not None:
        status = subprocess.run(["git", "status", "--porcelain", "--", str(rel)],
                                cwd=ROOT, capture_output=True, text=True)
        if status.stdout.strip():
            problems.append(f"snapshot working-tree drift under {rel}: "
                            + "; ".join(status.stdout.strip().splitlines()[:5]))
    return problems


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
        # docs/generated 只归"声明了图形映射"的快照所有，避免多快照互相覆盖；
        # legacy(v1) 快照不再触碰生成物，v2 起由 figures.json 声明所有权。
        owns_figures = bool(json.loads(artifacts["manifest.json"]).get("figures"))
        if owns_figures or not args.check:
            stale += synchronize(args.generated_dir, generated, args.check)
        if args.check and not stale:
            stale += verify_immutable(destination, artifacts)
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
