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
         "r2c_c2r.raw_trials.csv", "trials.csv", "protocol.json", "figures.json")
# 原始实验产物（进快照即为不可变原始证据）；matrix.csv/comparison.md 属派生。
RAW_INPUTS = ("matrix.md", "summary.json", "sixway.md", "e2e.json",
              "e2e_app.json", "r2c_c2r.json", "r2c_c2r.raw_trials.json",
              "r2c_c2r.raw_trials.csv", "trials.csv", "protocol.json", "figures.json")
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


def detect_npu_smi():
    """硬件无关环境（CI）没有 npu-smi；探测失败时留空而不是让采集崩溃。"""
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True,
                                text=True, timeout=30)
        return result.stdout[:600] if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_command_version(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = (result.stdout + result.stderr).strip()
        return output.splitlines()[0][:300] if result.returncode == 0 and output else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def detect_ccec_version():
    for option in ("--version", "-v"):
        version = detect_command_version(["ccec", option])
        if version != "unknown":
            return version
    return "unknown"


def build_hashes():
    hashes = {}
    build = ROOT / "build"
    if not build.is_dir():
        return hashes
    for path in sorted(build.iterdir()):
        if path.is_file() and (os.access(path, os.X_OK) or path.suffix == ".o"):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def provenance_problems(manifest):
    """Return defects that make a result snapshot unsuitable for publication."""
    schema = int(manifest.get("schema_version", 1))
    if schema in (1, 2):
        if (manifest.get("publication_status") == "legacy-unverified"
                and manifest.get("provenance_limitations")):
            return []
        return [f"schema v{schema} snapshot must declare publication_status=legacy-unverified and provenance_limitations"]
    if schema != 3:
        return [f"unsupported manifest schema_version {schema}"]
    problems = []
    unknown = {None, "", "?", "unknown", "UNKNOWN"}
    if manifest.get("dirty") is not False:
        problems.append("source worktree was dirty")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("commit", ""))):
        problems.append("commit is missing or is not a full Git SHA")
    for field in ("soc", "hardware_id", "cann_version"):
        if manifest.get(field) in unknown:
            problems.append(f"{field} is unknown")
    if manifest.get("versions", {}).get("ccec") in unknown:
        problems.append("ccec version is unknown")
    digest = re.compile(r"[0-9a-f]{64}")
    selected = manifest.get("selected_profile", {})
    if not selected.get("path") or not digest.fullmatch(str(selected.get("sha256", ""))):
        problems.append("selected hardware profile path/hash is missing or invalid")
    else:
        profile_path = Path(selected["path"])
        if profile_path.is_absolute() or ".." in profile_path.parts:
            problems.append("selected hardware profile must be archived under the repository")
        else:
            profile_path = ROOT / profile_path
            if not profile_path.is_file():
                problems.append("selected hardware profile does not exist")
            elif hashlib.sha256(profile_path.read_bytes()).hexdigest() != selected["sha256"]:
                problems.append("selected hardware profile hash does not match its contents")
    target = manifest.get("compiler_target")
    if target in unknown:
        problems.append("compiler target SoC is unknown")
    elif re.sub(r"[^a-z0-9]", "", str(target).lower()) != re.sub(
            r"[^a-z0-9]", "", str(selected.get("soc", "")).lower()):
        problems.append("compiler target SoC does not match selected profile")
    build = manifest.get("build_artifact_sha256", {})
    if not build:
        problems.append("built executable/object hashes are missing")
    else:
        for name, value in build.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "build":
                problems.append("built artifact path must be relative and under build/")
            if not digest.fullmatch(str(value)):
                problems.append("built executable/object hash is not SHA-256")
    if manifest.get("profile_mismatch_override"):
        problems.append("hardware profile mismatch override was enabled")
    if manifest.get("environment_drift"):
        problems.append("run environment changed between recorded experiments")
    return problems


def selected_profile():
    path = Path(os.environ.get(
        "AB_PROFILE", ROOT / "config" / "ascend910_93_profile.json")).resolve()
    if not path.is_file():
        return {"path": str(path), "sha256": "unknown"}
    try:
        label = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        label = str(path)
    data = path.read_bytes()
    try:
        soc = json.loads(data).get("soc", "unknown")
    except (json.JSONDecodeError, AttributeError):
        soc = "unknown"
    return {"path": label, "sha256": hashlib.sha256(data).hexdigest(), "soc": soc}


def record_run(directory, experiment, command, exit_code):
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 3,
        "created_utc": now,
        "started_utc": now,
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "soc": os.environ.get("AB_SOC", "unknown"),
        "compiler_target": os.environ.get("AB_SOC", "unknown"),
        "cann_path": os.environ.get(
            "AB_CANN", "/usr/local/Ascend/ascend-toolkit/latest"),
        "cann_version": detect_cann_version(),
        "versions": {
            "python": sys.version.split()[0],
            "cann": detect_cann_version(),
            "ccec": detect_ccec_version(),
        },
        "hardware_id": os.environ.get("AB_HARDWARE_ID", "unknown"),
        "npu_smi": detect_npu_smi(),
        "selected_profile": selected_profile(),
        "profile_mismatch_override": os.environ.get("AB_ALLOW_PROFILE_MISMATCH") == "1",
        "build_artifact_sha256": build_hashes(),
        "experiments": [],
    }
    entry = {"name": experiment, "command": command,
             "started_utc": now,
             "status": "running" if exit_code is None else "complete" if exit_code == 0 else "failed"}
    if exit_code is not None:
        entry["exit_code"] = exit_code
        entry["ended_utc"] = now
        manifest["build_artifact_sha256"] = build_hashes()
        manifest.setdefault("versions", {})["ccec"] = detect_ccec_version()
        current_profile = selected_profile()
        if manifest.get("selected_profile") != current_profile:
            manifest.setdefault("environment_drift", []).append("selected hardware profile changed during run")
        manifest["selected_profile"] = current_profile
        manifest["dirty"] = manifest.get("dirty", False) or bool(
            git("status", "--porcelain", "--untracked-files=all"))
        manifest["profile_mismatch_override"] = (
            manifest.get("profile_mismatch_override", False)
            or os.environ.get("AB_ALLOW_PROFILE_MISMATCH") == "1")
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
        if name == "protocol.json" and not value:
            raise ValueError("protocol.json must describe at least one measurement protocol")
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
        for key, item in row.items():
            if key.endswith(("_us", "_ms")) and item is not None:
                if not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
                    raise ValueError(f"{name} has invalid timing {key}={item!r}")


def validate_trials_csv(data):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    if not rows:
        raise ValueError("trials.csv contains no raw trials")
    required = {"runner", "trial", "n", "batch", "mean_us", "min_us", "max_rel", "correct"}
    if not required.issubset(rows[0]):
        raise ValueError("trials.csv is missing required columns")
    for row in rows:
        if row["correct"].lower() not in ("true", "1"):
            raise ValueError("trials.csv contains a failed correctness trial")
        for key in ("mean_us", "min_us"):
            value = float(row[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"trials.csv has invalid {key}")
        error = float(row["max_rel"])
        if not math.isfinite(error) or error < 0 or error > 1e-4:
            raise ValueError("trials.csv has invalid max_rel")
    return rows


def validate_trial_coverage(data, summary, matrix_rows):
    rows = validate_trials_csv(data)
    rounds = summary.get("rounds")
    native_enabled = summary.get("native_enabled")
    if not isinstance(rounds, int) or rounds <= 0 or not isinstance(native_enabled, bool):
        raise ValueError("summary.json must record positive rounds and boolean native_enabled")
    runners = {"self", "native"} if native_enabled else {"self"}
    expected = {(runner, trial, row["n"], row["batch"])
                for runner in runners for trial in range(1, rounds + 1)
                for row in matrix_rows}
    observed = []
    try:
        for row in rows:
            observed.append((row["runner"], int(row["trial"]),
                             int(row["n"]), int(row["batch"])))
    except (KeyError, ValueError) as error:
        raise ValueError(f"trials.csv has invalid identity columns: {error}") from error
    if len(observed) != len(set(observed)):
        raise ValueError("trials.csv contains duplicate runner/trial/shape rows")
    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing or extra:
        raise ValueError(f"trials.csv coverage mismatch: missing={len(missing)}, extra={len(extra)}")


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
    provenance = provenance_problems(manifest)
    if provenance:
        raise ValueError("publication provenance is incomplete: " + "; ".join(provenance))
    experiments = manifest.get("experiments", [])
    if not experiments or any(item.get("status") != "complete" for item in experiments):
        raise ValueError("only completed, successful runs may be published")
    if manifest.get("schema_version") == 3 and any(
            item.get("exit_code") != 0 for item in experiments):
        raise ValueError("schema v3 experiments require an explicit zero exit_code")
    if manifest.get("schema_version") == 3:
        required = ("matrix.md", "summary.json", "trials.csv", "protocol.json")
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            raise ValueError("schema v3 publication is missing required evidence: "
                             + ", ".join(missing))
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
                if name == "trials.csv":
                    validate_trials_csv(artifacts[name])
    if "matrix.md" not in artifacts:
        raise ValueError("publication requires matrix.md; combine experiments with AB_RUN_DIR")
    rows = parse_matrix(artifacts["matrix.md"].decode("utf-8"))
    summary = json.loads(artifacts.get("summary.json", b"{}"))
    if summary and summary["total_points"] != len(rows):
        raise ValueError("matrix.md and summary.json coverage disagree")
    if manifest.get("schema_version") == 3:
        validate_trial_coverage(artifacts["trials.csv"], summary, rows)
    if "matrix.csv" not in artifacts:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=["n", "batch", "ours_us", "native_us", "speedup"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        artifacts["matrix.csv"] = output.getvalue().encode()
    geometric = math.exp(sum(math.log(row["speedup"]) for row in rows) / len(rows))
    legacy = manifest.get("publication_status") == "legacy-unverified"
    warning = ("\n> **Historical evidence only:** provenance or correctness gates in this "
               "snapshot are incomplete; revalidation is required.\n\n" if legacy else "\n")
    report = ("# Published Benchmark Summary\n" + warning +
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
            if manifest.get("schema_version") == 3:
                figure_hashes = {}
                for figure in manifest["figures"]:
                    source = ROOT / "docs" / "figures" / figure
                    if not source.is_file():
                        raise ValueError(f"declared figure output is missing: {source}")
                    name = f"figures/{figure}"
                    artifacts[name] = source.read_bytes()
                    figure_hashes[name] = hashlib.sha256(artifacts[name]).hexdigest()
                manifest["figure_output_sha256"] = figure_hashes
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
    if manifest.get("schema_version") == 3:
        selected = manifest["selected_profile"]
        profile = ROOT / selected["path"]
        if not profile.is_file():
            problems.append(f"selected hardware profile missing: {profile}")
        elif hashlib.sha256(profile.read_bytes()).hexdigest() != selected["sha256"]:
            problems.append(f"selected hardware profile hash drift: {profile}")
        for name, digest in manifest.get("figure_output_sha256", {}).items():
            figure = destination / name
            if not figure.is_file():
                problems.append(f"published figure output missing: {figure}")
            elif hashlib.sha256(figure.read_bytes()).hexdigest() != digest:
                problems.append(f"published figure output hash drift: {figure}")
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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--snapshot", default="ascend910_9382-cann9.0.0")
    parser.add_argument("--published-root", type=Path, default=ROOT / "results/published")
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "docs/generated")
    parser.add_argument("--check", action="store_true", help="verify artifacts without writing")
    parser.add_argument("--replace-existing", action="store_true",
                        help="explicit migration only; allow rewriting an existing snapshot")
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
        if not args.check and destination.exists() and not args.replace_existing:
            raise ValueError("published snapshots are immutable; choose a new --snapshot "
                             "or use --replace-existing for an explicit migration")
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
    if args.check:
        status = json.loads(artifacts["manifest.json"]).get("publication_status")
        print("Artifact integrity verified; provenance remains legacy-unverified"
              if status == "legacy-unverified" else "Published artifacts and provenance verified")
    else:
        print(f"Published snapshot: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
