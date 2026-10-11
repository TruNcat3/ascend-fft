#!/usr/bin/env python3
"""Validate and list immutable-snapshot optimization targets, without hardware."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = "ascend910_9382-revalidation-20261010"
BASELINES = {
    "c2c": {"runner": "cann-native", "source": "matrix.csv",
            "baseline_field": "native_us", "candidate_field": "ours_us"},
    "r2c": {"runner": "torch-npu-rfft", "source": "r2c_c2r.json",
            "baseline_field": "torch_rfft", "candidate_field": "r2c"},
    "c2r": {"runner": "torch-npu-irfft", "source": "r2c_c2r.json",
            "baseline_field": "torch_irfft", "candidate_field": "c2r"},
}
PRIORITIES = {"c2c": "P0", "r2c": "P1", "c2r": "P2"}
SOURCE_FILES = {"matrix.csv", "r2c_c2r.json", "protocol.json", "summary.json"}


def read_source_targets(snapshot):
    """Derive targets from matched NPU timings, not rounded report labels."""
    targets = {}

    def add(operator, n, batch, ours, baseline, correct):
        if ours is None or baseline is None:
            return
        ours, baseline = float(ours), float(baseline)
        if not all(math.isfinite(v) and v > 0 for v in (ours, baseline)):
            raise ValueError(f"invalid source timing: {operator} N={n} batch={batch}")
        if correct is not True:
            raise ValueError(f"source correctness failure: {operator} N={n} batch={batch}")
        if baseline / ours < 1.0:
            key = (operator, int(n), int(batch))
            if key in targets:
                raise ValueError(f"duplicate source point: {key}")
            targets[key] = baseline / ours

    with (snapshot / "matrix.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            add("c2c", row["n"], row["batch"], row["ours_us"],
                row["native_us"], row["correct"] == "True")
    real = json.loads((snapshot / "r2c_c2r.json").read_text(encoding="utf-8"))
    for row in real["rows"]:
        for operator in ("r2c", "c2r"):
            baseline = BASELINES[operator]["baseline_field"]
            add(operator, row["n"], row["b"], row[operator],
                row[baseline], row[operator + "_ok"])
            ratio = row[operator + "_vs_torch"]
            if row[operator] is not None and ratio is not None:
                expected = float(row[baseline]) / float(row[operator])
                if not math.isclose(ratio, expected, rel_tol=1e-12):
                    raise ValueError(f"source ratio disagrees with timings: {operator}")
    return targets


def validate(document, root=ROOT):
    if document.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    if document.get("source_snapshot") != SNAPSHOT:
        raise ValueError("source snapshot drift: expected " + SNAPSHOT)
    if document.get("baselines") != BASELINES:
        raise ValueError("baseline drift: only the matched NPU baselines are allowed")
    snapshot = Path(root) / "results" / "published" / SNAPSHOT
    hashes = document.get("source_sha256", {})
    if set(hashes) != SOURCE_FILES:
        raise ValueError("source hash file set drift")
    for name, expected in hashes.items():
        actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("source artifact drift: " + name)
    objective = document.get("objective", {})
    if (objective.get("metric") != "matched_baseline_latency / candidate_latency"
            or objective.get("target_rule") != "speedup < 1.0"
            or objective.get("pointwise_minimum_speedup") != 1.0
            or objective.get("report_parity_band") != [0.95, 1.05]
            or objective.get("parity_is_not_objective_completion") is not True):
        raise ValueError("pointwise unity objective/report band drift")
    gates = document.get("gates", {})
    for name in ("correctness", "same_protocol", "candidate_closure",
                 "full_matrix_non_regression", "uncertainty"):
        if not isinstance(gates.get(name), str) or not gates[name].strip():
            raise ValueError("missing acceptance gate: " + name)
    phases = document.get("phases", [])
    if {(p["operator"], p["priority"]) for p in phases} != set(PRIORITIES.items()):
        raise ValueError("phase priority drift")
    axes = document.get("hardware_mapping_search_axes", [])
    if not axes or not all(isinstance(axis, str) and axis.strip() for axis in axes):
        raise ValueError("missing hardware mapping search axes")
    expected = read_source_targets(snapshot)
    actual = set()
    for target in document.get("targets", []):
        operator = target["operator"]
        key = (operator, target["n"], target["batch"])
        if type(target["n"]) is not int or type(target["batch"]) is not int:
            raise ValueError("target shape must use integer N and batch")
        if key in actual:
            raise ValueError(f"duplicate manifest target: {key}")
        actual.add(key)
        if target.get("priority") != PRIORITIES.get(operator):
            raise ValueError(f"target priority drift: {key}")
    missing, extra = set(expected) - actual, actual - set(expected)
    if missing or extra:
        raise ValueError(f"target set mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/convergence_targets.json")
    parser.add_argument("--check", action="store_true", help="validate only; do not list targets")
    args = parser.parse_args()
    try:
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        targets = validate(document)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"convergence targets invalid: {error}\n")
    print(f"Validated {len(targets)} strict sub-unity targets from {SNAPSHOT}; no hardware executed.")
    if not args.check:
        print("priority operator N batch baseline source_speedup")
        for (operator, n, batch), speedup in sorted(
                targets.items(), key=lambda item: (PRIORITIES[item[0][0]], item[0][1:])):
            print(f"{PRIORITIES[operator]} {operator} {n} {batch} "
                  f"{BASELINES[operator]['runner']} {speedup:.6f}x")


if __name__ == "__main__":
    main()
