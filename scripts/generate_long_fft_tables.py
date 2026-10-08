#!/usr/bin/env python3
"""Generate planned experiment tables, never measured results, from the manifest."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "long_fft_experiments.json"
DEFAULT_OUT = ROOT / "docs" / "generated" / "long-fft-tables.md"
COMMON_FIELDS = (
    ("experiment_id", "Experiment identifier from the manifest."),
    ("case_id", "Stable experiment-local shape identifier."),
    ("N", "Points per transform; record dimensions separately for multidimensional cases."),
    ("batch", "Independent transforms per execution."),
    ("total_points", "N times batch; equal-work series must match the declared total."),
    ("transform", "C2C, R2C or C2R; direction and normalization must also be recorded."),
    ("precision", "Input, output and accumulation precision."),
    ("direction", "Forward or inverse; never combine incompatible transform directions."),
    ("normalization", "Exact normalization applied to the output."),
    ("placement", "In-place or out-of-place, including any required copy."),
    ("layout", "Dense/strided layout and output order; compare identical data contracts."),
    ("timing_scope", "Device-only, host end-to-end, plan setup or first use; compare identical scopes."),
    ("application_id", "Predeclared workload proxy identifier, or an explicit not-applicable value."),
    ("input_mode", "Numerical input pattern or application proxy generator."),
    ("seed", "Recorded random seed; use an explicit not-applicable value for analytical inputs."),
    ("hardware", "Device identifier and hardware profile revision."),
    ("software", "CANN, compiler and baseline library versions."),
    ("revision", "Exact repository revision used to build the tested binary."),
    ("implementation", "Exact library/backend and selected configuration, not a version label alone."),
    ("configuration_id", "Stable mapping/core identifier shared with selected-configs.json and ablation pairs."),
    ("selected_config", "Complete selected architecture mapping and processing-unit parameters."),
    ("warmup", "Warmup executions excluded from steady-state timing."),
    ("repeat", "Timed executions within the trial."),
    ("trial", "Independent trial index; preserve every raw trial."),
    ("status", "planned, passed, failed or unsupported; planned is not a measured result."),
)


def cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def metric_fields(metrics):
    fields = []
    for metric in metrics:
        if isinstance(metric, str):
            fields.append((metric, "Record the measured value and unit; leave blank before execution."))
        elif isinstance(metric, dict):
            name = metric.get("name", metric.get("id"))
            if not name:
                raise ValueError("metric objects require name or id")
            fields.append((name, metric.get("description", metric.get("unit", "Measured result."))))
        else:
            raise ValueError("metrics must contain strings or objects")
    return fields


def load_manifest(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    experiments = document.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("manifest requires a nonempty experiments list")
    ids = set()
    for experiment in experiments:
        for key in ("id", "slug", "title", "runnable", "shape_spec", "metrics"):
            if key not in experiment:
                raise ValueError(f"experiment requires {key}")
        if experiment["id"] in ids:
            raise ValueError("duplicate experiment id")
        ids.add(experiment["id"])
        if not isinstance(experiment["runnable"], bool):
            raise ValueError("runnable must be a boolean")
        if not isinstance(experiment["shape_spec"], dict):
            raise ValueError("shape_spec must be an object")
        derived = experiment["shape_spec"].get("derive_from")
        if derived is not None:
            if derived != "applications" or not experiment.get("applications"):
                raise ValueError("derive_from must name a nonempty applications list")
        metric_fields(experiment["metrics"])
    return document


def expand_shapes(experiment):
    if "shapes" in experiment:
        return experiment["shapes"]
    specs = experiment.get("applications") or [experiment["shape_spec"]]
    shapes = []
    for spec in specs:
        context = spec.get("id", "")
        for n in spec.get("ns", []):
            if not isinstance(n, int) or n <= 0:
                raise ValueError("FFT lengths must be positive integers")
            points = [(batch, None) for batch in spec.get("batches", [])]
            for total in spec.get("fixed_total_points", []):
                if not isinstance(total, int) or total <= 0 or total % n:
                    raise ValueError("fixed_total_points must be positive and divisible by every length")
                points.append((total // n, total))
            for batch, total in points:
                if not isinstance(batch, int) or batch <= 0:
                    raise ValueError("batch sizes must be positive integers")
                case_id = f"{experiment['id']}-N{n}-B{batch}"
                if total is not None:
                    case_id += f"-P{total}"
                if context:
                    case_id += f"-{context}"
                shapes.append({"id": case_id, "n": n, "batch": batch,
                               "total_points": total if total is not None else n * batch,
                               "workload": context})
    return shapes


def generate(document):
    lines = ["# Long FFT Experiment Tables", "",
             "<!-- Generated by scripts/generate_long_fft_tables.py; edit the manifest instead. -->", "",
             "These are predeclared test conditions and empty result templates, not benchmark results.",
             "`planned` means the required implementation or runner is not available. Even a runnable",
             "experiment remains `not measured` until raw trials and correctness evidence are collected.", "",
             "See [the experiment plan](../benchmarks/long-fft-plan.md) for hypotheses and execution policy.", ""]
    for experiment in document["experiments"]:
        status = "not measured" if experiment["runnable"] else "planned"
        lines.extend([f"## {cell(experiment['id'])}: {cell(experiment['title'])}", "",
                      f"**Status:** {status}. **Purpose:** {cell(experiment.get('hypothesis', ''))}", ""])
        for key, label in (("shape_spec", "Shape rule"), ("pairs", "Ablation pairs"),
                           ("inputs", "Numerical inputs"), ("random_seeds", "Random seeds"),
                           ("hardware_note", "Hardware note"), ("controlled_variables", "Controls"),
                           ("baselines", "Baselines"), ("required_artifacts", "Required evidence"),
                           ("acceptance", "Acceptance")):
            if key in experiment:
                lines.extend([f"**{label}:** {cell(experiment[key])}", ""])
        lines.extend(["### Predeclared Cases", "",
                      "| case_id | N / dimensions | batch | total points | workload | transform | precision | status |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        shapes = expand_shapes(experiment)
        for index, shape in enumerate(shapes, start=1):
            if not isinstance(shape, dict):
                raise ValueError("each shape must be an object")
            case_id = shape.get("id", shape.get("case_id", f"{experiment['id']}-{index:03d}"))
            n = shape.get("N", shape.get("n", shape.get("dimensions", "unspecified")))
            target = document.get("target", {})
            values = (case_id, n, shape.get("batch", "unspecified"),
                      shape.get("total_points", ""), shape.get("workload", ""),
                      shape.get("transform", experiment.get("transform", target.get("transform", "C2C"))),
                      shape.get("precision", experiment.get("precision", target.get("precision", "fp32"))), status)
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        if not shapes:
            lines.append(f"| {experiment['id']}-pending | unspecified | unspecified | | | unspecified | unspecified | {status} |")
        fields = list(COMMON_FIELDS) + metric_fields(experiment["metrics"])
        if len({name for name, _ in fields}) != len(fields):
            raise ValueError("metric names must be unique and cannot duplicate common CSV fields")
        lines.extend(["", "### Result CSV Contract", "", "```csv",
                      ",".join(name for name, _ in fields), "```", "",
                      "No result rows are emitted: populate the CSV only from actual executions.", "",
                      "| field | meaning |", "| --- | --- |"])
        lines.extend(f"| `{cell(name)}` | {cell(description)} |" for name, description in fields)
        lines.extend(["", "### Empty Summary", "",
                      "| implementation / configuration | valid trials | median device time | baseline time | speedup | maximum error |",
                      "| --- | --- | --- | --- | --- | --- |",
                      "|  |  |  |  |  |  |", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="Check that output matches the manifest without writing.")
    args = parser.parse_args(argv)
    try:
        content = generate(load_manifest(args.config))
        if args.check:
            if not args.out.is_file() or args.out.read_text(encoding="utf-8") != content:
                print(f"stale or missing generated tables: {args.out}", file=sys.stderr)
                return 1
            return 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
