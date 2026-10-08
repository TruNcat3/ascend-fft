#!/usr/bin/env python3
"""Collect the long-FFT dynamic-input acceptance evidence (P1-A / G1 envelope).

Runs the committed `fft_check` binary over the G1 grid
N={8192,16384,32768,65536} x B={1,3,47} with an A/B/A input sequence from raw
float32 files (file contents change between calls), plus a short-path A/B/A
control and the E2E single-input/single-output transfer assertion. Results are
written to results/evidence/long-fft-acceptance/ as acceptance.json + outputs.txt
so the public long-FFT claim audits against a git-tracked snapshot.

  python3 scripts/collect_long_fft_evidence.py
"""
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "evidence" / "long-fft-acceptance"
NS = (8192, 16384, 32768, 65536)
BS = (1, 3, 47)
THRESHOLD = 1e-4


def gen_input(path, n, batch, pattern):
    elems = n * batch
    if pattern == "A":  # impulse at k = batch % n
        data = [0.0] * (2 * elems)
        k = batch % n
        for b in range(batch):
            data[b * 2 * n + 2 * k] = 1.0
    else:  # B: deterministic pseudo-random
        data = []
        state = 0x9E3779B9 ^ (n * 131 + batch)
        for _ in range(2 * elems):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            data.append(((state >> 8) & 0xFFFF) / 32768.0 - 1.0)
    path.write_bytes(struct.pack(f"<{len(data)}f", *data))


def run(args, env=None):
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          env=env, timeout=3600)
    return proc.returncode, proc.stdout + proc.stderr


def parse_point(out):
    m = re.search(r"maxAbs=([\d.eE+-]+) maxRel=([\d.eE+-]+)", out)
    seq = re.findall(r"seq\[\d+\]=(\S+) maxRel=([\d.eE+-]+) (PASS|FAIL)", out)
    seq_line = re.search(r"^seq: .*$", out, re.M)
    return {
        "max_rel": float(m.group(2)) if m else None,
        "max_abs": float(m.group(1)) if m else None,
        "seq": [{"input": s[0], "max_rel": float(s[1]), "pass": s[2] == "PASS"}
                for s in seq],
        "seq_summary": seq_line.group(0) if seq_line else "",
        "pass": bool(re.search(r"^PASS$", out, re.M)),
    }


def main():
    binary = ROOT / "build" / "fft_check"
    if not binary.is_file():
        print("build/fft_check missing; run scripts/build.sh check",
              file=sys.stderr)
        return 2
    points = []
    transcripts = []
    with tempfile.TemporaryDirectory(prefix="lfft-evi-") as tmp:
        tmp = Path(tmp)
        for n in NS:
            for b in BS:
                fa = tmp / f"A_{n}_{b}.bin"
                fb = tmp / f"B_{n}_{b}.bin"
                gen_input(fa, n, b, "A")
                gen_input(fb, n, b, "B")
                seq = f"{fa},{fb},{fa}"
                rc, out = run(["./build/fft_check", str(n), str(b), "3"],
                              env=dict(os.environ, AB_INPUT_SEQ=seq))
                point = {"n": n, "b": b, "rc": rc, **parse_point(out),
                         "e2e_transfers": re.search(
                             r"^(E2E transfers: .*)$", out, re.M).group(1)
                         if re.search(r"^E2E transfers: .*$", out, re.M) else ""}
                points.append(point)
                transcripts.append(f"===== long A/B/A n={n} b={b} rc={rc} =====\n{out}")
        rc, out = run(["./build/fft_check", "4096", "3", "3"],
                      env=dict(os.environ, AB_INPUT_SEQ="impulse,random-seeded,impulse"))
        control = {"n": 4096, "b": 3, "rc": rc, **parse_point(out),
                   "path": "short"}
        transcripts.append(f"===== short A/B/A control rc={rc} =====\n{out}")
        rc, out = run(["./build/fft_check", "8192", "1", "3"],
                      env=dict(os.environ, AB_E2E="3"))
        transfers = re.search(r"^(E2E transfers: .*)$", out, re.M)
        e2e = {"rc": rc, "transfers": transfers.group(1) if transfers else "",
               "line": (re.search(r"^(E2E n=.*)$", out, re.M).group(1)
                        if re.search(r"^E2E n=.*$", out, re.M) else "")}
        transcripts.append(f"===== E2E transfer assertion rc={rc} =====\n{out}")

    ok = (all(p["pass"] and p["max_rel"] is not None
              and p["max_rel"] <= THRESHOLD for p in points)
          and all(s["pass"] for p in points for s in p["seq"])
          and control["pass"] and control["rc"] == 0
          and e2e["rc"] == 0 and "in=1 out=1" in e2e["transfers"])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    document = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        "command": "python3 scripts/collect_long_fft_evidence.py",
        "binary": "build/fft_check (AB_INPUT_SEQ A/B/A file inputs)",
        "threshold": THRESHOLD,
        "grid": {"ns": list(NS), "bs": list(BS)},
        "points": points,
        "short_control": control,
        "e2e_transfers": e2e,
        "status": "pass" if ok else "fail",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "outputs.txt").write_text("\n".join(transcripts), encoding="utf-8")
    (OUT / "acceptance.json").write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"{sum(1 for p in points if p['pass'])}/{len(points)} long points, "
          f"control={'PASS' if control['pass'] else 'FAIL'}, "
          f"e2e={e2e['transfers']!r} -> {document['status']} -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
