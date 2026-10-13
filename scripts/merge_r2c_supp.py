#!/usr/bin/env python3
"""Merge r2c_c2r supplement (n=64/8192) into the main 42-row result.

Usage: python3 merge_r2c.py <main.json> <suppl.json>
Rewrites main.json (+ .raw_trials.json/.raw_trials.csv) with merged rows.
"""
import csv
import io
import json
import sys


def main():
    main_p, suppl_p = sys.argv[1], sys.argv[2]
    main = json.load(open(main_p))
    suppl = json.load(open(suppl_p))
    rows = main.get("rows", main)
    srows = suppl.get("rows", suppl)
    have = {(r["n"], r["b"]) for r in rows}
    added = 0
    for r in srows:
        if (r["n"], r["b"]) not in have:
            rows.append(r)
            added += 1
    rows.sort(key=lambda r: (r["n"], r["b"]))
    if isinstance(main, dict):
        main["rows"] = rows
    else:
        main = rows
    json.dump(main, open(main_p, "w"), indent=1, ensure_ascii=False)

    # merge raw trials
    raw_p = main_p.replace(".json", ".raw_trials.json")
    suppl_raw_p = suppl_p.replace(".json", ".raw_trials.json")
    raw = json.load(open(raw_p))
    sraw = json.load(open(suppl_raw_p))
    trials = raw.get("raw_trials", raw if isinstance(raw, list) else [])
    straws = sraw.get("raw_trials", sraw if isinstance(sraw, list) else [])
    seen = {(t.get("impl"), t.get("n"), t.get("b"), t.get("trial")) for t in trials}
    added_t = 0
    for t in straws:
        key = (t.get("impl"), t.get("n"), t.get("b"), t.get("trial"))
        if key not in seen:
            trials.append(t)
            added_t += 1
    if isinstance(raw, dict):
        raw["raw_trials"] = trials
    else:
        raw = trials
    json.dump(raw, open(raw_p, "w"), indent=1, ensure_ascii=False)

    # rebuild raw csv
    csv_p = main_p.replace(".json", ".raw_trials.csv")
    if trials:
        fields = sorted({k for t in trials for k in t})
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(trials)
        open(csv_p, "w").write(buf.getvalue())

    ok = sum(1 for r in rows if r.get("ok"))
    r2c = sum(1 for r in rows if r.get("r2c") is not None)
    c2r = sum(1 for r in rows if r.get("c2r") is not None)
    print(f"merged rows +{added} -> {len(rows)} (ok={ok}), "
          f"r2c={r2c} c2r={c2r} points; raw trials +{added_t} -> {len(trials)}")


if __name__ == "__main__":
    main()
