#!/usr/bin/env python3
"""Merge r2c_c2r supplement (n=64/8192) into the main 42-row result — fail closed.

Usage: python3 scripts/merge_r2c_supp.py <main.json> <suppl.json>
Rewrites main.json (+ .raw_trials.json/.raw_trials.csv) with merged rows.

Merge refuses to proceed (exit 2) when the two sessions are not
interchangeable: differing measurement provenance (hardware / CANN /
measurement commit / dirty state), differing protocol, or overlapping
(n, b) rows / raw trials.  The merged document records `merged_from`
with the supplement's path and sha256.
"""
import csv, hashlib, io, json, os, sys

IDENTITY_FIELDS = ("hardware", "cann", "git_sha", "git_dirty")


def die(msg):
    print(f"merge_r2c_supp: REFUSING TO MERGE: {msg}", file=sys.stderr)
    sys.exit(2)


def load(path, what):
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"{what} unreadable ({path}): {exc}")


def main():
    if len(sys.argv) != 3:
        die("usage: merge_r2c_supp.py <main.json> <suppl.json>")
    main_p, suppl_p = sys.argv[1], sys.argv[2]
    main = load(main_p, "main")
    suppl = load(suppl_p, "supplement")
    if not isinstance(main, dict) or not isinstance(suppl, dict):
        die("inputs must be JSON objects with protocol/provenance/rows")

    # --- protocol must match exactly -------------------------------
    if main.get("protocol") != suppl.get("protocol"):
        die(f"protocol differs:\n  main : {main.get('protocol')}\n"
            f"  suppl: {suppl.get('protocol')}")

    # --- measurement identity must match ---------------------------
    mprov, sprov = main.get("provenance") or {}, suppl.get("provenance") or {}
    for field in IDENTITY_FIELDS:
        if mprov.get(field) != sprov.get(field):
            die(f"provenance.{field} differs: main={mprov.get(field)!r} "
                f"suppl={sprov.get(field)!r}")

    # --- overlapping rows are an error, not something to skip ------
    rows = main.get("rows", [])
    srows = suppl.get("rows", [])
    have = {(r.get("n"), r.get("b")) for r in rows}
    overlap = [(r.get("n"), r.get("b")) for r in srows
               if (r.get("n"), r.get("b")) in have]
    if overlap:
        die(f"overlapping rows between sessions: {sorted(overlap)}")
    added = len(srows)
    rows = rows + srows
    rows.sort(key=lambda r: (r["n"], r["b"]))
    main["rows"] = rows

    # --- overlapping raw trials are an error, not deduplicated ------
    raw_p = main_p.replace(".json", ".raw_trials.json")
    suppl_raw_p = suppl_p.replace(".json", ".raw_trials.json")
    raw = load(raw_p, "main raw trials")
    sraw = load(suppl_raw_p, "supplement raw trials")
    trials = raw.get("raw_trials", raw if isinstance(raw, list) else [])
    straws = sraw.get("raw_trials", sraw if isinstance(sraw, list) else [])
    seen = {(t.get("impl"), t.get("n"), t.get("b"), t.get("trial"))
            for t in trials}
    clash = [(t.get("impl"), t.get("n"), t.get("b"), t.get("trial"))
             for t in straws
             if (t.get("impl"), t.get("n"), t.get("b"), t.get("trial")) in seen]
    if clash:
        die(f"overlapping raw trials between sessions: {sorted(clash)}")
    added_t = len(straws)
    trials = trials + straws
    if isinstance(raw, dict):
        raw["raw_trials"] = trials
    else:
        raw = trials

    # --- merged document records its supplement source ---------------
    data = open(suppl_p, "rb").read()
    main["merged_from"] = {
        "path": os.path.relpath(suppl_p),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    json.dump(main, open(main_p, "w"), indent=1, ensure_ascii=False)
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
