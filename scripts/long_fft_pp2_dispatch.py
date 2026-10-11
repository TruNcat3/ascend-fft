#!/usr/bin/env python3
"""R5/pp2 family schema and shape-aware static dispatch (phase 1).

P2 (PR#4 review): pp2 (64x32 half-tile + full double-buffer) is a
different dataflow family from the R4 OLS feature space (128x32 +
pipe in {nb,pp}).  Mixing its samples into the R4 model pollutes the
fit (topk drops from 2/2 to 1/2).  This module keeps pp2 out of R4 and
provides a *static* shape-aware dispatch table instead of a learned
selector.

Phase 1 policy (explicit, no model):
  * Default production family is ``nb`` (R2-B winner, 128x32 +
    resident + K=256 + blocks=48).  pp2 is opt-in only.
  * A shape is eligible for the pp2 *candidate* list only when the
    device-chain gate promoted it in **two independent sessions**
    (the R5 acceptance floor).  Session-only wins (e.g. 8192x1 in
    session2) stay on the default family until a second session
    confirms them.
  * Eligibility is a *chain-gate* statement.  The pinned E2E gate for
    pp2 is still ``diagnostic``/``unsupported-driver`` on this driver
    (pageable collections only; 65536x47 pinned hits the IOMMU defect),
    so ``production_ready`` is always False and the runtime default
    never switches to pp2 automatically.
  * Combinations that violate the UB budget, the driver pinned-E2E
    contract, or the workspace contract report an explicit
    ``fallback_reason`` -- never a silent wrong pick.

Phase 2 (not yet implemented) will add an independent R5 model trained
on held-out shapes and a second session, separate from R4.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVID = ROOT / "results" / "evidence"

# UB budget for the pp2 64x32 resident configuration (see
# collect_boundary_attribution.ub_peak_bytes with AB_LT_* = pp2 env):
# 4 * 64 * 32 * 8 + 16 * 256 = 135168.  The on-chip UB is 196608 (192KiB);
# any config whose modeled peak exceeds the budget must fall back.
PP2_UB_BUDGET = 196608
PP2_UB_PEAK = 135168

# The pp2 family constants matching the R5 prototype entry
# kfft_lt_tr_t64x32_k256ri_pp2.
PP2_FAMILY = {
    "family": "pp2",
    "tile_h": 64,
    "tile_w": 32,
    "stripe_k": 256,
    "idx_mode": "resident",
    "pipe": "pp2",
    "ub_bytes": PP2_UB_PEAK,
    "blocks": 48,
}

# Default production family (R2-B winner).
DEFAULT_FAMILY = {
    "family": "nb",
    "tile_h": 128,
    "tile_w": 32,
    "stripe_k": 256,
    "idx_mode": "resident",
    "pipe": "nb",
    "ub_bytes": 131072,
    "blocks": 48,
}

# Shapes whose pinned E2E is known to fail on the current driver
# (R5 diagnosis: pciedma addr convert failed, launch=507001).
PINNED_E2E_DRIVER_BLOCKED = frozenset({(65536, 47)})


def chain_gate_candidates(archive_path):
    """Candidates from one attribution archive's chain_gate (pure IO helper)."""
    doc = json.loads(Path(archive_path).read_text(encoding="utf-8"))
    gate = doc.get("chain_gate") or doc.get("gate") or {}
    return frozenset(tuple(map(int, k.split("x")))
                     for k in gate.get("candidates", []))


def eligible_pp2_shapes(evidence_dir=None):
    """Shapes eligible for the pp2 candidate list: chain-gate candidates
    in >=2 independent clean sessions (the R5 acceptance floor)."""
    evidence = Path(evidence_dir) if evidence_dir else EVID
    counts = {}
    for path in sorted(evidence.glob("long-fft-boundary-attribution-pp2*/"
                                     "attribution.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("problems") or doc.get("incomplete"):
            continue
        gate = doc.get("chain_gate") or doc.get("gate") or {}
        for k in gate.get("candidates", []):
            n, b = map(int, k.split("x"))
            counts[(n, b)] = counts.get((n, b), 0) + 1
    return frozenset(shape for shape, c in counts.items() if c >= 2)


def select_family(n, batch, eligible=None, e2e_host_mode="pinned"):
    """Shape-aware static dispatch (phase 1, pure).

    Returns a dict with selected_family, selection_reason, and
    fallback_reason (None when no fallback applies).  e2e_host_mode is
    the caller's intended host buffer mode; it only affects the
    fallback_reason narrative, never the family choice itself (pp2
    stays opt-in regardless of host mode until pinned E2E is collected).
    """
    shape = (int(n), int(batch))
    if eligible is None:
        eligible = eligible_pp2_shapes()
    ub_ok = PP2_UB_PEAK <= PP2_UB_BUDGET
    if not ub_ok:
        return {
            "selected_family": "nb",
            "selection_reason": "default",
            "fallback_reason": f"pp2 ub_peak {PP2_UB_PEAK} exceeds "
                               f"budget {PP2_UB_BUDGET}",
            "e2e_host_mode": e2e_host_mode,
        }
    if shape in eligible:
        reason = ("chain-gate candidate in >=2 independent sessions; "
                  "pp2 is opt-in (pinned E2E still diagnostic)")
        fallback = None
        if shape in PINNED_E2E_DRIVER_BLOCKED and e2e_host_mode == "pinned":
            fallback = ("pinned E2E unsupported-driver-IOMMU for this "
                        "shape; use AB_E2E_HOST=pageable for diagnostics")
        return {
            "selected_family": "pp2",
            "selection_reason": reason,
            "fallback_reason": fallback,
            "e2e_host_mode": e2e_host_mode,
        }
    return {
        "selected_family": "nb",
        "selection_reason": "default (insufficient multi-session pp2 "
                            "evidence for this shape)",
        "fallback_reason": None,
        "e2e_host_mode": e2e_host_mode,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--evidence-dir", default=None)
    ap.add_argument("--e2e-host-mode", default="pinned",
                    choices=("pinned", "pageable"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    eligible = eligible_pp2_shapes(a.evidence_dir)
    result = select_family(a.n, a.batch, eligible, a.e2e_host_mode)
    result["eligible_pp2_shapes"] = sorted(f"{n}x{b}" for n, b in eligible)
    if a.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"selected_family={result['selected_family']} "
              f"reason={result['selection_reason']!r} "
              f"fallback={result['fallback_reason']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
