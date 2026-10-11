"""P2 (PR#4 review): pp2 family schema and shape-aware static dispatch.

Hardware-free: eligible_pp2_shapes counts chain-gate candidates across
independent clean sessions; select_family implements the phase-1 static
table (default nb, pp2 only for >=2-session shapes, explicit fallback
for the pinned-E2E driver-blocked shape).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import long_fft_pp2_dispatch as disp  # noqa: E402


def _archive(candidates, problems=None, incomplete=None):
    return {
        "chain_gate": {"candidates": list(candidates), "promoted": True},
        "problems": problems or [],
        "incomplete": incomplete or [],
        "status": "pass",
    }


def _write(tmp, name, doc):
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "attribution.json").write_text(json.dumps(doc))


class FamilySchemaTest(unittest.TestCase):
    def test_pp2_family_constants(self):
        self.assertEqual(disp.PP2_FAMILY["tile_h"], 64)
        self.assertEqual(disp.PP2_FAMILY["tile_w"], 32)
        self.assertEqual(disp.PP2_FAMILY["pipe"], "pp2")
        self.assertEqual(disp.PP2_FAMILY["idx_mode"], "resident")
        self.assertEqual(disp.PP2_FAMILY["stripe_k"], 256)
        self.assertEqual(disp.PP2_FAMILY["ub_bytes"], 135168)
        self.assertLessEqual(disp.PP2_FAMILY["ub_bytes"], disp.PP2_UB_BUDGET)

    def test_default_family_is_nb(self):
        self.assertEqual(disp.DEFAULT_FAMILY["family"], "nb")
        self.assertEqual(disp.DEFAULT_FAMILY["pipe"], "nb")


class EligibleShapesTest(unittest.TestCase):
    def test_requires_two_independent_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "long-fft-boundary-attribution-pp2",
                   _archive(["16384x47", "32768x47"]))
            _write(tmp, "long-fft-boundary-attribution-pp2-session2",
                   _archive(["16384x47", "32768x47", "8192x1"]))
            got = disp.eligible_pp2_shapes(tmp)
            # 8192x1 only in one session -> not eligible
            self.assertEqual(got, frozenset({(16384, 47), (32768, 47)}))

    def test_problem_archives_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "long-fft-boundary-attribution-pp2",
                   _archive(["16384x47"], problems=["bad"]))
            _write(tmp, "long-fft-boundary-attribution-pp2-session2",
                   _archive(["16384x47"]))
            got = disp.eligible_pp2_shapes(tmp)
            self.assertEqual(got, frozenset())   # only 1 clean session

    def test_incomplete_archives_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "long-fft-boundary-attribution-pp2",
                   _archive(["16384x47"], incomplete=["missing"]))
            _write(tmp, "long-fft-boundary-attribution-pp2-session2",
                   _archive(["16384x47"]))
            self.assertEqual(disp.eligible_pp2_shapes(tmp), frozenset())

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(disp.eligible_pp2_shapes(tmp), frozenset())


class SelectFamilyTest(unittest.TestCase):
    ELIGIBLE = frozenset({(16384, 47), (32768, 47), (65536, 47)})

    def test_eligible_shape_selects_pp2(self):
        r = disp.select_family(32768, 47, eligible=self.ELIGIBLE)
        self.assertEqual(r["selected_family"], "pp2")
        self.assertIn("2 independent sessions", r["selection_reason"])
        self.assertIsNone(r["fallback_reason"])

    def test_ineligible_shape_stays_nb(self):
        r = disp.select_family(64, 1, eligible=self.ELIGIBLE)
        self.assertEqual(r["selected_family"], "nb")
        self.assertIn("insufficient", r["selection_reason"])
        self.assertIsNone(r["fallback_reason"])

    def test_driver_blocked_shape_has_fallback_note(self):
        r = disp.select_family(65536, 47, eligible=self.ELIGIBLE,
                               e2e_host_mode="pinned")
        self.assertEqual(r["selected_family"], "pp2")
        self.assertIn("unsupported-driver-IOMMU", r["fallback_reason"])

    def test_driver_blocked_pageable_has_no_fallback(self):
        r = disp.select_family(65536, 47, eligible=self.ELIGIBLE,
                               e2e_host_mode="pageable")
        self.assertEqual(r["selected_family"], "pp2")
        self.assertIsNone(r["fallback_reason"])

    def test_production_never_ready_for_pp2(self):
        # phase 1: pp2 is always opt-in; the selector never claims
        # production readiness (pinned E2E is still diagnostic).
        r = disp.select_family(32768, 47, eligible=self.ELIGIBLE)
        self.assertNotIn("production_ready", r)   # not claimed at all
        self.assertEqual(disp.DEFAULT_FAMILY["family"], "nb")


class RealEvidenceTest(unittest.TestCase):
    """Integration against the checked-in R5 archives."""

    def test_real_eligible_shapes(self):
        got = disp.eligible_pp2_shapes()
        # both sessions promoted 16384x47 / 32768x47 / 65536x47;
        # 8192x1 only session2 -> not eligible
        self.assertIn((16384, 47), got)
        self.assertIn((32768, 47), got)
        self.assertIn((65536, 47), got)
        self.assertNotIn((8192, 1), got)

    def test_real_select_smoke(self):
        r = disp.select_family(32768, 47)
        self.assertEqual(r["selected_family"], "pp2")
        r2 = disp.select_family(1024, 64)
        self.assertEqual(r2["selected_family"], "nb")


if __name__ == "__main__":
    unittest.main()
