"""PR #2 stage 5: the docs table and summary must be recomputable from the
committed acceptance archives (no hand-written performance numbers)."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_long_fft_evidence as summ  # noqa: E402


def _new_schema():
    for path in summ.ACCEPTANCE.values():
        if not path.is_file():
            return False
        doc = json.loads(path.read_text(encoding="utf-8"))
        points = doc.get("points") or []
        if not points or "trials" not in points[0]:
            return False
    return True


class EvidenceSummaryTest(unittest.TestCase):
    @unittest.skipUnless(all(p.is_file() for p in summ.ACCEPTANCE.values()),
                         "acceptance archives not present")
    def test_build_is_deterministic(self):
        self.assertEqual(summ.build_summary(), summ.build_summary())

    @unittest.skipUnless(_new_schema(),
                         "pre-trial archive schema; refresh evidence first")
    def test_committed_summary_matches_archives(self):
        committed = json.loads(summ.SUMMARY_OUT.read_text(encoding="utf-8"))
        self.assertEqual(committed, summ.build_summary())

    @unittest.skipUnless(_new_schema(),
                         "pre-trial archive schema; refresh evidence first")
    def test_committed_docs_table_matches_archives(self):
        self.assertEqual(
            summ.MD_OUT.read_text(encoding="utf-8"),
            summ.render_markdown(summ.build_summary()))

    @unittest.skipUnless(_new_schema(),
                         "pre-trial archive schema; refresh evidence first")
    def test_summary_carries_both_modes_and_contracts(self):
        summary = summ.build_summary()
        self.assertEqual(set(summary["modes"]), {"host", "device"})
        self.assertIn("boundary=2", summary["modes"]["host"]["boundary"])
        self.assertIn("boundary=0", summary["modes"]["device"]["boundary"])
        self.assertTrue(summary["worst_max_rel"] <= 1e-4)
        self.assertTrue(summary["median_e2e_speedup_host_over_device"],
                        "speedup table empty")
        for mode in summary["modes"].values():
            self.assertEqual(len(mode["shapes"]), 12,
                             "every grid shape must be summarized")
            for s in mode["shapes"]:
                self.assertIsNotNone(s["e2e_us"],
                                     "trial stats missing from summary")


if __name__ == "__main__":
    unittest.main()
