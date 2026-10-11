"""R0.2: the P0 report must be recomputable from the committed archives.

Hardware-free: re-renders docs/benchmarks/long-fft-p0-report.md in memory
and compares against the committed file, so no performance number in the
report can drift outside the lock chain (archives + baseline + msprof json).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_p0_report as gen  # noqa: E402

SOURCES = (gen.DEVICE, gen.HOST, gen.BASELINE, gen.MSPROF)
# PR-B: the fused archive is cited only once it exists (pre-PR-B renders
# stay byte-identical, and the lock chain must not demand a missing file).
FUSED_PRESENT = gen.FUSED.is_file()
ALL_SOURCES = SOURCES + ((gen.FUSED,) if FUSED_PRESENT else ())


class P0ReportTest(unittest.TestCase):
    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_committed_report_matches_archives(self):
        self.assertEqual(gen.OUT.read_text(encoding="utf-8"), gen.render())

    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_report_cites_only_archived_sources(self):
        text = gen.render()
        for path in ALL_SOURCES:
            self.assertIn(path.relative_to(ROOT).as_posix(), text)

    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_check_mode_passes(self):
        self.assertEqual(gen.main(["--check"]), 0)

    @unittest.skipUnless(FUSED_PRESENT, "fused archive not present")
    def test_fused_section_rendered_when_archive_present(self):
        text = gen.render()
        self.assertIn("## 9. PR-B R1", text)
        self.assertIn(gen.FUSED.relative_to(ROOT).as_posix(), text)
        self.assertIn("boundary_impl: fused", text)

    @unittest.skipIf(FUSED_PRESENT, "fused archive present")
    def test_fused_section_absent_without_archive(self):
        text = gen.render()
        self.assertNotIn("## 9. PR-B R1", text)
        self.assertNotIn("见 §9", text)


if __name__ == "__main__":
    unittest.main()
