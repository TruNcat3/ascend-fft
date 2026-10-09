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


class P0ReportTest(unittest.TestCase):
    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_committed_report_matches_archives(self):
        self.assertEqual(gen.OUT.read_text(encoding="utf-8"), gen.render())

    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_report_cites_only_archived_sources(self):
        text = gen.render()
        for path in SOURCES:
            self.assertIn(path.relative_to(ROOT).as_posix(), text)

    @unittest.skipUnless(all(p.is_file() for p in SOURCES),
                         "evidence archives not present")
    def test_check_mode_passes(self):
        self.assertEqual(gen.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
