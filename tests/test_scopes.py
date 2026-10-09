"""PR #2 stage 1: lock the scopes timing vocabulary and its semantics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scopes  # noqa: E402


# Real-format captures (field order and units are part of the contract).
DEVICE_LINE = ("scopes: plan_setup=49029.1 us first_use=50230.8 us "
               "host_end_to_end mean=16529.5 min=14930.6 us "
               "h2d=2500.0 us device_chain=9900.0 us d2h=2400.0 us reps=5")
HOST_LINE = ("scopes: plan_setup=49029.1 us first_use=50230.8 us "
             "host_end_to_end mean=49539.7 min=47909.4 us "
             "h2d=2000.0 us device_chain=546.6 us d2h=1980.0 us reps=5")
SHORT_LINE = ("scopes: plan_setup=900.0 us first_use=1100.0 us "
              "host_end_to_end mean=90.0 min=85.0 us "
              "h2d=NA device_chain=40.0 us d2h=NA reps=3")


class ParseTest(unittest.TestCase):
    def test_device_line_fields(self):
        f = scopes.parse_scopes(DEVICE_LINE)
        self.assertEqual(sorted(f), sorted(scopes.FIELDS))
        self.assertAlmostEqual(f["device_chain"], 9900.0)
        self.assertAlmostEqual(f["h2d"], 2500.0)
        self.assertAlmostEqual(f["d2h"], 2400.0)
        self.assertAlmostEqual(f["e2e_mean"], 16529.5)
        self.assertAlmostEqual(f["e2e_min"], 14930.6)
        self.assertEqual(f["reps"], 5)

    def test_host_line_fields(self):
        f = scopes.parse_scopes(HOST_LINE)
        self.assertAlmostEqual(f["device_chain"], 546.6)
        self.assertEqual(f["reps"], 5)

    def test_short_line_na(self):
        f = scopes.parse_scopes(SHORT_LINE)
        self.assertIsNone(f["h2d"])
        self.assertIsNone(f["d2h"])
        self.assertAlmostEqual(f["device_chain"], 40.0)

    def test_retired_device_only_rejected(self):
        legacy = ("scopes: plan_setup=1.0 us first_use=2.0 us "
                  "host_end_to_end mean=10.0 min=9.0 us "
                  "device_only=4.0 us reps=3")
        with self.assertRaises(ValueError):
            scopes.parse_scopes(legacy)

    def test_missing_field_rejected(self):
        with self.assertRaises(ValueError):
            scopes.parse_scopes(DEVICE_LINE.replace(" d2h=2400.0 us", ""))

    def test_not_a_scopes_line(self):
        with self.assertRaises(ValueError):
            scopes.parse_scopes("PASS maxRel=1e-7")


class ValidateTest(unittest.TestCase):
    def assertValid(self, line, mode):
        problems = scopes.validate_scopes(scopes.parse_scopes(line), mode)
        self.assertEqual(problems, [])

    def test_device_line_valid(self):
        self.assertValid(DEVICE_LINE, "long_device")
        self.assertValid(DEVICE_LINE, "long_host")

    def test_host_and_device_share_contract(self):
        for line in (HOST_LINE, DEVICE_LINE):
            f = scopes.parse_scopes(line)
            for mode in ("long_host", "long_device"):
                # both boundary styles validate under either label: the
                # vocabulary and ranges are identical, only NA rules differ
                self.assertEqual(scopes.validate_scopes(f, mode), [])
        self.assertTrue(any("unknown mode" in p for p in
                            scopes.validate_scopes(f, "unknown")))

    def test_short_requires_na(self):
        f = scopes.parse_scopes(SHORT_LINE)
        self.assertEqual(scopes.validate_scopes(f, "short"), [])
        problems = scopes.validate_scopes(f, "long_host")
        self.assertTrue(any("h2d" in p for p in problems))
        self.assertTrue(any("d2h" in p for p in problems))

    def test_short_with_measured_transfers_rejected(self):
        f = scopes.parse_scopes(SHORT_LINE.replace("h2d=NA", "h2d=10.0 us")
                                .replace("d2h=NA", "d2h=11.0 us"))
        problems = scopes.validate_scopes(f, "short")
        self.assertTrue(any("must report h2d=NA" in p for p in problems))
        self.assertTrue(any("must report d2h=NA" in p for p in problems))

    def test_transfers_inside_chain_breaks_budget(self):
        # fold both transfers into device_chain: component sum exceeds wall
        f = scopes.parse_scopes(
            DEVICE_LINE.replace("device_chain=9900.0 us",
                                "device_chain=14800.0 us")
                       .replace("h2d=2500.0 us", "h2d=NA")
                       .replace("d2h=2400.0 us", "d2h=NA"))
        problems = scopes.validate_scopes(f, "long_device")
        self.assertTrue(any("h2d must be measured" in p for p in problems))

    def test_budget_above_wall_rejected(self):
        f = scopes.parse_scopes(
            DEVICE_LINE.replace("host_end_to_end mean=16529.5 min=14930.6",
                                "host_end_to_end mean=6000.0 min=5900.0"))
        problems = scopes.validate_scopes(f, "long_device")
        self.assertTrue(any("below h2d+device_chain+d2h" in p
                            for p in problems))

    def test_nonpositive_chain_rejected(self):
        f = scopes.parse_scopes(DEVICE_LINE.replace("device_chain=9900.0 us",
                                                    "device_chain=0.0 us"))
        problems = scopes.validate_scopes(f, "long_device")
        self.assertTrue(any("device_chain must be > 0" in p for p in problems))

    def test_infer_mode(self):
        self.assertEqual(scopes.infer_mode(
            scopes.parse_scopes(SHORT_LINE), True), "short")
        self.assertEqual(scopes.infer_mode(
            scopes.parse_scopes(DEVICE_LINE), True), "long_device")
        self.assertEqual(scopes.infer_mode(
            scopes.parse_scopes(HOST_LINE), False), "long_host")


# P0 six-segment decomposition (PR #2 performance comment): real capture from
# `AB_BOUNDARY=device ./build/fft_check 8192 1 3` on ascend910_9382.
SEG_DEVICE_SCOPES = ("scopes: plan_setup=752.5 us first_use=882.4 us "
                     "host_end_to_end mean=348.6 min=342.0 us "
                     "h2d=61.2 us device_chain=167.0 us d2h=62.6 us reps=3")
SEG_DEVICE_LINE = ("segments: transpose_in=57.8 us fft1=9.4 us twiddle=6.1 us "
                   "transpose_boundary=43.0 us fft2=7.4 us transpose_out=43.3 us")
SEG_NA_LINE = "segments: NA"


class SegmentsTest(unittest.TestCase):
    def test_parse_device_segments(self):
        seg = scopes.parse_segments(SEG_DEVICE_LINE)
        self.assertEqual(sorted(seg), sorted(scopes.SEGMENT_FIELDS))
        self.assertAlmostEqual(seg["transpose_in"], 57.8)
        self.assertAlmostEqual(seg["transpose_out"], 43.3)

    def test_parse_na(self):
        self.assertIsNone(scopes.parse_segments(SEG_NA_LINE))

    def test_missing_segment_rejected(self):
        with self.assertRaises(ValueError):
            scopes.parse_segments(SEG_DEVICE_LINE.replace(" twiddle=6.1 us", ""))

    def test_unexpected_field_rejected(self):
        with self.assertRaises(ValueError):
            scopes.parse_segments(SEG_DEVICE_LINE + " device_only=1.0 us")

    def test_not_a_segments_line(self):
        with self.assertRaises(ValueError):
            scopes.parse_segments("scopes: plan_setup=1.0 us")

    def test_device_segments_telescope_into_chain(self):
        fields = scopes.parse_scopes(SEG_DEVICE_SCOPES)
        seg = scopes.parse_segments(SEG_DEVICE_LINE)
        self.assertEqual(scopes.validate_segments(seg, fields,
                                                  "long_device"), [])
        self.assertAlmostEqual(sum(seg.values()),
                               fields["device_chain"], places=1)

    def test_device_requires_segments(self):
        fields = scopes.parse_scopes(SEG_DEVICE_SCOPES)
        problems = scopes.validate_segments(None, fields, "long_device")
        self.assertTrue(any("six segments" in p for p in problems))

    def test_na_required_on_host_and_short(self):
        fields = scopes.parse_scopes(SEG_DEVICE_SCOPES)
        for mode in ("long_host", "short"):
            self.assertEqual(scopes.validate_segments(None, fields, mode), [])
            problems = scopes.validate_segments(
                scopes.parse_segments(SEG_DEVICE_LINE), fields, mode)
            self.assertTrue(any("must be NA" in p for p in problems))

    def test_sum_mismatch_rejected(self):
        # segments that no longer telescope with the reported chain span
        fields = scopes.parse_scopes(
            SEG_DEVICE_SCOPES.replace("device_chain=167.0 us",
                                      "device_chain=9900.0 us"))
        seg = scopes.parse_segments(SEG_DEVICE_LINE)
        problems = scopes.validate_segments(seg, fields, "long_device")
        self.assertTrue(any("sum(segments)=" in p for p in problems))

    def test_negative_segment_rejected(self):
        fields = scopes.parse_scopes(SEG_DEVICE_SCOPES)
        seg = scopes.parse_segments(
            SEG_DEVICE_LINE.replace("fft1=9.4 us", "fft1=-1.0 us"))
        problems = scopes.validate_segments(seg, fields, "long_device")
        self.assertTrue(any("fft1 must be >= 0" in p for p in problems))

    def test_unknown_mode_rejected(self):
        problems = scopes.validate_segments(None, {}, "sideways")
        self.assertTrue(any("unknown mode" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
