import math
import unittest

from tests.helpers import mk
from drift import (EPSILON, drift_report, psi, psi_from_shares, shares,
                   verdict)
from generator import generate_portfolio

EDGES = [0.25, 0.50, 0.75]


class TestPsiMath(unittest.TestCase):
    def test_identical_distributions_zero(self):
        vals = [0.1, 0.3, 0.6, 0.9]
        self.assertEqual(psi(vals, vals, EDGES), 0.0)

    def test_hand_computed(self):
        ref = [0.1, 0.1, 0.6, 0.6]
        cur = [0.1, 0.6, 0.6, 0.6]
        expected = round((0.25 - 0.5) * math.log(0.5)
                         + (0.75 - 0.5) * math.log(1.5), 6)
        self.assertEqual(psi(ref, cur, EDGES), expected)

    def test_empty_bin_uses_epsilon(self):
        value = psi_from_shares([1.0, 0.0], [0.0, 1.0])
        expected = round((EPSILON - 1.0) * math.log(EPSILON)
                         + (1.0 - EPSILON) * math.log(1.0 / EPSILON), 6)
        self.assertEqual(value, round(expected, 6))

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(shares([0.1, 0.4, 0.9], EDGES)), 1.0)

    def test_verdict_bands(self):
        self.assertEqual(verdict(0.05), "pass")
        self.assertEqual(verdict(0.10), "warn")
        self.assertEqual(verdict(0.25), "warn")
        self.assertEqual(verdict(0.26), "fail")


class TestDriftReport(unittest.TestCase):
    def test_planted_drift_fails(self):
        report = drift_report(generate_portfolio())
        self.assertEqual(report["verdict"], "fail")
        self.assertGreater(report["score_psi"], 0.25)
        self.assertGreater(report["features"]["dti_ratio"]["csi"], 0.25)

    def test_undrifted_feature_stays_clean(self):
        report = drift_report(generate_portfolio())
        self.assertEqual(report["features"]["credit_utilization"]["verdict"], "pass")

    def test_control_passes(self):
        report = drift_report(generate_portfolio(drift=False, miscalibrated=False))
        self.assertEqual(report["verdict"], "pass")
        self.assertLess(report["score_psi"], 0.10)

    def test_missing_window_raises(self):
        with self.assertRaises(ValueError):
            drift_report([mk(window="reference"), mk(window="reference")])


if __name__ == "__main__":
    unittest.main()
