import random
import unittest

from tests.helpers import mk
from generator import generate_portfolio
from metrics import (accuracy_report, auc, auc_bruteforce,
                     calibration_by_band, rescore, score_band,
                     variance_report)


class TestAuc(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    def test_reversed(self):
        self.assertEqual(auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)

    def test_ties_half_credit(self):
        self.assertEqual(auc([0.5, 0.5], [1, 0]), 0.5)
        self.assertEqual(auc_bruteforce([0.5, 0.5], [1, 0]), 0.5)

    def test_hand_case(self):
        scores, labels = [0.9, 0.4, 0.5, 0.1], [1, 1, 0, 0]
        self.assertEqual(auc(scores, labels), 0.75)
        self.assertEqual(auc_bruteforce(scores, labels), 0.75)

    def test_matches_bruteforce_random_with_ties(self):
        rng = random.Random(7)
        for _ in range(25):
            scores = [round(rng.random(), 1) for _ in range(30)]
            labels = [rng.randint(0, 1) for _ in range(30)]
            if len(set(labels)) < 2:
                continue
            self.assertEqual(auc(scores, labels), auc_bruteforce(scores, labels))

    def test_matches_bruteforce_on_generated_slice(self):
        recs = [r for r in generate_portfolio(per_window=40)
                if r["window"] == "current"]
        s = [r["score"] for r in recs]
        y = [r["label"] for r in recs]
        self.assertEqual(auc(s, y), auc_bruteforce(s, y))

    def test_single_class_raises(self):
        with self.assertRaises(ValueError):
            auc([0.5, 0.6], [1, 1])
        with self.assertRaises(ValueError):
            auc_bruteforce([0.5], [0])


class TestCalibration(unittest.TestCase):
    def test_score_band_edges(self):
        self.assertEqual(score_band(0.10), "S1")
        self.assertEqual(score_band(0.15), "S2")
        self.assertEqual(score_band(0.44), "S3")
        self.assertEqual(score_band(0.60), "S5")

    def test_hand_case(self):
        recs = [mk(score=0.37, label=1), mk(score=0.37, label=1),
                mk(score=0.37, label=0), mk(score=0.37, label=0)]
        row = next(r for r in calibration_by_band(recs) if r["band"] == "S3")
        self.assertEqual((row["count"], row["predicted"], row["observed"]),
                         (4, 0.37, 0.5))

    def test_planted_bad_band_caught(self):
        acc = accuracy_report(generate_portfolio())
        s3 = next(r for r in acc["calibration"] if r["band"] == "S3")
        self.assertGreater(s3["abs_error"], 0.20)
        self.assertEqual(acc["calibration_verdict"], "fail")
        self.assertEqual(acc["worst_band"], "S3")

    def test_control_calibration_passes(self):
        acc = accuracy_report(generate_portfolio(drift=False, miscalibrated=False))
        self.assertEqual(acc["calibration_verdict"], "pass")


class TestVariance(unittest.TestCase):
    def test_seeded_runs_stable(self):
        report = variance_report(generate_portfolio(per_window=50))
        self.assertEqual(report["verdict"], "pass")
        self.assertLess(report["mean_score_std"], 0.02)

    def test_large_jitter_fails(self):
        report = variance_report(generate_portfolio(per_window=50),
                                 jitter_sd=0.20)
        self.assertEqual(report["verdict"], "fail")

    def test_rescore_deterministic_and_bounded(self):
        recs = generate_portfolio(per_window=10)
        self.assertEqual(rescore(recs, 101), rescore(recs, 101))
        self.assertTrue(all(0.01 <= s <= 0.99 for s in rescore(recs, 101)))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            variance_report([])


if __name__ == "__main__":
    unittest.main()
