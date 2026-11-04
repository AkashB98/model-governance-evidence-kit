import unittest

from tests.helpers import mk  # noqa: F401  (path setup)
from generator import FIELDS, generate_portfolio, records_to_csv, records_to_jsonl


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_portfolio(), generate_portfolio())

    def test_counts_and_windows(self):
        recs = generate_portfolio(per_window=50)
        self.assertEqual(len(recs), 100)
        self.assertEqual(sum(1 for r in recs if r["window"] == "reference"), 50)
        self.assertEqual(sum(1 for r in recs if r["window"] == "current"), 50)

    def test_fields_and_ranges(self):
        for r in generate_portfolio(per_window=30):
            self.assertEqual(list(sorted(r)), list(sorted(FIELDS)))
            self.assertTrue(0.01 <= r["score"] <= 0.99)
            self.assertIn(r["label"], (0, 1))

    def test_unique_ids(self):
        ids = [r["record_id"] for r in generate_portfolio(per_window=40)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_seed_changes_output(self):
        self.assertNotEqual(generate_portfolio(seed=1), generate_portfolio(seed=2))

    def test_negative_per_window_raises(self):
        with self.assertRaises(ValueError):
            generate_portfolio(per_window=-1)

    def test_zero_per_window(self):
        self.assertEqual(generate_portfolio(per_window=0), [])

    def test_drift_shifts_current_dti(self):
        drifted = generate_portfolio()
        clean = generate_portfolio(drift=False, miscalibrated=False)
        mean = lambda recs, w: sum(r["dti_ratio"] for r in recs
                                   if r["window"] == w) / 800
        self.assertGreater(mean(drifted, "current") - mean(drifted, "reference"), 0.10)
        self.assertLess(abs(mean(clean, "current") - mean(clean, "reference")), 0.03)

    def test_serialisers_round_trip_shape(self):
        recs = generate_portfolio(per_window=5)
        self.assertEqual(len(records_to_jsonl(recs).splitlines()), 10)
        self.assertEqual(len(records_to_csv(recs).splitlines()), 11)


if __name__ == "__main__":
    unittest.main()
