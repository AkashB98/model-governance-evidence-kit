import hashlib
import unittest

from tests.helpers import mk
from evidence import (build_evidence, evidence_hash, evidence_json,
                      input_hash, render_markdown)
from generator import generate_portfolio


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.records = generate_portfolio()
        self.ev = build_evidence(self.records)

    def test_seeded_verdicts(self):
        self.assertEqual(self.ev["overall_verdict"], "fail")
        self.assertEqual(self.ev["verdicts"],
                         {"drift": "fail", "accuracy": "fail", "variance": "pass"})

    def test_control_verdict_pass(self):
        ev = build_evidence(generate_portfolio(drift=False, miscalibrated=False))
        self.assertEqual(ev["overall_verdict"], "pass")

    def test_byte_identical_across_runs(self):
        again = build_evidence(generate_portfolio())
        self.assertEqual(evidence_json(self.ev), evidence_json(again))
        self.assertEqual(evidence_hash(self.ev), evidence_hash(again))

    def test_input_hash_is_sha256_of_records(self):
        self.assertEqual(len(self.ev["input_sha256"]), 64)
        self.assertEqual(self.ev["input_sha256"], input_hash(self.records))
        changed = generate_portfolio(seed=43)
        self.assertNotEqual(input_hash(changed), self.ev["input_sha256"])

    def test_metadata_fixed(self):
        self.assertEqual(self.ev["generated_at"], "2026-11-04T00:00:00Z")
        self.assertEqual(self.ev["model_name"], "synthetic-credit-score-v1")
        self.assertTrue(self.ev["deterministic"])

    def test_markdown_headlines(self):
        md = render_markdown(self.ev)
        self.assertIn("Overall verdict: FAIL", md)
        self.assertIn(self.ev["input_sha256"], md)
        self.assertIn("periodic model-level validation evidence for an auditor", md)
        self.assertEqual(md, render_markdown(build_evidence(self.records)))

    def test_invalid_records_raise(self):
        with self.assertRaises(ValueError):
            build_evidence([mk(label=5)])


if __name__ == "__main__":
    unittest.main()
