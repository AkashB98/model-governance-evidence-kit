import unittest

from tests.helpers import mk
from generator import generate_portfolio
from validation import validate_records


class TestValidation(unittest.TestCase):
    def test_generated_clean(self):
        self.assertEqual(validate_records(generate_portfolio(per_window=20)), [])

    def test_missing_field(self):
        r = mk()
        del r["score"]
        self.assertTrue(any("score" in i for i in validate_records([r])))

    def test_score_out_of_range(self):
        self.assertTrue(validate_records([mk(score=1.5)]))

    def test_bool_label_rejected(self):
        self.assertTrue(validate_records([mk(label=True)]))

    def test_bad_window(self):
        self.assertTrue(validate_records([mk(window="future")]))

    def test_duplicate_id(self):
        self.assertTrue(validate_records([mk(), mk()]))

    def test_non_list(self):
        self.assertEqual(validate_records("nope"), ["records must be a list"])


if __name__ == "__main__":
    unittest.main()
