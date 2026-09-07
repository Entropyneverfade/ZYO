# 回归测试：故意破坏候选解，验证原始约束和目标复算能够发现问题。
import unittest

import zyo as zo
from zyo.validation import evaluate_objective, validate_candidate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.m = zo.Model("validate")
        self.x = self.m.add_var("x", lb=0, ub=2, vtype=zo.INTEGER)
        self.y = self.m.add_var("y", lb=-1, ub=4)
        self.m.add_constr(self.x + self.y <= 3)
        self.m.maximize(2 * self.x - self.y + 5)

    def test_valid_candidate_and_objective(self):
        residuals = validate_candidate(
            self.m, {"x": 2, "y": 1}, 1e-7, 1e-7
        )

        self.assertTrue(residuals.is_feasible)
        self.assertEqual(evaluate_objective(self.m, {"x": 2, "y": 1}), 8)

    def test_each_violation_is_reported_in_original_units(self):
        residuals = validate_candidate(
            self.m, {"x": 2.25, "y": 5}, 1e-7, 1e-7
        )

        self.assertAlmostEqual(residuals.bound, 1.0)
        self.assertAlmostEqual(residuals.integrality, 0.25)
        self.assertAlmostEqual(residuals.constraint, 4.25)

    def test_missing_nonfinite_or_extra_values_are_rejected(self):
        invalid = (
            {"x": 1},
            {"x": 1, "y": float("nan")},
            {"x": 1, "y": 0, "z": 4},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(zo.NumericalError):
                    validate_candidate(self.m, values, 1e-7, 1e-7)


if __name__ == "__main__":
    unittest.main()
