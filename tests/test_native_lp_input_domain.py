"""公共原 LP 入口须在编码和求解前拒绝非法数值输入。"""

import math
import unittest

from scipy.sparse import csc_matrix

from zyo.native_lp import build_homogeneous, build_model, solve_certified_lp


class NativeLPInputDomainTests(unittest.TestCase):
    def setUp(self):
        self.matrix = csc_matrix([[0.0]])
        self.cost = [-1.0]
        self.lower = [0.0]
        self.upper = [1.0]
        self.rhs = [0.0]

    def test_nan_upper_bound_is_not_treated_as_unbounded(self):
        # 当前错误会把 NaN 当作无上界并把有界教学题报为 UNBOUNDED。
        with self.assertRaisesRegex(ValueError, 'upper'):
            solve_certified_lp(self.matrix, self.cost, self.lower, [math.nan],
                               self.rhs, iteration_limit=3)

    def test_build_model_rejects_nan_bound_instead_of_dropping_it(self):
        # 独立证书模型也不能把 NaN 通过 None 重写成真正无界。
        with self.assertRaisesRegex(ValueError, 'upper'):
            build_model(self.matrix, self.cost, self.lower, [math.nan], self.rhs)

    def test_homogeneous_encoder_rejects_nonfinite_matrix_and_rhs(self):
        for label, matrix, rhs in (
                ('matrix', csc_matrix([[math.nan]]), self.rhs),
                ('matrix', csc_matrix([[math.inf]]), self.rhs),
                ('right-hand side', self.matrix, [math.nan]),
                ('right-hand side', self.matrix, [math.inf])):
            with self.subTest(label=label, rhs=rhs, data=matrix.data.tolist()):
                with self.assertRaisesRegex(ValueError, label):
                    build_homogeneous(matrix, self.lower, self.upper, rhs)

    def test_rejects_wrong_signed_infinite_variable_bounds(self):
        for lower, upper, label in ((math.inf, math.inf, 'lower'),
                                    (-math.inf, -math.inf, 'upper')):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(ValueError, label):
                    solve_certified_lp(self.matrix, self.cost, [lower], [upper],
                                       self.rhs, iteration_limit=3)

    def test_public_solver_rejects_nonfinite_cost_before_solving(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(cost=bad):
                with self.assertRaisesRegex(ValueError, 'cost'):
                    solve_certified_lp(self.matrix, [bad], self.lower, self.upper,
                                       self.rhs, iteration_limit=3)

    def test_public_solver_rejects_invalid_tolerances(self):
        for label, bad in (('feasibility_tolerance', math.inf),
                           ('objective_tolerance', math.nan),
                           ('pricing_tolerance', math.nan),
                           ('pricing_tolerance', -1.0)):
            with self.subTest(label=label, value=bad):
                with self.assertRaisesRegex(ValueError, label):
                    solve_certified_lp(self.matrix, self.cost, self.lower, self.upper,
                                       self.rhs, iteration_limit=3, **{label: bad})

    def test_valid_one_sided_and_free_bounds_are_preserved(self):
        # -inf 下界和 +inf 上界是合法开放侧，不属于非法数值输入。
        matrix = csc_matrix([[1.0, 0.0]])
        wide, lower, upper, *_ = build_homogeneous(
            matrix, [-math.inf, 0.0], [math.inf, math.inf], [0.0])
        self.assertEqual(wide.shape, (1, 3))
        self.assertEqual(lower[0], -math.inf)
        self.assertEqual(upper[0], math.inf)
        self.assertEqual(upper[1], math.inf)
        model = build_model(matrix, [0.0, 0.0], [-math.inf, 0.0],
                            [math.inf, math.inf], [0.0])
        self.assertEqual(len(model.variables), 2)


if __name__ == '__main__':
    unittest.main()
