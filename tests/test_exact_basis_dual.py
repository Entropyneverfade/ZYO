"""小基精确对偶见证的手算正反例；不把它当作 LP 最优性测试。"""

from fractions import Fraction
import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix, hstack, identity

from zyo.exact_basis_dual import reconstruct_exact_basis_dual


class ExactBasisDualTests(unittest.TestCase):
    def test_one_by_one_recovers_binary64_rational_multiplier(self):
        result = reconstruct_exact_basis_dual(csc_matrix([[0.5]]), [0.25], [0])
        self.assertIsNone(result.refusal_reason)
        self.assertEqual(result.multipliers, (Fraction(1, 2),))

    def test_two_by_two_solves_transposed_basis_equations(self):
        # 2y0+y1=5、y0+y1=3，故 y=(2,1)。
        result = reconstruct_exact_basis_dual(
            csc_matrix([[2., 1.], [1., 1.]]), [5., 3.], [0, 1])
        self.assertEqual(result.multipliers, (Fraction(2), Fraction(1)))

    def test_singular_basis_is_refused(self):
        result = reconstruct_exact_basis_dual(
            csc_matrix([[1., 2.], [2., 4.]]), [1., 2.], [0, 1])
        self.assertIsNone(result.multipliers)
        self.assertEqual(result.refusal_reason, 'singular_basis')

    def test_row_and_column_limits_are_refused_before_dense_work(self):
        many_rows = reconstruct_exact_basis_dual(identity(9, format='csc'),
                                                 np.zeros(9), list(range(9)))
        self.assertEqual(many_rows.refusal_reason, 'rows_out_of_scope')
        many_columns = reconstruct_exact_basis_dual(
            csc_matrix((1, 65)), np.zeros(65), [0])
        self.assertEqual(many_columns.refusal_reason, 'columns_out_of_scope')

    def test_nonfinite_basis_coefficient_and_cost_are_refused(self):
        for matrix, costs in ((csc_matrix([[math.inf]]), [1.]),
                              (csc_matrix([[1.]]), [math.nan])):
            with self.subTest(matrix=matrix.data.tolist(), costs=costs):
                result = reconstruct_exact_basis_dual(matrix, costs, [0])
                self.assertIsNone(result.multipliers)
                self.assertEqual(result.refusal_reason, 'nonfinite_input')

    def test_bit_budget_refuses_large_input_or_intermediate_rational(self):
        result = reconstruct_exact_basis_dual(
            csc_matrix([[2.**-50]]), [1.], [0], max_bits=32)
        self.assertIsNone(result.multipliers)
        self.assertEqual(result.refusal_reason, 'bit_budget')

    def test_invalid_basic_indices_are_refused(self):
        matrix = identity(2, format='csc')
        for basic in ([0, 0], [0, 2], [0], [0, 1.5]):
            with self.subTest(basic=basic):
                result = reconstruct_exact_basis_dual(matrix, [1., 1.], basic)
                self.assertIsNone(result.multipliers)
                self.assertEqual(result.refusal_reason, 'invalid_basis')

    def test_four_by_nine_auxiliary_basis_has_hand_derived_multipliers(self):
        # 源于现有四变量方向辅助模型：两自由变量拆正负列，外加四行逻辑列。
        oriented_transpose = csc_matrix([[1., 0.], [0., 1.],
                                         [1., -2.], [-2., 1.]])
        auxiliary = hstack((oriented_transpose,
                            csc_matrix(np.ones((4, 1)))), format='csc')
        wide = hstack((auxiliary[:, [2]], auxiliary[:, [0, 1]],
                       -auxiliary[:, [0, 1]], -identity(4, format='csc')),
                      format='csc')
        costs = [1., 0., 0., 0., 0., 0., 0., 0., 0.]
        basic = [1, 6, 0, 2]
        result = reconstruct_exact_basis_dual(wide, costs, basic)
        self.assertEqual(result.multipliers,
                         (Fraction(1, 2), Fraction(0), Fraction(1, 6), Fraction(1, 3)))
        self.assertIsNone(result.refusal_reason)
        # 逐列用输入的原始 binary64 系数独立重算 B^T y=c_B。
        for column in basic:
            exact_column = [Fraction.from_float(float(value))
                            for value in wide[:, column].toarray().reshape(-1)]
            self.assertEqual(sum(a*y for a, y in zip(exact_column, result.multipliers)),
                             Fraction.from_float(float(costs[column])))


if __name__ == '__main__':
    unittest.main()
