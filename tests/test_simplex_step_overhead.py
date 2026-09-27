# 基值变化由手算方程独立规定；轨迹测试防止提速改变翻界、退化及上界方向。
import unittest

import numpy as np
from scipy.sparse import csc_matrix

import zyo.sparse_simplex as simplex


class BasicValueStepTests(unittest.TestCase):
    def advance(self, values, indices, moving, step):
        helper = getattr(simplex, '_advance_basic_values', None)
        self.assertTrue(callable(helper), '需要有可独立验证的基值更新操作')
        helper(values, np.asarray(indices, dtype=np.intp), np.asarray(moving), step)

    def test_unordered_basis_and_mixed_signs_preserve_nonbasic_values(self):
        values = np.array([91., -4., 73., 10., 2.])
        self.advance(values, [3, 1, 4], [2., -3., .5], 2.)
        np.testing.assert_array_equal(values, [91., 2., 73., 6., 1.])

    def test_zero_step_preserves_every_value(self):
        values = np.array([-3., 2., 10.])
        self.advance(values, [2, 0], [5., -7.], 0.)
        np.testing.assert_array_equal(values, [-3., 2., 10.])

    def test_separate_multiply_and_subtract_match_scalar_extreme_scales(self):
        values = np.array([1e100, -1e-100, 1e-12, -2., 3.])
        expected = values.copy()
        indices, moving, step = [4, 0, 1, 3], np.array([2., 1e101, -1e-99, -3.]), .125
        # 独立逐项参考，不用被测函数生成预期；不允许合并为改变舍入的乘加。
        for position, column in enumerate(indices):
            expected[column] -= step*moving[position]
        self.advance(values, indices, moving, step)
        np.testing.assert_array_equal(values, expected)

    def test_entering_and_basic_compensation_preserve_original_equations(self):
        # B=diag(2,3)，入列(4,-6)，B^-1a=(2,-2)，x0从0增到1.5。
        matrix = np.array([[4., 0., 2.], [-6., 3., 0.]])
        values = np.array([0., 8., 5.])
        original_rhs = matrix @ values
        values[0] += 1.5
        self.advance(values, [2, 1], [2., -2.], 1.5)
        np.testing.assert_array_equal(values, [1.5, 11., 2.])
        np.testing.assert_array_equal(matrix @ values, original_rhs)


class DriverTrajectoryTests(unittest.TestCase):
    def test_hand_two_pivots_keep_actions_and_objectives(self):
        result = simplex.revised_simplex(csc_matrix([[1., 0., -1., 0.], [0., 1., 0., -1.]]),
                    [-2., -1., 0., 0.], np.zeros(4), [np.inf, np.inf, 1., 1.],
                    basic=[2, 3], initial={0: 0., 1: 0.}, iteration_limit=2)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual([(h['entering'], h['leaving'], h['step'], h['objective'])
                          for h in result.history], [(0, 2, 1., -2.), (1, 3, 1., -3.)])

    def test_flip_keeps_basis_and_equations(self):
        result = simplex.revised_simplex(csc_matrix([[1., -1.]]), [-1., 0.],
                    [0., 0.], [1., 10.], basic=[1], initial={0: 0.}, iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.history[0]['kind'], 'bound_flip')
        np.testing.assert_array_equal(result.values, [1., 1.])

    def test_zero_pivot_still_keeps_exact_zero_trajectory(self):
        result = simplex.revised_simplex(csc_matrix([[1., 1.]]), [-1., 0.],
                    [0., 0.], [np.inf, np.inf], basic=[1], iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual((result.history[0]['step'], result.history[0]['objective']), (0., 0.))
        np.testing.assert_array_equal(result.values, [0., 0.])

    def test_maximisation_from_upper_bound_moves_in_correct_direction(self):
        result = simplex.revised_simplex(csc_matrix([[1., -1.]]), [-1., 0.],
                    [0., 0.], [2., 5.], basic=[1], initial={0: 2.},
                    maximize=True, iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.history[0]['kind'], 'bound_flip')
        self.assertEqual(result.objective, 0.)
        np.testing.assert_array_equal(result.values, [0., 0.])


if __name__ == '__main__':
    unittest.main()
