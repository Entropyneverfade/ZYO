# 预算预期独立手算：每次翻界或换基算一步，两阶段和人工驱除共享额度。
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from zyo.sparse_simplex import revised_simplex, solve_lp
from zyo import Model


class SimplexBudgetTests(unittest.TestCase):
    def two_moves(self, budget, *, flips=False):
        # x=s0、y=s1，min -2x-y；上限均为1，最优(1,1)。
        upper = [1., 1., 10., 10.] if flips else [np.inf, np.inf, 1., 1.]
        return revised_simplex(csc_matrix([[1., 0., -1., 0.], [0., 1., 0., -1.]]),
                               [-2., -1., 0., 0.], np.zeros(4), upper,
                               basic=[2, 3], initial={0: 0., 1: 0.}, iteration_limit=budget)

    def test_zero_budget_does_not_execute_a_pivot(self):
        result = self.two_moves(0)
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertEqual(result.pivots, 0)
        self.assertEqual(result.iterations, 0)
        np.testing.assert_array_equal(result.values, [0., 0., 0., 0.])

    def test_one_budget_stops_after_exactly_one_pivot(self):
        result = self.two_moves(1)
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertEqual(result.pivots, 1)
        self.assertEqual(result.iterations, 1)
        self.assertAlmostEqual(result.objective, -2.)

    def test_optimum_at_budget_endpoint_is_checked_without_an_extra_move(self):
        result = self.two_moves(2)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.pivots, 2)
        self.assertAlmostEqual(result.objective, -3.)

    def test_bound_flip_consumes_the_budget_without_a_pivot(self):
        result = self.two_moves(1, flips=True)
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.pivots, 0)
        np.testing.assert_array_equal(result.values, [1., 0., 1., 0.])
        self.assertEqual(result.bound_flips, 1)

    def test_zero_length_pivot_still_consumes_one_step(self):
        # x+s=0，x,s>=0，只能全零；负简约成本须用一个退化换基消除。
        result = revised_simplex(csc_matrix([[1., 1.]]), [-1., 0.],
                                 [0., 0.], [np.inf, np.inf], basic=[1], iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.pivots, 1)
        self.assertEqual(result.history[0]['step'], 0.)

    def test_two_phases_share_one_budget(self):
        # 2x-s=0；s>=2、0<=x<=3。Phase-I一步得x=1；Phase-II再一步得x=3。
        for budget, state, value in ((0, 'ITERATION_LIMIT', None),
                                     (1, 'ITERATION_LIMIT', 1.), (2, 'OPTIMAL', 3.)):
            with self.subTest(budget=budget):
                result = solve_lp(csc_matrix([[2., -1.]]), [-1., 0.],
                                  [0., 2.], [3., np.inf], iteration_limit=budget)
                self.assertEqual(result.status, state)
                self.assertEqual(result.iterations, budget)
                self.assertLessEqual(result.pivots, budget)
                if value is not None:
                    self.assertAlmostEqual(result.values[0], value)

    def test_artificial_cleanup_consumes_the_same_budget(self):
        # 固定x=0、2x=0：Phase-I无需优化移动，但人工列须一次退化驱除。
        for budget, state in ((0, 'ITERATION_LIMIT'), (1, 'OPTIMAL')):
            with self.subTest(budget=budget):
                result = solve_lp(csc_matrix([[2.]]), [1.], [0.], [0.], iteration_limit=budget)
                self.assertEqual(result.status, state)
                self.assertEqual(result.iterations, budget)
                self.assertEqual(result.pivots, budget)

    def test_public_iteration_count_includes_a_bound_flip(self):
        model = Model('flip-budget')
        x = model.add_var('x', ub=1.)
        model.add_constr(x <= 10.)
        model.minimize(-x)
        result = model.solve('native_simplex', iteration_limit=1)
        self.assertEqual(result.status.value, 'OPTIMAL')
        self.assertEqual(result.iteration_count, 1)
        self.assertAlmostEqual(result.objective, -1.)

    def test_invalid_budgets_are_rejected_before_solving(self):
        for bad in (-1, .5, True, np.inf):
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    self.two_moves(bad)


if __name__ == '__main__':
    unittest.main()
