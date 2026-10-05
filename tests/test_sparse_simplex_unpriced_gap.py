"""微小未定价改善与极窄变量盒的独立手算回归。"""

import math
import unittest

from scipy.sparse import csc_matrix

from zyo.sparse_simplex import revised_simplex


class UnpricedImprovementTests(unittest.TestCase):
    def test_ignored_negative_reduced_cost_with_infinite_domain_is_not_optimal(self):
        # x0=0，x1>=0 且目标 -10^-10*x1；真实问题无界，不允许报告 0 为最优。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1e-10],
                                 [0.0, 0.0], [0.0, math.inf], basic=[0],
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.values[1], 0.0)
        self.assertEqual(result.history, [])

    def test_ignored_negative_reduced_cost_with_wide_finite_box_is_not_optimal(self):
        # x1=10^12 可行，原目标 -100；当前零点目标 0 不是全局最优。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1e-10],
                                 [0.0, 0.0], [0.0, 1e12], basic=[0],
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.history, [])

    def test_degenerate_true_optimum_can_remain_unproven(self):
        # x0+x1=0 且两者非负，仅有原点可行；负简约成本无法真正下降。
        # 浮点当前基没有完成独立对偶证书时，保守的 NUMERICAL_ERROR 比假证明安全。
        result = revised_simplex(csc_matrix([[1.0, 1.0]]), [0.0, -1e-10],
                                 [0.0, 0.0], [math.inf, 1e12], basic=[0],
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.values[1], 0.0)
        self.assertEqual(result.objective, 0.0)


class NarrowBoxSideTests(unittest.TestCase):
    def test_exact_lower_side_takes_precedence_over_near_upper(self):
        # 两界仅差10^-13，小于旧1e-12贴界门；精确下界0不能被重标成上界。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1e15],
                                 [0.0, 0.0], [0.0, 1e-13], basic=[0],
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.values[1], 1e-13)
        self.assertEqual(result.objective, -100.0)
        self.assertEqual(result.history[0]['kind'], 'bound_flip')

    def test_exact_upper_side_takes_precedence_over_near_lower(self):
        # 从精确上界出发，目标 +10^15*x1 必须下行至0。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, 1e15],
                                 [0.0, 0.0], [0.0, 1e-13], basic=[0],
                                 initial={1: 1e-13}, iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(result.values[1], 0.0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.history[0]['kind'], 'bound_flip')

    def test_interior_value_near_both_sides_is_not_arbitrarily_assigned(self):
        # 5e-14 同时落在两侧的绝对贴界门内，却非任一真实界。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1e15],
                                 [0.0, 0.0], [0.0, 1e-13], basic=[0],
                                 initial={1: 5e-14}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.history, [])


if __name__ == '__main__':
    unittest.main()
