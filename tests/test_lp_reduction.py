# 有界 LP 归约 + 标准形单纯形的独立验收：最优值必须与手算一致，恢复解必须满足原始数据。
import math
import unittest

import numpy as np

from zyo.lp_reduction import solve_bounded_lp


class LPReductionTests(unittest.TestCase):
    def setUp(self):
        # min -x-2y ; x+y<=4 ; x+3y<=6 ; x,y>=0 -> 最优 (3,1)，目标 -5。
        self.body = np.array([[1.0, 1.0], [1.0, 3.0]])
        self.rhs = np.array([4.0, 6.0])
        self.le = ['<=', '<=']
        self.cost = np.array([-1.0, -2.0])

    def check(self, result, expected_objective, expected_values=None, places=7):
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, expected_objective, places=places)
        self.assertLessEqual(result.standard.get('row_violation', 0.0), 1e-7)
        self.assertLessEqual(result.standard.get('bound_violation', 0.0), 1e-7)
        if expected_values is not None:
            np.testing.assert_allclose(result.values, expected_values, atol=1e-6)
        return result

    # ---------- 手算基准 ----------

    def test_nonnegative_variables_match_the_hand_solution(self):
        result = solve_bounded_lp(self.body, self.cost, np.zeros(2),
                                  np.array([math.inf, math.inf]), self.rhs, sense=self.le)
        self.check(result, -5.0, [3.0, 1.0])

    def test_finite_upper_bounds_constrain_the_solution(self):
        # x,y <= 2。此时 (2,2) 违反 x+3y<=6（=8），可行最优是 x+3y=6 与 x=2 的交点
        # (2, 4/3)，目标 -2-8/3 = -14/3 ≈ -4.6667；独立网格枚举确认同一值。
        result = solve_bounded_lp(self.body, self.cost, np.zeros(2),
                                  np.array([2.0, 2.0]), self.rhs, sense=self.le)
        self.check(result, -14.0/3.0, [2.0, 4.0/3.0])

    def test_lower_bounded_variables(self):
        # x,y >= 1 ; x+y <= 4 ; min x+y -> (1,1)，目标 2。
        result = solve_bounded_lp(np.array([[1.0, 1.0]]), np.array([1.0, 1.0]),
                                  np.array([1.0, 1.0]), np.array([math.inf, math.inf]),
                                  np.array([4.0]), sense=['<='])
        self.check(result, 2.0, [1.0, 1.0])

    # ---------- 已覆盖的情形 ----------
    #
    # 归约 + 两阶段单纯形现已覆盖：非负、区间上界、下界、只上界、自由变量、等式行、
    # 不等式行（含负右端）、最大化、不可行行、无界。等式行**不再**获得松弛列，
    # 否则 == 会被放松成 <=（见下面的 test_equality_rows_are_not_relaxed）。

    def test_free_variables(self):
        # x,y 自由 ; x+y >= 5 ; x-y == 0 ; min x+y -> (2.5,2.5)，目标 5（网格枚举确认）。
        result = solve_bounded_lp(np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([1.0, 1.0]),
                                  np.array([-math.inf, -math.inf]),
                                  np.array([math.inf, math.inf]),
                                  np.array([5.0, 0.0]), sense=['>=', '=='])
        self.check(result, 5.0, [2.5, 2.5])

    def test_equality_rows_are_not_relaxed(self):
        # x-y == 4 ; x,y >= 0 ; min x+y -> x=4, y=0，目标 4。
        # 若给等式行加松弛列，约束会退化成 x-y <= 4，最优变成 (0,0) 目标 0（曾实测到）。
        result = solve_bounded_lp(np.array([[1.0, -1.0]]), np.array([1.0, 1.0]),
                                  np.zeros(2), np.array([math.inf, math.inf]),
                                  np.array([4.0]), sense=['=='])
        self.check(result, 4.0, [4.0, 0.0])

    def test_upper_bounded_only_variables_keep_their_unbounded_below_direction(self):
        # x,y <= 0（无下界）; x+y >= -5 ; min x+2y -> 真最优 (0,-5)，目标 -10（网格枚举）。
        # 归约把 x 写成 u-z 后**不能**再加界行 z+s=u，否则等于偷偷补上 x >= 0，
        # 把可行域压成 {(0,0)}；此前实测返回 (0,0) 并报行违反 5。
        result = solve_bounded_lp(np.array([[1.0, 1.0]]), np.array([1.0, 2.0]),
                                  np.array([-math.inf, -math.inf]), np.zeros(2),
                                  np.array([-5.0]), sense=['>='])
        self.check(result, -10.0, [0.0, -5.0])

    def test_unbounded_objective_is_reported(self):
        # min -x-2y ; x-y <= 1 ; x,y >= 0：y 可任意增大而不违反任何约束，目标无下界。
        # 归约是双射，所以标准形报告的 UNBOUNDED 必须原样传回原始问题。
        result = solve_bounded_lp(np.array([[1.0, -1.0]]), np.array([-1.0, -2.0]),
                                  np.zeros(2), np.array([math.inf, math.inf]),
                                  np.array([1.0]), sense=['<='])
        self.assertEqual(result.status, 'UNBOUNDED', result.message)
        self.assertLessEqual(result.standard.get('row_violation', 0.0), 1e-7)

    def test_equality_rows_without_sense_argument_are_treated_as_equalities(self):
        # 默认 sense=None：每行都是等式。x+y=4 ; x+3y=6 -> (3,1)，min -x-2y = -5。
        result = solve_bounded_lp(self.body, self.cost, np.zeros(2),
                                  np.array([math.inf, math.inf]), self.rhs)
        self.check(result, -5.0, [3.0, 1.0])

    # ---------- 失败与边界情形 ----------

    def test_infeasible_rows_are_reported(self):
        # x >= 0 且 x <= -1 不可行。
        result = solve_bounded_lp(np.array([[1.0]]), np.array([1.0]), np.array([0.0]),
                                  np.array([math.inf]), np.array([-1.0]), sense=['<='])
        self.assertNotEqual(result.status, 'OPTIMAL')

    def test_bad_sense_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_bounded_lp(np.array([[1.0]]), np.array([1.0]), np.array([0.0]),
                             np.array([math.inf]), np.array([1.0]), sense=['<>'])

    # ---------- 随机题独立复核 ----------

    def test_returned_point_satisfies_original_data_for_random_cases(self):
        rng = np.random.default_rng(20260912)
        checked = 0
        for _ in range(10):
            n, m = 4, 3
            body = rng.normal(size=(m, n))
            rhs = rng.uniform(1.0, 6.0, size=m)
            cost = rng.normal(size=n)
            lower = np.zeros(n)
            upper = rng.uniform(2.0, 6.0, size=n)
            result = solve_bounded_lp(body, cost, lower, upper, rhs, sense=['<=']*m)
            if result.status != 'OPTIMAL':
                continue
            checked += 1
            self.assertLessEqual(np.max(body @ result.values), np.max(rhs)+1e-7)
            for row in range(m):
                self.assertLessEqual(float(body[row] @ result.values-rhs[row]), 1e-7)
            self.assertGreaterEqual(np.min(result.values-lower), -1e-7)
            self.assertLessEqual(np.max(result.values-upper), 1e-7)
            self.assertAlmostEqual(result.objective, float(cost @ result.values), places=8)
        self.assertGreater(checked, 0, 'expected at least one solvable random case')


if __name__ == '__main__':
    unittest.main()
