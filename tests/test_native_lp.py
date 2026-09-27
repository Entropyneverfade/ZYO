# 自研 LP 一站式入口的独立验收：最优值必须由独立 KKT 证书确认，热启动复用必须可审计。
# 所有预期值来自手算或网格枚举，不取自被测求解器。
import math
import unittest

import numpy as np

from zyo.native_lp import build_homogeneous, solve_certified_lp

INF = math.inf


class NativeLPTests(unittest.TestCase):
    def setUp(self):
        # min -x-2y ; x+y<=4 ; x+3y<=6 ; x,y>=0 -> (3,1)，目标 -5（手算）。
        self.body = np.array([[1.0, 1.0], [1.0, 3.0]])
        self.rhs = np.array([4.0, 6.0])
        self.sense = ['<=', '<=']
        self.cost = np.array([-1.0, -2.0])
        self.lower = np.zeros(2)
        self.upper = np.array([INF, INF])

    def solve(self, **overrides):
        arguments = dict(matrix=self.body, costs=self.cost, lower=self.lower,
                         upper=self.upper, rhs=self.rhs, sense=self.sense)
        arguments.update(overrides)
        return solve_certified_lp(**arguments)

    # ---------- 齐次编码 ----------

    def test_homogeneous_encoding_is_exact_in_both_directions(self):
        wide, lower, upper, row_lower, row_upper, logical = build_homogeneous(
            self.body, self.lower, self.upper, self.rhs, sense=self.sense)
        self.assertEqual(wide.shape, (2, 4))
        self.assertEqual(logical, [2, 3])
        # 行界由逻辑变量的界承载：s_i = a_i'x ∈ (-inf, rhs_i]。
        np.testing.assert_allclose(lower[2:], [-INF, -INF])
        np.testing.assert_allclose(upper[2:], self.rhs)
        # 对任意可行 x，取 s = A x 后齐次方程精确成立。
        point = np.array([3.0, 1.0, 4.0, 6.0])
        self.assertLessEqual(float(np.max(np.abs(wide @ point))), 1e-12)

    def test_equality_rows_become_fixed_logical_variables(self):
        wide, lower, upper, row_lower, row_upper, logical = build_homogeneous(
            self.body, self.lower, self.upper, self.rhs, sense=['==', '=='])
        np.testing.assert_allclose(lower[2:], self.rhs)
        np.testing.assert_allclose(upper[2:], self.rhs)

    def test_bad_sense_entries_are_rejected(self):
        with self.assertRaises(ValueError):
            build_homogeneous(self.body, self.lower, self.upper, self.rhs, sense=['<>', '<='])

    # ---------- 手算基准 + 证书 ----------

    def test_hand_solution_is_confirmed_by_the_certificate(self):
        result = self.solve()
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        np.testing.assert_allclose(result.values, [3.0, 1.0], atol=1e-7)
        self.assertIsNotNone(result.certificate)
        self.assertTrue(result.certificate.verified, result.certificate.reason)
        # 手算行乘子 λ=(-0.5,-0.5)（<= 行取负号），由齐次对偶直接给出。
        np.testing.assert_allclose(result.dual, [-0.5, -0.5], atol=1e-7)
        self.assertLessEqual(result.record['multiplier_identity_gap'], 1e-9)

    def test_maximization_returns_the_user_direction_objective(self):
        # max x+2y ; x+y<=4 ; x+3y<=6 -> (3,1)，目标 5（网格枚举）。
        result = self.solve(costs=np.array([1.0, 2.0]), maximize=True)
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=8)
        np.testing.assert_allclose(result.values, [3.0, 1.0], atol=1e-7)
        self.assertTrue(result.certificate.verified, result.certificate.reason)

    def test_bounded_variables_are_respected(self):
        # x,y <= 2 时最优 (2, 4/3)，目标 -14/3（网格枚举确认）。
        result = self.solve(upper=np.array([2.0, 2.0]))
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, -14.0/3.0, places=8)
        np.testing.assert_allclose(result.values, [2.0, 4.0/3.0], atol=1e-7)

    def test_equality_rows_are_not_relaxed(self):
        # x-y == 4 ; x,y >= 0 ; min x+y -> (4,0)，目标 4。
        result = solve_certified_lp(matrix=np.array([[1.0, -1.0]]), costs=np.array([1.0, 1.0]),
                                   lower=np.zeros(2), upper=np.array([INF, INF]),
                                   rhs=np.array([4.0]), sense=['=='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 4.0, places=8)

    def test_greater_equal_rows_are_handled(self):
        # x+y >= 5 ; x-y == 0 ; min x+y -> (2.5,2.5)，目标 5。
        result = solve_certified_lp(matrix=np.array([[1.0, 1.0], [1.0, -1.0]]),
                                   costs=np.array([1.0, 1.0]), lower=np.zeros(2),
                                   upper=np.array([INF, INF]), rhs=np.array([5.0, 0.0]),
                                   sense=['>=', '=='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=8)

    def test_infeasible_problem_is_reported_without_a_certificate(self):
        result = solve_certified_lp(matrix=np.array([[1.0]]), costs=np.array([1.0]),
                                   lower=np.array([0.0]), upper=np.array([INF]),
                                   rhs=np.array([-1.0]), sense=['<='])
        self.assertEqual(result.status, 'INFEASIBLE', result.message)
        self.assertFalse(result.optimal)
        self.assertIsNone(result.certificate)

    def test_unbounded_problem_is_reported_without_a_certificate(self):
        result = solve_certified_lp(matrix=np.array([[1.0, -1.0]]), costs=np.array([-1.0, -2.0]),
                                   lower=np.zeros(2), upper=np.array([INF, INF]),
                                   rhs=np.array([1.0]), sense=['<='])
        self.assertEqual(result.status, 'UNBOUNDED', result.message)
        self.assertFalse(result.optimal)

    # ---------- 随机题：证书即判据 ----------

    def test_random_bounded_lps_are_certified(self):
        rng = np.random.default_rng(20260914)
        certified = 0
        for trial in range(12):
            rows, columns = 3, 4
            body = rng.normal(size=(rows, columns))
            rhs = rng.uniform(1.0, 6.0, size=rows)
            lower = np.zeros(columns)
            upper = rng.uniform(2.0, 6.0, size=columns)
            cost = rng.normal(size=columns)
            result = solve_certified_lp(body, cost, lower, upper, rhs, sense=['<=']*rows)
            if result.status != 'OPTIMAL':
                continue
            certified += 1
            self.assertTrue(result.optimal, f'trial {trial}: {result.message}')
            self.assertTrue(result.certificate.verified, result.certificate.reason)
            # 独立复核：原始可行 + 目标一致。
            self.assertLessEqual(float(np.max(body @ result.values-rhs)), 1e-7)
            self.assertGreaterEqual(float(np.min(result.values-lower)), -1e-7)
            self.assertLessEqual(float(np.max(result.values-upper)), 1e-7)
            self.assertAlmostEqual(result.objective, float(cost @ result.values), places=8)
        self.assertGreater(certified, 0, 'expected at least one optimal random case')

    # ---------- 热启动 ----------

    def test_warm_basis_is_reused_when_still_feasible(self):
        first = self.solve()
        self.assertTrue(first.optimal, first.message)
        # 仅收紧目标系数，最优基不变：复用应被接受。
        second = self.solve(costs=np.array([-1.0, -1.0]), warm_basis=first.basic)
        self.assertTrue(second.record['warm_start']['reusable'],
                        second.record['warm_start']['reason'])
        self.assertTrue(second.optimal, second.message)

    def test_warm_basis_rejection_is_recorded_and_the_solve_continues(self):
        first = self.solve()
        # 把 x 的上界压到 1：原最优基（x=3）不再可行，复用必须被拒并留痕。
        # 新最优手算：min -x-2y 即 max x+2y，在 x<=1 下 y <= min(4-x, (6-x)/3) = 5/3
        # （x=1 时），故最优点 (1, 5/3)、目标 -1-10/3 = -13/3 ≈ -4.3333。
        # 注意 (1,3) 不是可行点：x+3y = 10 > 6。
        second = self.solve(upper=np.array([1.0, INF]), warm_basis=first.basic)
        self.assertFalse(second.record['warm_start']['reusable'])
        self.assertIn('not primal feasible', second.record['warm_start']['reason'])
        self.assertTrue(second.optimal, second.message)
        self.assertAlmostEqual(second.objective, -13.0/3.0, places=8)
        np.testing.assert_allclose(second.values, [1.0, 5.0/3.0], atol=1e-7)

    def test_reused_solution_matches_the_cold_start(self):
        first = self.solve()
        warm = self.solve(warm_basis=first.basic)
        cold = self.solve()
        self.assertEqual(warm.status, cold.status)
        self.assertAlmostEqual(warm.objective, cold.objective, places=9)
        np.testing.assert_allclose(warm.values, cold.values, atol=1e-9)


    # ---------- 自由变量：精确拆分 ----------
    #
    # 有界变量形式要求每个非基变量停在某一侧界上，自由变量没有界可停，此前一律报
    # "Phase-I could not establish a basis"（实测 MIPLIB 的 roll3000 有 1 个自由变量、
    # sct2 有 141 个）。现在按 x_j = p_j − n_j 精确拆分：投影到原变量上是满射，可行集与目标
    # 值不变；最优性条件等价（拆分要求 r⁺ ≥ 0 且 r⁻ ≥ 0，合起来正是自由变量的 r_j = 0）。

    def test_free_variables_are_split_and_the_certificate_confirms_it(self):
        # x, y 自由 ; x+y >= 5 ; x-y == 0 ; min x+y -> (2.5,2.5)，目标 5（网格枚举确认）。
        result = solve_certified_lp(
            matrix=np.array([[1.0, 1.0], [1.0, -1.0]]), costs=np.array([1.0, 1.0]),
            lower=np.array([-INF, -INF]), upper=np.array([INF, INF]),
            rhs=np.array([5.0, 0.0]), sense=['>=', '=='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=8)
        np.testing.assert_allclose(result.values, [2.5, 2.5], atol=1e-6)
        self.assertTrue(result.certificate.verified, result.certificate.reason)
        # 自由变量的简约成本必须为零（证书对自由变量正是这条要求）。
        for index in (0, 1):
            self.assertAlmostEqual(result.certificate.reduced_costs[index], 0.0, places=7)
        self.assertEqual(result.record['free_variables'], 2)
        self.assertEqual(result.record['split_free_variables'], 2)
        self.assertEqual(result.record['augmented_columns'], 4)

    def test_a_mixture_of_free_lower_and_bounded_variables(self):
        # 自由 x、下界 y >= 1；x + y >= 5 ; min x + 2y -> y 取 1，x 取 4，目标 6。
        result = solve_certified_lp(
            matrix=np.array([[1.0, 1.0]]), costs=np.array([1.0, 2.0]),
            lower=np.array([-INF, 1.0]), upper=np.array([INF, INF]),
            rhs=np.array([5.0]), sense=['>='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 6.0, places=7)
        np.testing.assert_allclose(result.values, [4.0, 1.0], atol=1e-6)

    def test_free_variable_with_an_upper_bound_only_is_not_split(self):
        # 只上界变量不是自由变量，不应触发拆分（它有自己的界可停）。
        result = solve_certified_lp(
            matrix=np.array([[1.0, 1.0]]), costs=np.array([1.0, 1.0]),
            lower=np.array([-INF, 0.0]), upper=np.array([3.0, INF]),
            rhs=np.array([2.0]), sense=['>='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 2.0, places=7)
        self.assertEqual(result.record['free_variables'], 0)
        self.assertNotIn('split_free_variables', result.record)

    def test_a_free_variable_keeps_the_hand_optimum_in_a_wider_model(self):
        # 与 lp_reduction 的手算题同源：自由 x,y；x+y >= 5 ; x-y == 0 ; 目标 x+y = 5。
        # 再叠加一个等式把解钉住：x + 2y == 7.5 -> 与 x=y 联立得 x=y=2.5。
        result = solve_certified_lp(
            matrix=np.array([[1.0, 1.0], [1.0, -1.0], [1.0, 2.0]]),
            costs=np.array([1.0, 1.0]), lower=np.array([-INF, -INF]),
            upper=np.array([INF, INF]), rhs=np.array([5.0, 0.0, 7.5]),
            sense=['>=', '==', '=='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=7)
        np.testing.assert_allclose(result.values, [2.5, 2.5], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
