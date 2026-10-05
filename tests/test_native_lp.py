# 自研 LP 一站式入口的独立验收：最优值必须由独立 KKT 证书确认，热启动复用必须可审计。
# 所有预期值来自手算或网格枚举，不取自被测求解器。
import math
import unittest
from unittest.mock import patch

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

    def test_phase_one_limit_feasible_point_is_separate_from_optimal_result(self):
        # 手算 x>=1；辅助限额中间点 (x,s)=(1,1) 满足原模型，但不证明最优。
        from zyo.sparse_simplex import SimplexResult
        limited = SimplexResult('ITERATION_LIMIT', iterations=2, pivots=2,
                                phase_one={'status': 'ITERATION_LIMIT'},
                                phase_one_candidate=np.array([1., 1.]))
        with patch('zyo.native_lp.solve_lp', return_value=limited):
            result = solve_certified_lp(np.array([[1.]]), np.array([1.]),
                                        np.array([0.]), np.array([INF]),
                                        np.array([1.]), sense=['>='])
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertFalse(result.optimal)
        self.assertIsNone(result.values)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.certificate)
        self.assertIsNone(result.restart_candidate)
        np.testing.assert_allclose(result.primal_feasible_candidate, [1.])
        self.assertTrue(result.record['phase_one_candidate_check']['original_primal_feasible'])

    def test_phase_one_limit_rejects_original_infeasible_candidate(self):
        # 同题 (x,s)=(0,0) 虽满足齐次方程，却违反原始 x>=1。
        from zyo.sparse_simplex import SimplexResult
        limited = SimplexResult('ITERATION_LIMIT', iterations=2, pivots=2,
                                phase_one={'status': 'ITERATION_LIMIT'},
                                phase_one_candidate=np.array([0., 0.]))
        with patch('zyo.native_lp.solve_lp', return_value=limited):
            result = solve_certified_lp(np.array([[1.]]), np.array([1.]),
                                        np.array([0.]), np.array([INF]),
                                        np.array([1.]), sense=['>='])
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertIsNone(result.primal_feasible_candidate)
        self.assertFalse(result.record['phase_one_candidate_check']['original_primal_feasible'])

    def test_phase_two_numeric_failure_preserves_original_checked_candidate(self):
        # 第二阶段数值故障与第一阶段原域可行证据并存，二者状态严格分离。
        from zyo.sparse_simplex import SimplexResult
        failed = SimplexResult('NUMERICAL_ERROR', iterations=3, pivots=3,
                               phase_one={'status': 'PHASE_ONE_FEASIBLE'},
                               phase_one_candidate=np.array([1., 1.]))
        with patch('zyo.native_lp.solve_lp', return_value=failed):
            result = solve_certified_lp(np.array([[1.]]), np.array([1.]),
                                        np.array([0.]), np.array([INF]),
                                        np.array([1.]), sense=['>='])
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertIsNone(result.values)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.certificate)
        np.testing.assert_allclose(result.primal_feasible_candidate, [1.])
        self.assertTrue(result.record['phase_one_candidate_check']['original_primal_feasible'])

    def test_experimental_zero_pivot_policy_preserves_hand_lp_certificate(self):
        # 实验开关须经公共原生接口实际传入，手算最优题仍以独立 KKT 为准。
        result = self.solve(stable_zero_pivot=True)
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, -5.0)
        self.assertTrue(result.certificate.verified)
        self.assertTrue(result.record['pricing']['stable_zero_pivot'])

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

    def test_required_warm_start_rejection_never_runs_cold(self):
        first = self.solve()
        # 冻结分段实验要求实际从给定状态续跑；旧实现拒绝该状态后会冷启动。
        with self.assertRaisesRegex(ValueError, 'required warm start rejected'):
            self.solve(upper=np.array([1.0, INF]), warm_basis=first.basic,
                       require_warm_start=True)

    def test_reused_solution_matches_the_cold_start(self):
        first = self.solve()
        warm = self.solve(warm_basis=first.basic)
        cold = self.solve()
        self.assertEqual(warm.status, cold.status)
        self.assertAlmostEqual(warm.objective, cold.objective, places=9)
        np.testing.assert_allclose(warm.values, cold.values, atol=1e-9)

    def test_validated_lower_side_is_the_side_the_solver_actually_uses(self):
        # x0+x1=0、x0∈[-1,1]、x1∈[0,10]。复用检查在 x1=0 时可行；
        # 若求解器猜上界 10，则 x0=-10 越界，旧实现会暗中退回冷启动。
        result = solve_certified_lp(
            np.array([[1.0, 1.0]]), np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]), np.array([1.0, 10.0]), np.array([0.0]),
            sense=['=='], warm_basis=[0])
        self.assertTrue(result.record['warm_start']['reusable'])
        self.assertTrue(result.phase_one.get('warm_start'), result.phase_one)
        self.assertTrue(result.optimal, result.message)
        np.testing.assert_allclose(result.values, [0.0, 0.0], atol=1e-9)

    def test_explicit_upper_side_is_honored_by_certified_lp(self):
        # x0+x1=10、0<=x0<=1、0<=x1<=10：x1=10 才让基 x0 可行。
        result = solve_certified_lp(
            np.array([[1.0, 1.0]]), np.array([1.0, 0.0]),
            np.zeros(2), np.array([1.0, 10.0]), np.array([10.0]),
            sense=['=='], warm_basis=[0], warm_nonbasic_at_upper=[1])
        self.assertTrue(result.record['warm_start']['reusable'], result.record['warm_start'])
        self.assertTrue(result.phase_one.get('warm_start'), result.phase_one)
        self.assertTrue(result.optimal, result.message)
        np.testing.assert_allclose(result.values, [0.0, 10.0], atol=1e-9)

    def test_iteration_limited_basis_can_warm_start_a_new_call(self):
        # 本题手算最优 (3,1)、目标 -5；一次更新仅到目标 -4。
        first = self.solve(iteration_limit=1)
        self.assertEqual(first.status, 'ITERATION_LIMIT')
        self.assertFalse(first.optimal)
        self.assertIsNone(first.values)
        self.assertIsNotNone(first.restart_candidate)
        self.assertTrue(first.restart_candidate['available'])
        second = self.solve(warm_basis=first.restart_candidate['basic'],
                            warm_nonbasic_at_upper=first.restart_candidate['nonbasic_at_upper'],
                            iteration_limit=5)
        self.assertTrue(second.phase_one.get('warm_start'), second.phase_one)
        self.assertTrue(second.optimal, second.message)
        self.assertAlmostEqual(second.objective, -5.0, places=8)

    def test_limited_basis_exports_only_uncertified_internal_dual_diagnostic(self):
        # 一次枢轴到 (x,y)=(0,2)：基列的乘子为 (0,-2/3)，未入基 x 的
        # 简约成本 -1/3 违反下界对偶符号条件；这只是当前齐次基的诊断量。
        first = self.solve(iteration_limit=1)
        self.assertEqual(first.status, 'ITERATION_LIMIT')
        self.assertFalse(first.optimal)
        self.assertIsNone(first.certificate)
        self.assertIn('internal_max_dual_violation', first.record)
        self.assertAlmostEqual(first.record['internal_max_dual_violation'], 1.0/3.0,
                               places=8)
        self.assertLessEqual(first.record['internal_max_primal_violation'], 1e-12)
        # 乘子候选只供原模型独立弱对偶审计，不填正式 dual 字段。
        self.assertIsNone(first.dual)
        np.testing.assert_allclose(first.record['uncertified_row_multiplier_candidate'],
                                   [0.0, -2.0/3.0], atol=1e-8)


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
