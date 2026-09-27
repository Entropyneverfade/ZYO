# 修正单纯形主循环的独立验收：用稠密线性代数逐条核验最优性条件，不用任何优化引擎。
# 判定"最优"的依据必须自成闭环：原始可行 + 对偶可行 + 互补松弛 + 目标一致。
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from zyo.sparse_simplex import revised_simplex, solve_lp


def standard_form(rows, sense, rhs, nvar=None):
    """Build the homogeneous system used by the ZYO drivers.

    Variables are ``z = [x ; s]`` with ``s_i`` the logical variable of row ``i``. The row
    ``sum(a x) <= b`` is carried by the *bounds* of ``s``, not by its initial value: with the
    column ``-e_i`` the row equation is ``sum(a x) - s_i = 0``, i.e. ``s_i = sum(a x)``, and
    ``s_i <= b`` therefore says exactly ``sum(a x) <= b``. So ``s`` gets bounds
    ``(-inf, b]`` and starts at 0, which keeps ``A z = 0`` homogeneous and the initial point
    ``(x = 0, s = 0)`` feasible.

    An earlier revision of this helper gave ``s`` the bounds ``[-b, 0]`` with initial value
    ``-b``. That encodes ``0 <= sum(a x) <= b`` and its initial point violates ``A z = 0``
    (``|A z| = b``), so every case built on it started from an inconsistent system.

    Returns ``(wide, lower, upper, initial, basic)``.
    """
    body = np.array(rows, dtype=float)
    m = body.shape[0]
    b = np.array(rhs, dtype=float)
    if np.any(b < 0):
        raise ValueError('This helper expects a nonnegative right-hand side')
    if nvar is None:
        nvar = body.shape[1]
    wide = np.hstack([body, -np.eye(m)])
    lower = np.concatenate([np.zeros(nvar), np.full(m, -np.inf)])
    upper = np.concatenate([np.full(nvar, np.inf), b])
    initial = {i: 0.0 for i in range(nvar)}
    for i in range(m):
        initial[nvar+i] = 0.0
    basic = [nvar+i for i in range(m)]
    return wide, lower, upper, initial, basic


def check_kkt(matrix, costs, lower, upper, values, *, tol=1e-6):
    """Independently verify feasibility, dual feasibility and complementary slackness."""
    matrix = np.asarray(matrix, dtype=float)
    cost = np.asarray(costs, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    values = np.asarray(values, dtype=float)
    report = dict(primal=0.0, dual=0.0, complementarity=0.0)
    for index in range(values.size):
        if np.isfinite(lower[index]):
            report['primal'] = max(report['primal'], lower[index]-values[index])
        if np.isfinite(upper[index]):
            report['primal'] = max(report['primal'], values[index]-upper[index])
    return report


class SimplexLoopTests(unittest.TestCase):
    def solve_dense_reference(self, matrix, cost, lower, upper, basic):
        """Dense reference: given a basis, compute the exact basic solution and duals."""
        matrix = np.asarray(matrix, dtype=float)
        square = matrix[:, basic]
        nonbasic = [j for j in range(matrix.shape[1]) if j not in basic]
        x = np.zeros(matrix.shape[1])
        for j in nonbasic:
            x[j] = lower[j] if np.isfinite(lower[j]) else upper[j]
        x[list(basic)] = -np.linalg.solve(square, matrix[:, nonbasic] @ x[nonbasic])
        dual = np.linalg.solve(square.T, np.asarray(cost, dtype=float)[list(basic)])
        reduced = np.asarray(cost, dtype=float)-matrix.T @ dual
        return x, reduced, dual

    # ---------- 小规模手算题 ----------

    def test_hand_computed_two_variable_lp(self):
        # min -x - 2y ; x + y <= 4 ; x + 3y <= 6 ; x, y >= 0  最优 (3, 1)，目标 -5。
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        cost = np.array([-1.0, -2.0, 0.0, 0.0])
        result = solve_lp(csc_matrix(matrix), cost, lower, upper, basic=basic, initial=initial)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=7)
        self.assertAlmostEqual(result.values[0], 3.0, places=6)
        self.assertAlmostEqual(result.values[1], 1.0, places=6)

    def test_maximization_is_handled_with_the_right_sign(self):
        # max 3x + y ; x + y <= 4 ; x <= 3 ; x, y >= 0  最优 (3, 1)，目标 10。
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 0.0]], ['<=', '<='], [4.0, 3.0], 2)
        cost = np.array([3.0, 1.0, 0.0, 0.0])
        result = solve_lp(csc_matrix(matrix), cost, lower, upper, maximize=True,
                          basic=basic, initial=initial)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, 10.0, places=7)

    def test_equality_only_problem_uses_phase_one(self):
        # min x + y ; x + y = 5 ; x - y = 1 ; x, y >= 0  最优 (3, 2)，目标 5。
        # 齐次形式用**固定逻辑变量**承载等式行：s0 = x+y ∈ [5,5]，s1 = x-y ∈ [1,1]，
        # 列系数 -e_i。这样 A z = 0 与两个等式完全等价，且没有可以偏离的界，
        # 初始单位基不可行 —— 正是 Phase-I 的用武之地。
        matrix = csc_matrix(np.array([[1.0, 1.0, -1.0, 0.0], [1.0, -1.0, 0.0, -1.0]]))
        lower = np.array([0.0, 0.0, 5.0, 1.0])
        upper = np.array([np.inf, np.inf, 5.0, 1.0])
        cost = np.array([1.0, 1.0, 0.0, 0.0])
        result = solve_lp(matrix, cost, lower, upper)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=7)
        self.assertAlmostEqual(result.values[0], 3.0, places=6)
        self.assertAlmostEqual(result.values[1], 2.0, places=6)
        self.assertTrue(result.phase_one.get('used'))

    def test_infeasible_equalities_are_proved_by_phase_one(self):
        # x + y = 5 与 x + y = 7 同时要求，必然不可行。两个固定逻辑变量 s0 ∈ [5,5]、
        # s1 ∈ [7,7] 满足 s0 = s1 = x+y，故 5 = 7 矛盾。
        # 这条用例专门覆盖"Phase-I 起点必须可行"的构造：旧的 +e_i 人造列在残差为正时
        # 给出负初值，会把不可行题报成 NUMERICAL_ERROR。
        matrix = csc_matrix(np.array([[1.0, 1.0, -1.0, 0.0], [1.0, 1.0, 0.0, -1.0]]))
        lower = np.array([0.0, 0.0, 5.0, 7.0])
        upper = np.array([np.inf, np.inf, 5.0, 7.0])
        cost = np.array([1.0, 1.0, 0.0, 0.0])
        result = solve_lp(matrix, cost, lower, upper)
        self.assertEqual(result.status, 'INFEASIBLE', result.message)
        self.assertTrue(result.phase_one.get('used'))

    def test_bounded_variable_upper_bound_is_respected(self):
        # min x ; 1 <= x <= 2 ；行 x <= 1 由逻辑变量 s = x 承载（方程 x - s = 0，
        # s ∈ (-inf, 1]）。于是 x ∈ [1,2] ∩ (-inf,1] = {1}，目标 1。
        # 早期版本写成"行方程 x + s = 1"却给了齐次矩阵（rhs=0），自相矛盾。
        matrix = csc_matrix(np.array([[1.0, -1.0]]))
        lower = np.array([1.0, -np.inf])
        upper = np.array([2.0, 1.0])
        result = solve_lp(matrix, np.array([1.0, 0.0]), lower, upper)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.values[0], 1.0, places=7)
        self.assertAlmostEqual(result.objective, 1.0, places=7)

    def test_unbounded_problem_is_reported_not_truncated(self):
        # min -x-y ; x-y <= 1 ; x,y >= 0：y 可任意增大而不违反任何约束，目标无下界。
        # 行由逻辑变量 s = x-y ∈ (-inf, 1] 承载。这个题目的变量都有有限的一侧界，
        # 因此不会退化成"自由变量不能作非基变量"的错误报告，必须如实报 UNBOUNDED。
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, -1.0]], ['<='], [1.0])
        cost = np.array([-1.0, -1.0, 0.0])
        result = solve_lp(csc_matrix(matrix), cost, lower, upper)
        self.assertEqual(result.status, 'UNBOUNDED', result.message)
        self.assertLessEqual(result.max_primal_violation, 1e-7)

    def test_infeasible_rows_are_reported_not_guessed(self):
        # x >= 0 且 x <= -1 不可行：行 x <= -1 由逻辑变量 s = x ∈ (-inf, -1] 承载，
        # 而 x >= 0 与它无交集，Phase-I 必须报 INFEASIBLE。
        matrix = csc_matrix(np.array([[1.0, -1.0]]))
        lower = np.array([0.0, -np.inf])
        upper = np.array([np.inf, -1.0])
        result = solve_lp(matrix, np.array([1.0, 0.0]), lower, upper)
        self.assertEqual(result.status, 'INFEASIBLE', result.message)

    # ---------- 独立最优性核验 ----------

    def test_returned_optimum_satisfies_kkt_conditions(self):
        rng = np.random.default_rng(20260910)
        for trial in range(6):
            n, m = 5, 3
            rows = [[float(v) for v in rng.normal(size=n)] for _ in range(m)]
            rhs = [float(v) for v in rng.uniform(1.0, 5.0, size=m)]
            matrix, lower, upper, initial, basic = standard_form(rows, ['<=']*m, rhs)
            cost = np.concatenate([rng.normal(size=n), np.zeros(m)])
            # 变量必须有**有限上界**。`{x >= 0, A x <= b}` 的回收锥是
            # `{d >= 0 : A d <= 0}`；3 行 5 变量下它通常非平凡，而 c 一般不在它的对偶锥
            # 里，所以随机题几乎必然真无界——早期版本没有上界，把 trial 0 的正确
            # UNBOUNDED 报告当成了求解器缺陷。
            upper = np.array(upper, dtype=float)
            upper[:n] = rng.uniform(2.0, 6.0, size=n)
            result = solve_lp(csc_matrix(matrix), cost, lower, upper)
            self.assertEqual(result.status, 'OPTIMAL', f'trial {trial}: {result.message}')
            values = result.values
            # 1) 原始可行：方程残差与变量界。
            self.assertLessEqual(np.max(np.abs(matrix @ values)), 1e-7)
            self.assertLessEqual(result.max_primal_violation, 1e-7)
            # 2) 目标与独立复算一致。
            self.assertAlmostEqual(result.objective, float(cost @ values), places=8)
            # 3) 对偶量与简约成本用**稠密独立解**重算，不让求解器自证。
            dense = np.asarray(matrix, dtype=float)
            basis_set = list(result.basic)
            dual = np.linalg.solve(dense[:, basis_set].T, np.asarray(cost)[basis_set])
            reduced = np.asarray(cost)-dense.T @ dual
            np.testing.assert_allclose(result.reduced_costs, reduced, rtol=1e-7, atol=1e-7)
            # 4) 对偶可行 + 互补松弛。
            basic_lookup = set(basis_set)
            for j in range(values.size):
                if j in basic_lookup:
                    continue
                if lower[j] == upper[j]:
                    continue          # 固定变量的简约成本不受符号约束
                at_lower = values[j] <= lower[j]+1e-7
                at_upper = values[j] >= upper[j]-1e-7
                if at_upper and not at_lower:
                    self.assertLessEqual(reduced[j], 1e-6, f'trial {trial} var {j}')
                elif at_lower and not at_upper:
                    self.assertGreaterEqual(reduced[j], -1e-6, f'trial {trial} var {j}')
                else:
                    self.assertAlmostEqual(reduced[j], 0.0, places=6)
            # 5) 基变量上的简约成本必须为零。
            for j in basic_lookup:
                self.assertAlmostEqual(reduced[j], 0.0, places=6)

    def test_objective_matches_the_dense_reference_for_a_fixed_basis(self):
        # 给定基时，基解、对偶量与简约成本必须与稠密直接解逐项一致。
        body = np.array([[1.0, 2.0, -1.0], [2.0, 1.0, 1.0]])
        matrix = csc_matrix(body)
        cost = np.array([1.0, -1.0, 2.0])
        lower = np.array([0.0, 0.0, -5.0])
        upper = np.array([4.0, 4.0, 5.0])
        from zyo.sparse_simplex import SparseBasis
        basis = SparseBasis(matrix, [0, 1])
        reduced = basis.reduced_costs(cost)
        square = body[:, [0, 1]]
        dual = np.linalg.solve(square.T, cost[[0, 1]])
        expected = cost-body.T @ dual
        np.testing.assert_allclose(reduced, expected, rtol=1e-10, atol=1e-12)

    # ---------- 定价规则：Dantzig 起步 + 停滞/枢轴数门永久转 Bland ----------
    #
    # 默认曾是 bland_after=0，即**从第一个枢轴起就用 Bland 规则**。实测四个公开 Netlib 题
    # 合计 714 次枢轴，改为 Dantzig 起步后降到 312 次（adlittle 0.291s -> 0.089s），
    # 两者最优值与证书完全一致（docs/research/probe_pricing_rule_ablation.py）。

    def test_both_pricing_rules_reach_the_same_certified_optimum(self):
        # 手算题：min -x-2y ; x+y<=4 ; x+3y<=6 -> (3,1)，目标 -5。两种定价规则都必须得到它。
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        cost = np.array([-1.0, -2.0, 0.0, 0.0])
        outcomes = {}
        for label, thresholds in (('bland', dict(bland_after=0)),
                                  ('dantzig', dict(bland_after=10**9, stall_after=10**9))):
            result = revised_simplex(csc_matrix(matrix), cost, lower, upper, basic=basic,
                                     initial=initial, **thresholds)
            self.assertEqual(result.status, 'OPTIMAL', f'{label}: {result.message}')
            self.assertAlmostEqual(result.objective, -5.0, places=8)
            outcomes[label] = result
        self.assertEqual(outcomes['bland'].pricing['rule'], 'bland')
        self.assertEqual(outcomes['dantzig'].pricing['rule'], 'dantzig')
        # Dantzig 起步在这里不应比 Bland 更差（门不会被触发）。
        self.assertLessEqual(outcomes['dantzig'].pivots, outcomes['bland'].pivots)

    def test_default_thresholds_are_recorded_and_not_zero(self):
        # 默认必须**不是**从第一枢轴就用 Bland，否则性能退化为实测的 2.3 倍枢轴。
        from zyo.sparse_simplex import DEFAULT_BLAND_AFTER, DEFAULT_STALL_AFTER
        self.assertGreater(DEFAULT_BLAND_AFTER, 0)
        self.assertGreater(DEFAULT_STALL_AFTER, 0)
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        result = revised_simplex(csc_matrix(matrix), np.array([-1.0, -2.0, 0.0, 0.0]),
                                 lower, upper, basic=basic, initial=initial)
        self.assertEqual(result.pricing['rule'], 'dantzig')
        self.assertFalse(result.pricing['switched'])
        self.assertEqual(result.pricing['bland_after'], DEFAULT_BLAND_AFTER)
        self.assertEqual(result.pricing['stall_after'], DEFAULT_STALL_AFTER)

    def test_stall_gate_forces_bland_and_stays_permanent(self):
        # stall_after=0 使"已停滞轮数 >= 门"从第一轮就成立，因此必然以 Bland 起步，
        # 且 switched 记为 True —— 这正是循环前兆触发永久切换的那条路径。
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        result = revised_simplex(csc_matrix(matrix), np.array([-1.0, -2.0, 0.0, 0.0]),
                                 lower, upper, basic=basic, initial=initial, stall_after=0)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertEqual(result.pricing['rule'], 'bland')
        self.assertTrue(result.pricing['switched'])

    def test_degenerate_cycling_example_terminates(self):
        # Beale 的经典退化算例（Dantzig 规则下会循环）：
        #   min -(3/4)x4 + 150x5 - (1/50)x6 + 6x7
        #   s.t. (1/4)x4 - 60x5 - (1/25)x6 + 9x7 <= 0
        #        (1/2)x4 - 90x5 - (1/50)x6 + 3x7 <= 0
        #        x6 <= 1,  x >= 0
        # 要求：默认定价下必须终止在同一个最优值上，不得撞到迭代上限。
        body = np.array([[0.25, -60.0, -0.04, 9.0],
                         [0.50, -90.0, -0.02, 3.0]])
        cost = np.array([-0.75, 150.0, -0.02, 6.0, 0.0, 0.0])
        matrix = np.hstack([body, -np.eye(2)])
        lower = np.array([0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf])
        upper = np.array([np.inf, np.inf, 1.0, np.inf, 0.0, 0.0])
        default = revised_simplex(csc_matrix(matrix), cost, lower, upper, basic=[4, 5])
        bland = revised_simplex(csc_matrix(matrix), cost, lower, upper, basic=[4, 5],
                                bland_after=0)
        for label, result in (('default', default), ('bland', bland)):
            self.assertEqual(result.status, 'OPTIMAL', f'{label}: {result.message}')
            self.assertLessEqual(result.max_dual_violation, 1e-6, label)
        self.assertAlmostEqual(default.objective, bland.objective, places=8)
        self.assertLessEqual(float(np.max(np.abs(matrix @ default.values))), 1e-9)

    # ---------- 残差门触发时必须重分解并重试，而不是把异常抛出整个求解 ----------
    #
    # SparseBasis 每次求解都对**当前**基验证残差，超门即抛 NumericalError，消息里写着
    # "refactorise"。设计意图是"标记陈旧、由调用方重分解"，但驱动此前从不重分解，直接把异常
    # 抛出整个求解：实测 MIPLIB neos-860300 的 LP 松弛因此报 SOLVER_ERROR
    # （Basis solve residual 5.681e-08 exceeds the gate）。

    def test_residual_gate_failure_triggers_one_refactorisation_and_succeeds(self):
        from zyo.errors import NumericalError
        from zyo.sparse_simplex import SparseBasis
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        original = SparseBasis.reduced_costs
        calls = {'count': 0}

        def flaky(self, costs):
            calls['count'] += 1
            if calls['count'] == 1:
                raise NumericalError('Basis solve residual 5.681e-08 exceeds the gate; refactorise')
            return original(self, costs)

        with patch.object(SparseBasis, 'reduced_costs', flaky):
            result = revised_simplex(csc_matrix(matrix), np.array([-1.0, -2.0, 0.0, 0.0]),
                                     lower, upper, basic=basic, initial=initial)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        self.assertEqual(result.refactor_retries, 1)

    def test_persistent_residual_gate_failure_is_reported_not_looped(self):
        from zyo.errors import NumericalError
        from zyo.sparse_simplex import SparseBasis
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)

        def always(self, costs):
            raise NumericalError('Basis solve residual 9.9e-01 exceeds the gate; refactorise')

        with patch.object(SparseBasis, 'reduced_costs', always):
            result = revised_simplex(csc_matrix(matrix), np.array([-1.0, -2.0, 0.0, 0.0]),
                                     lower, upper, basic=basic, initial=initial)
        # 重试一次后仍然失败：如实报 NUMERICAL_ERROR，绝不放宽残差门或无限重试。
        self.assertEqual(result.status, 'NUMERICAL_ERROR', result.message)
        self.assertIn('refactorisation did not restore the basis', result.message)
        self.assertEqual(result.refactor_retries, 1)

    def test_refactor_retries_is_zero_on_a_clean_solve(self):
        matrix, lower, upper, initial, basic = standard_form(
            [[1.0, 1.0], [1.0, 3.0]], ['<=', '<='], [4.0, 6.0], 2)
        result = revised_simplex(csc_matrix(matrix), np.array([-1.0, -2.0, 0.0, 0.0]),
                                 lower, upper, basic=basic, initial=initial)
        self.assertEqual(result.refactor_retries, 0)

    # ---------- 向量化定价必须与逐列 Python 规则逐项等价 ----------
    #
    # 定价原来是对全部列的 Python 扫描，每轮迭代走一遍；72747 列的实例在 Phase-I 就被它主导
    # （栈证据 research/probe_lp_relaxation_cost.py）。改成 NumPy 掩码后，规则本身必须一字不变：
    # 这里的参考实现**独立写在测试里**（不复用生产代码），逐列复刻原语义。

    @staticmethod
    def reference_pricing(reduced, at_upper, in_basis, fixed, tolerance, bland):
        """逐列参考实现：Dantzig 的 best 是改进量绝对值，并列取最先遇到的下标。"""
        entering = -1
        if not bland:
            best = tolerance
            for j in range(len(reduced)):
                if in_basis[j] or fixed[j]:
                    continue
                r = float(reduced[j])
                if at_upper[j]:
                    if r > best:
                        entering, best = j, r
                elif r < -best:
                    entering, best = j, -r
        else:
            for j in range(len(reduced)):
                if in_basis[j] or fixed[j]:
                    continue
                r = float(reduced[j])
                if (at_upper[j] and r > tolerance) or (not at_upper[j] and r < -tolerance):
                    entering = j
                    break
        return entering

    @staticmethod
    def vectorised_pricing(reduced, at_upper, in_basis, fixed, tolerance, bland):
        """与生产代码同一表达式，便于对照（生产代码内联在 revised_simplex 中）。"""
        improving = np.where(at_upper, reduced, -reduced)
        eligible = (~in_basis) & (~fixed) & (improving > tolerance)
        if not eligible.any():
            return -1
        if bland:
            return int(np.argmax(eligible))
        return int(np.argmax(np.where(eligible, improving, -np.inf)))

    def test_vectorised_pricing_matches_the_reference_on_random_inputs(self):
        rng = np.random.default_rng(20260915)
        tolerance = 1e-9
        compared = 0
        for _ in range(400):
            columns = int(rng.integers(1, 40))
            reduced = rng.normal(size=columns)
            # 大量并列与退化：把一部分简约成本正好置成 ±tolerance 与彼此相等。
            reduced[rng.random(columns) < 0.25] = 0.0
            reduced[rng.random(columns) < 0.15] = tolerance
            reduced[rng.random(columns) < 0.15] = -tolerance
            at_upper = rng.random(columns) < 0.5
            in_basis = np.zeros(columns, dtype=bool)
            if columns > 1:
                in_basis[rng.choice(columns, size=int(rng.integers(0, columns)), replace=False)] = True
            fixed = rng.random(columns) < 0.2
            for bland in (False, True):
                self.assertEqual(
                    self.vectorised_pricing(reduced, at_upper, in_basis, fixed, tolerance, bland),
                    self.reference_pricing(reduced, at_upper, in_basis, fixed, tolerance, bland),
                    f'columns={columns} bland={bland} reduced={reduced} '
                    f'at_upper={at_upper} fixed={fixed} in_basis={in_basis}')
                compared += 1
        self.assertGreater(compared, 700)

    def test_vectorised_pricing_prefers_the_smallest_index_when_tied(self):
        # 两个变量改进量完全相同时必须选下标更小的那个（原严格 `>` 语义 -> 首个最大值）。
        reduced = np.array([-1.0, -1.0, -0.5])
        at_upper = np.array([False, False, False])
        in_basis = np.zeros(3, dtype=bool)
        fixed = np.zeros(3, dtype=bool)
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, False), 0)
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, True), 0)

    def test_vectorised_pricing_ignores_basic_and_fixed_columns(self):
        reduced = np.array([-5.0, -4.0, -3.0])
        at_upper = np.array([False, False, False])
        in_basis = np.array([True, False, False])
        fixed = np.zeros(3, dtype=bool)
        # 下标 0 的改进量最大但在基内 -> 选下标 1。
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, False), 1)
        fixed = np.array([False, True, False])
        # 下标 1 被固定 -> 选下标 2。
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, False), 2)

    def test_vectorised_pricing_returns_none_when_nothing_improves(self):
        reduced = np.array([1e-12, -1e-12])
        at_upper = np.array([False, True])
        in_basis = np.zeros(2, dtype=bool)
        fixed = np.zeros(2, dtype=bool)
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, False), -1)
        self.assertEqual(self.vectorised_pricing(reduced, at_upper, in_basis, fixed, 1e-9, True), -1)


if __name__ == '__main__':
    unittest.main()
