# ZYO-051 独立验收：基复用必须"只在真的有效时才说可复用"，并如实报告失效原因。
# 预期由稠密线性代数或手算给出，不依赖被测模块的判定结果。
import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from zyo.sparse_simplex import SparseBasis
from zyo.warm_start import evaluate_reuse


def build(rows):
    return csc_matrix(np.array(rows, dtype=float))


# 3 行 5 列：前 3 列为单位列（逻辑变量），后 2 列为结构列。
MATRIX = [
    [1.0, 0.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 1.0, -1.0, 1.0],
]


class WarmStartReuseTests(unittest.TestCase):
    def setUp(self):
        self.matrix = build(MATRIX)
        self.basic = [0, 1, 2]
        self.costs = np.array([0.0, 0.0, 0.0, -1.0, -2.0])

    # ---------- 接受真正可复用的基 ----------

    def test_accepts_a_feasible_basis_for_new_bounds(self):
        # 逻辑列停在 [0,4] 内，基解 x_B = 0（非基变量都取 0）-> 可行。
        lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        upper = np.array([4.0, 5.0, 6.0, 10.0, 10.0])
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        self.assertTrue(decision.reusable, decision.reason)
        self.assertTrue(decision.basis_valid)
        self.assertTrue(decision.primal_feasible)
        self.assertLessEqual(decision.max_equation_residual, 1e-9)
        np.testing.assert_allclose(decision.basic_values, [0.0, 0.0, 0.0], atol=1e-12)

    def test_accepts_a_basis_with_a_nonzero_basic_solution(self):
        # 非基结构列取到非零上界，基解相应变化；这里选一组让基解确实非零且可行的界。
        # 非基列取上界 x4=1, x5=1 -> x_B = -(列4 + 列5) = (-3, -4, 0)；
        # 因此逻辑列必须有足够的负下界才可行，这就是"界变化决定基是否还能复用"的实测例子。
        lower = np.array([-5.0, -5.0, -5.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0, 1.0, 1.0])
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs,
                                  residual_tolerance=1e-9, feasibility_tolerance=1e-7)
        # 非基变量停在**有界的一侧**（此处下界 0），故基解为 0，仍然可行。
        self.assertTrue(decision.reusable, decision.reason)
        np.testing.assert_allclose(decision.basic_values, [0.0, 0.0, 0.0], atol=1e-10)

    def test_rejects_when_nonzero_nonbasic_bounds_make_the_basis_infeasible(self):
        # 同一组界，但把非基变量强制停在上界 1：x_B = (-3,-4,0)，第一个逻辑列下界
        # 若为 0 则违反 -> 必须判为不可复用（这正是界收紧后热启动失效的直接例子）。
        lower = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        upper = np.array([10.0, 10.0, 10.0, 1.0, 1.0])
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        self.assertFalse(decision.reusable)
        self.assertTrue(decision.basis_valid)
        self.assertFalse(decision.primal_feasible)
        self.assertGreater(decision.max_primal_violation, 0.5)

    # ---------- 拒绝失效的基并给出原因 ----------

    def test_rejects_when_the_new_bounds_make_the_basis_infeasible(self):
        # 收紧第一个逻辑列的上界到 -1，与基解 0 冲突 -> 不可复用。
        lower = np.array([-5.0, 0.0, 0.0, 0.0, 0.0])
        upper = np.array([-1.0, 5.0, 6.0, 10.0, 10.0])
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        self.assertFalse(decision.reusable)
        self.assertTrue(decision.basis_valid)
        self.assertFalse(decision.primal_feasible)
        self.assertGreater(decision.max_primal_violation, 0.5)
        self.assertIn('not primal feasible', decision.reason)

    def test_rejects_a_singular_basis(self):
        # 用两列相同的列构造奇异基。
        matrix = build([[1.0, 1.0], [1.0, 1.0]])
        decision = evaluate_reuse(matrix, [0, 1], np.zeros(2), np.ones(2), np.zeros(2))
        self.assertFalse(decision.reusable)
        self.assertFalse(decision.basis_valid)
        self.assertIn('singular', decision.reason)

    def test_rejects_a_basis_of_the_wrong_size(self):
        decision = evaluate_reuse(self.matrix, [0, 1], np.zeros(5), np.ones(5), self.costs)
        self.assertFalse(decision.reusable)
        self.assertIn('rows', decision.reason)

    def test_rejects_duplicate_columns(self):
        decision = evaluate_reuse(self.matrix, [0, 0, 1], np.zeros(5), np.ones(5), self.costs)
        self.assertFalse(decision.reusable)
        self.assertIn('duplicate', decision.reason)

    def test_reports_a_free_nonbasic_variable_instead_of_guessing(self):
        # 第 4 列两侧都无界且非基：基解不确定，必须报告而不是猜一个值。
        lower = np.array([0.0, 0.0, 0.0, -math.inf, 0.0])
        upper = np.array([5.0, 5.0, 5.0, math.inf, 5.0])
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        self.assertFalse(decision.reusable)
        self.assertIn('free', decision.reason)

    # ---------- 与直接建基的一致性 ----------

    def test_reported_values_match_a_directly_built_basis(self):
        lower = np.array([0.0, 0.0, 0.0, 0.5, 0.25])
        upper = np.array([10.0] * 5)
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        direction = np.zeros(5)
        direction[3], direction[4] = 0.5, 0.25
        basis = SparseBasis(self.matrix, self.basic)
        expected = basis.basic_solution({3: 0.5, 4: 0.25})
        np.testing.assert_allclose(decision.basic_values, expected, atol=1e-10)
        residual = np.max(np.abs(np.array(MATRIX) @ (
            np.array([*expected, 0.5, 0.25]) if False else
            np.array([expected[0], expected[1], expected[2], 0.5, 0.25]))))
        self.assertLessEqual(residual, 1e-10)

    def test_objective_is_recomputed_from_the_costs(self):
        lower = np.zeros(5)
        upper = np.array([10.0] * 5)
        decision = evaluate_reuse(self.matrix, self.basic, lower, upper, self.costs)
        expected = float(self.costs @ np.array([*decision.basic_values, 0.0, 0.0]))
        self.assertAlmostEqual(decision.objective, expected, places=10)

    def test_checks_dictionary_is_complete_for_audit(self):
        decision = evaluate_reuse(self.matrix, self.basic, np.zeros(5), np.ones(5), self.costs)
        for key in ('basis_valid', 'equation_residual_ok', 'bounds_ok'):
            self.assertIn(key, decision.checks)


if __name__ == '__main__':
    unittest.main()
