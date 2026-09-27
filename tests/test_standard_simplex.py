# 标准形单纯形独立验收：每步枢轴都可手算核对；不依赖外部优化引擎。
import unittest

import numpy as np

from zyo.standard_simplex import standard_simplex


class StandardSimplexTests(unittest.TestCase):
    def test_hand_computed_two_variable_lp(self):
        # min -x-2y ; x+y<=4 ; x+3y<=6 ; x,y>=0 -> 最优 (3,1)，目标 -5。
        # 标准形 A=[A0|I]，b=(4,6)，初始基为两个松弛变量。
        matrix = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 3.0, 0.0, 1.0]])
        result = standard_simplex(matrix, np.array([-1.0, -2.0, 0.0, 0.0]),
                                  [2, 3], np.array([4.0, 6.0]))
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        np.testing.assert_allclose(result.values[:2], [3.0, 1.0], atol=1e-7)
        # 两次枢轴：x 入基（步长 4），随后 y 入基（步长 1），与手算一致。
        self.assertEqual(len(result.pivots), 2)
        self.assertAlmostEqual(result.pivots[0]['step'], 4.0, places=8)
        self.assertAlmostEqual(result.pivots[1]['step'], 1.0, places=8)

    def test_maximization_via_negated_objective(self):
        # max 3x+y ; x+y<=4 ; x<=3 -> min -3x-y = -10，最优 (3,1)。
        matrix = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
        result = standard_simplex(matrix, np.array([-3.0, -1.0, 0.0, 0.0]),
                                  [2, 3], np.array([4.0, 3.0]))
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, -10.0, places=8)

    def test_optimum_at_a_bound_is_reported(self):
        # min x ; x + s = 0.5 ; x,s >= 0 -> 最优 x=0，目标 0（不用松弛入基）。
        matrix = np.array([[1.0, 1.0]])
        result = standard_simplex(matrix, np.array([1.0, 0.0]), [1], np.array([0.5]))
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, 0.0, places=9)

    def test_unbounded_problem_is_reported_not_truncated(self):
        # min -x ; x - s = 0 ; x,s >= 0 -> 目标无下界。
        matrix = np.array([[1.0, -1.0]])
        result = standard_simplex(matrix, np.array([-1.0, 0.0]), [1], np.array([0.0]))
        self.assertEqual(result.status, 'UNBOUNDED', result.message)

    def test_infeasible_starting_basis_is_rejected(self):
        # x + s = -1 时 x=s=0 不可行，且不存在可行基。
        matrix = np.array([[1.0, 1.0]])
        result = standard_simplex(matrix, np.array([0.0, 0.0]), [0], np.array([-1.0]))
        self.assertEqual(result.status, 'INFEASIBLE', result.message)

    def test_singular_basis_is_reported(self):
        matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
        result = standard_simplex(matrix, np.array([1.0, 1.0]), [0, 1], np.array([1.0, 2.0]))
        self.assertEqual(result.status, 'NUMERICAL_ERROR', result.message)

    def test_wrong_sizes_are_rejected(self):
        matrix = np.array([[1.0, 1.0]])
        # 成本向量长度与列数不符 -> 参数错误，不是"不可行"。
        self.assertEqual(standard_simplex(matrix, np.array([1.0]), [0],
                                          np.array([1.0])).status, 'NUMERICAL_ERROR')
        # 基大小与行数不符 -> 无法构成基。
        self.assertEqual(standard_simplex(matrix, np.array([1.0, 1.0]), [0, 1],
                                          np.array([1.0, 2.0])).status, 'INFEASIBLE')

    def test_solution_satisfies_the_constraints_and_complementarity(self):
        # 随机小规模题：逐条独立核验 Ax=b、x>=0、目标复算与基变量简约成本为零。
        rng = np.random.default_rng(20260911)
        for trial in range(5):
            rows, columns = 3, 6
            body = rng.normal(size=(rows, columns-rows))
            matrix = np.hstack([body, np.eye(rows)])
            cost = rng.normal(size=columns)
            rhs = rng.uniform(1.0, 5.0, size=rows)
            basic = list(range(columns-rows, columns))
            result = standard_simplex(matrix, cost, basic, rhs)
            if result.status not in ('OPTIMAL', 'UNBOUNDED', 'INFEASIBLE'):
                self.fail(f'trial {trial}: unexpected status {result.status}')
            if result.status != 'OPTIMAL':
                continue
            self.assertLessEqual(np.max(np.abs(matrix @ result.values - rhs)), 1e-7)
            self.assertGreaterEqual(np.min(result.values), -1e-7)
            self.assertAlmostEqual(result.objective, float(cost @ result.values), places=8)


if __name__ == '__main__':
    unittest.main()
