# 稠密参考实现的独立验收：符号约定必须与方程 A x = rhs 一致，并与稀疏驱动逐枢轴一致。
# 参考实现的价值全在"可独立核对"，所以这里既查手算题，也查两个实现的一致性。
import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from zyo.dense_reference import dense_simplex
from zyo.sparse_simplex import revised_simplex, solve_lp


def homogeneous(rows, rhs, upper_variables):
    """行 `sum(a x) <= b` 的齐次形式：逻辑变量 s_i = sum(a_i x) ∈ (-inf, b_i]，列 -e_i。

    ``upper_variables`` 给原始变量一个有限上界，使问题有界（否则随机目标下
    `{x >= 0, A x <= b}` 的回收锥通常让目标无界）。
    """
    body = np.array(rows, dtype=float)
    m, n = body.shape
    b = np.array(rhs, dtype=float)
    matrix = np.hstack([body, -np.eye(m)])
    lower = np.concatenate([np.zeros(n), np.full(m, -math.inf)])
    upper = np.concatenate([np.asarray(upper_variables, dtype=float), b])
    basic = [n+i for i in range(m)]
    initial = {j: 0.0 for j in range(n+m)}
    cost = np.concatenate([np.zeros(n), np.zeros(m)])
    return matrix, lower, upper, basic, initial, cost


class DenseReferenceTests(unittest.TestCase):
    def setUp(self):
        # min -x-2y ; x+y<=4 ; x+3y<=6 ; x,y>=0 -> (3,1)，目标 -5。
        self.body = [[1.0, 1.0], [1.0, 3.0]]
        self.rhs = [4.0, 6.0]
        self.matrix, self.lower, self.upper, self.basic, self.initial, _ = homogeneous(
            self.body, self.rhs, [math.inf, math.inf])
        self.cost = np.array([-1.0, -2.0, 0.0, 0.0])

    def test_hand_computed_optimum_and_equation_residual(self):
        result = dense_simplex(self.matrix, self.cost, self.lower, self.upper,
                               basic=self.basic, initial=self.initial)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=7)
        np.testing.assert_allclose(result.values, [3.0, 1.0, 4.0, 6.0], atol=1e-7)
        # 参考实现的每一步都必须满足 A x = 0；这正是早期版本符号写反后失守的不变量。
        self.assertLessEqual(float(np.max(np.abs(self.matrix @ result.values))), 1e-9)

    def test_pivot_history_matches_the_hand_calculation(self):
        result = dense_simplex(self.matrix, self.cost, self.lower, self.upper,
                               basic=self.basic, initial=self.initial)
        # 手算：先 x 入基走 4（s0=x+y 到上界 4），再 y 入基走 1（s1=x+3y 到上界 6）。
        steps = [(entry['entering'], entry['step']) for entry in result.traffic]
        self.assertEqual(steps, [(0, 4.0), (1, 1.0)])

    def test_dense_and_sparse_agree(self):
        dense = dense_simplex(self.matrix, self.cost, self.lower, self.upper,
                             basic=self.basic, initial=self.initial)
        sparse = revised_simplex(self.matrix, self.cost, self.lower, self.upper,
                                 basic=self.basic, initial=self.initial)
        self.assertEqual(dense.status, sparse.status)
        self.assertAlmostEqual(dense.objective, sparse.objective, places=8)
        np.testing.assert_allclose(dense.values, sparse.values, atol=1e-8)
        self.assertEqual(set(dense.basic), set(sparse.basic))
        self.assertEqual(len(dense.traffic), len(sparse.history))

    def test_infeasible_start_is_reported_not_repaired(self):
        # x 的下界抬到 5，而 x+y<=4 使 x<=4：初始逻辑基不可行，必须如实上报。
        # 非基变量的初值也要跟着挪到新下界上，否则会被判为"非基变量落在界内"。
        lower = self.lower.copy()
        lower[0] = 5.0
        initial = dict(self.initial)
        initial[0] = 5.0
        result = dense_simplex(self.matrix, self.cost, lower, self.upper,
                               basic=self.basic, initial=initial)
        self.assertEqual(result.status, 'START_INFEASIBLE', result.message)
        self.assertGreater(result.max_primal_violation, 1e-7)

    def test_nonbasic_variable_inside_its_bounds_is_rejected(self):
        # 非基变量必须精确停在某一侧界上；落在界内说明调用方的初值与基不匹配。
        result = dense_simplex(self.matrix, self.cost, self.lower, self.upper,
                               basic=self.basic, initial={0: 2.0, 1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(result.status, 'NUMERICAL_ERROR', result.message)

    def test_random_bounded_instances_agree_with_the_sparse_driver(self):
        rng = np.random.default_rng(20260913)
        for trial in range(8):
            m, n = 3, 4
            rows = rng.normal(size=(m, n))
            rhs = rng.uniform(2.0, 6.0, size=m)
            widths = rng.uniform(2.0, 6.0, size=n)
            matrix, lower, upper, basic, initial, _ = homogeneous(rows, rhs, widths)
            cost = np.concatenate([rng.normal(size=n), np.zeros(m)])
            dense = dense_simplex(matrix, cost, lower, upper, basic=basic, initial=initial)
            sparse = solve_lp(csc_matrix(matrix), cost, lower, upper)
            self.assertEqual(dense.status, 'OPTIMAL', f'trial {trial}: {dense.message}')
            self.assertEqual(sparse.status, 'OPTIMAL', f'trial {trial}: {sparse.message}')
            self.assertAlmostEqual(dense.objective, sparse.objective, places=6)
            self.assertLessEqual(float(np.max(np.abs(matrix @ sparse.values))), 1e-7)
            self.assertLessEqual(sparse.max_primal_violation, 1e-7)
            for index in range(cost.size):
                if math.isfinite(lower[index]):
                    self.assertGreaterEqual(sparse.values[index], lower[index]-1e-7)
                if math.isfinite(upper[index]):
                    self.assertLessEqual(sparse.values[index], upper[index]+1e-7)


if __name__ == '__main__':
    unittest.main()
