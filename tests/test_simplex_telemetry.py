# 手算统计不等同于优化证明：记录全部成功LU工作、共享动作额度及各阶段真实退出。
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from zyo import Model
from zyo.sparse_simplex import SparseBasis, revised_simplex, solve_lp


class SimplexTelemetryTests(unittest.TestCase):
    def solve_counted(self, matrix, costs, lower, upper, **kwargs):
        # 只读包裹实际分解，成功后计数，不替换LU、不反馈给内核。
        calls = []
        original = SparseBasis.refactorise

        def observed(basis):
            result = original(basis)
            calls.append(tuple(basis.basic))
            return result

        with patch.object(SparseBasis, 'refactorise', observed):
            result = solve_lp(csc_matrix(matrix), costs, lower, upper, **kwargs)
        return result, calls

    def test_one_row_cleanup_counts_all_three_factorisations(self):
        result, calls = self.solve_counted([[2.]], [1.], [0.], [0.], iteration_limit=1)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(len(calls), 3)  # Phase-I、人工驱除、Phase-II各一次。
        self.assertEqual(result.refactorisations, 3)
        self.assertEqual(result.phase_one['cleanup_refactorisations'], 1)
        self.assertEqual(result.iterations, 1)

    def test_cleanup_limit_exit_counts_work_without_extra_movement(self):
        result, calls = self.solve_counted([[2.]], [1.], [0.], [0.], iteration_limit=0)
        self.assertEqual(result.status, 'ITERATION_LIMIT')
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.refactorisations, 2)
        self.assertEqual(result.iteration_budget['used'], 0)

    def test_two_cleanup_pivots_count_four_factorisations(self):
        result, calls = self.solve_counted([[2., 0.], [0., 3.]], [1., 2.], [0., 0.], [0., 0.],
                                           iteration_limit=2)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.refactorisations, 4)
        self.assertEqual(result.phase_one['cleanup_refactorisations'], 2)
        self.assertEqual(result.iterations, 2)

    def test_phase_one_infeasible_keeps_shared_budget_record(self):
        # 固定x=0、s>=2、2x-s=0不可行；Phase-I从正人工活动即可确认。
        result = solve_lp(csc_matrix([[2., -1.]]), [1., 0.], [0., 2.], [0., np.inf],
                          iteration_limit=20)
        self.assertEqual(result.status, 'INFEASIBLE')
        self.assertEqual(result.iteration_budget,
                         dict(limit=20, used=0, phase_one=0, phase_two=0,
                              unit='executed pivot or bound flip'))

    def test_rejected_warm_and_logical_starts_remain_in_total_work(self):
        # 两个起点都把s置于上界4，导致x=4越过上界3；冷Phase-I可行。
        result, calls = self.solve_counted([[1., -1.]], [-1., 0.], [0., 2.], [3., 4.],
                                           basic=[0], iteration_limit=10)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertAlmostEqual(result.objective, -3.)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.refactorisations, 4)
        self.assertEqual(result.iteration_budget['phase_two'], 1)
        self.assertEqual(result.iteration_budget['phase_one'], 0)

    def test_direct_early_exits_keep_zero_budget_use_and_real_work(self):
        cases = [
            # 界矛盾在分解前拒绝；自由非基/界内非基在分解后拒绝。
            ([[1., -1.]], [1., 0.], [2., 0.], [1., np.inf], [1], None, 0),
            ([[1., -1.]], [1., 0.], [-np.inf, 0.], [np.inf, np.inf], [1], None, 1),
            ([[1., -1.]], [1., 0.], [0., 0.], [2., np.inf], [1], {0: 1.}, 1),
        ]
        for matrix, costs, lower, upper, basis, initial, factors in cases:
            with self.subTest(initial=initial, lower=lower):
                result = revised_simplex(csc_matrix(matrix), costs, lower, upper,
                                         basic=basis, initial=initial, iteration_limit=3)
                self.assertEqual(result.iteration_budget.get('limit'), 3)
                self.assertEqual(result.iteration_budget.get('used'), 0)
                self.assertEqual(result.refactorisations, factors)

    def test_public_logical_path_has_phase_breakdown_without_phase_one(self):
        model = Model('budget-record')
        x = model.add_var('x', ub=1.)
        model.add_constr(x <= 10.)
        model.minimize(-x)
        result = model.solve('native_simplex', iteration_limit=1)
        self.assertEqual(result.status.value, 'OPTIMAL')
        self.assertEqual(result.metadata['iteration_budget'],
                         dict(limit=1, used=1, phase_one=0, phase_two=1,
                              unit='executed pivot or bound flip'))


if __name__ == '__main__':
    unittest.main()
