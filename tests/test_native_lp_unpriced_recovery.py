"""微小未定价项只可由独立原模型证书恢复最优声明。"""

import math
import unittest
from unittest.mock import patch

from scipy.sparse import csc_matrix

from zyo.native_lp import solve_certified_lp
import zyo.native_lp as native_lp


class NativeLPUnpricedRecoveryTests(unittest.TestCase):
    @staticmethod
    def _auxiliary_data():
        rows = csc_matrix([[1.0, 0.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [1.0, -2.0, 1.0],
                           [-2.0, 1.0, 1.0]])
        return (rows, [0.0, 0.0, 1.0], [-math.inf, -math.inf, 0.0],
                [math.inf]*3, [1.0, 1.0, 0.0, 0.0])

    def test_four_row_auxiliary_has_independent_half_unit_certificate(self):
        # 手算 max v：四个不等式在 (π0,π1,v)=(1/2,1/2,1/2) 均取等；
        # 非负行权重 (1/2,0,1/6,1/3) 消去 π，权重和为1，给上界 v<=1/2。
        result = solve_certified_lp(
            *self._auxiliary_data(), sense=['<=']*4,
            maximize=True, iteration_limit=100)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertTrue(result.optimal)
        self.assertAlmostEqual(result.objective, 0.5, places=12)
        self.assertTrue(result.certificate.verified)
        self.assertEqual(result.record['solver_status'], 'NUMERICAL_ERROR')
        self.assertTrue(result.record['posthoc_unpriced_basis_certificate']['accepted'])

    def test_explicit_unpriced_marker_requires_full_original_certificate(self):
        # 分层回归：先独立核对公共入口的后验证书，不依赖底层标记实现时机。
        actual_solve = native_lp.solve_lp

        def marked_stop(*args, **kwargs):
            outcome = actual_solve(*args, **kwargs)
            self.assertEqual(outcome.status, 'NUMERICAL_ERROR')
            outcome.unpriced_optimality = True
            return outcome

        with patch.object(native_lp, 'solve_lp', side_effect=marked_stop):
            result = solve_certified_lp(
                *self._auxiliary_data(), sense=['<=']*4,
                maximize=True, iteration_limit=100)
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertTrue(result.certificate.verified)
        self.assertEqual(result.record['solver_status'], 'NUMERICAL_ERROR')

    def test_unrelated_numerical_stop_is_not_promoted_from_same_candidate(self):
        # 故障注入只撤销停止原因标记；即使当前点另有最优见证，也不能
        # 对任意 NUMERICAL_ERROR 事后改写实际求解器状态。
        actual_solve = native_lp.solve_lp

        def unrelated_stop(*args, **kwargs):
            outcome = actual_solve(*args, **kwargs)
            outcome.unpriced_optimality = False
            return outcome

        with patch.object(native_lp, 'solve_lp', side_effect=unrelated_stop):
            result = solve_certified_lp(
                *self._auxiliary_data(), sense=['<=']*4,
                maximize=True, iteration_limit=100)
        self.assertEqual(result.record['solver_status'], 'NUMERICAL_ERROR')
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertFalse(result.optimal)
        self.assertNotIn('posthoc_unpriced_basis_certificate', result.record)

    def test_bounded_tiny_improvement_is_not_recovered_as_optimal(self):
        # 真最优是 -100；当前起点 x1=0 的基不能凭定价阈值取得最优证书。
        result = solve_certified_lp(csc_matrix([[1.0, 0.0]]), [0.0, -1e-10],
                                    [0.0, 0.0], [0.0, 1e12], [0.0],
                                    sense=['=='], iteration_limit=1,
                                    warm_basis=[0], warm_nonbasic_at_upper=[],
                                    require_warm_start=True)
        self.assertFalse(result.optimal)
        self.assertNotEqual(result.status, 'OPTIMAL')

    def test_unbounded_tiny_improvement_is_not_recovered_as_optimal(self):
        result = solve_certified_lp(csc_matrix([[1.0, 0.0]]), [0.0, -1e-10],
                                    [0.0, 0.0], [0.0, math.inf], [0.0],
                                    sense=['=='], iteration_limit=1,
                                    warm_basis=[0], warm_nonbasic_at_upper=[],
                                    require_warm_start=True)
        self.assertFalse(result.optimal)
        self.assertNotEqual(result.status, 'OPTIMAL')


if __name__ == '__main__':
    unittest.main()
