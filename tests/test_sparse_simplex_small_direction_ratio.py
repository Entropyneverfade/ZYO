"""比例检验不能用简约成本阈值漏掉会放大为显著界移动的小方向。"""

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from zyo.sparse_simplex import (_ratio_limit_scalar, _ratio_limit_vectorized,
                                revised_simplex, NumericalError)


class SmallDirectionRatioTests(unittest.TestCase):
    def setUp(self):
        # 手算：x0 + 10^-12*x1=0，x0 从 0 降至 -0.5 时 x1 只能升至 5*10^11。
        self.basic = [0]
        self.moving = np.array([1e-12])
        self.values = np.array([0.0, 0.0])
        self.lower = np.array([-0.5, 0.0])
        self.upper = np.array([0.5, 1e12])

    def test_scalar_ratio_keeps_small_direction_with_large_physical_move(self):
        # 会抓住 d<-pricing_tolerance 才参加比例检验这一错误条件。
        step, leaving, at_upper = _ratio_limit_scalar(
            self.basic, self.moving, self.values, self.lower, self.upper, 1e-9)
        self.assertEqual(step, 5e11)
        self.assertEqual(leaving, 0)
        self.assertFalse(at_upper)

    def test_vector_ratio_keeps_small_direction_with_large_physical_move(self):
        # 向量快路径必须与手算的严格最小可行比例一致。
        step, leaving, at_upper = _ratio_limit_vectorized(
            self.basic, self.moving, self.values, self.lower, self.upper, 1e-9)
        self.assertEqual(step, 5e11)
        self.assertEqual(leaving, 0)
        self.assertFalse(at_upper)

    def test_native_action_cannot_flip_past_basic_lower_bound(self):
        # 原错误把 x1 翻到 10^12，使 x0=-1，物理界越出 0.5。
        result = revised_simplex(csc_matrix([[1.0, 1e-12]]), [0.0, -1.0],
                                 self.lower, self.upper, basic=self.basic,
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0]['kind'], 'pivot')
        self.assertTrue(math.isclose(result.history[0]['step'], 5e11,
                                      rel_tol=1e-15, abs_tol=0.0))
        self.assertLessEqual(result.max_primal_violation, 1e-7)
        self.assertGreaterEqual(result.values[0], -0.5-1e-7)

    def test_small_finite_blocker_prevents_false_unbounded_status(self):
        # x1 虽无显式上界，等式与 x0>=-0.5 仍给出 x1<=5*10^11。
        upper = np.array([0.5, math.inf])
        result = revised_simplex(csc_matrix([[1.0, 1e-12]]), [0.0, -1.0],
                                 self.lower, upper, basic=self.basic,
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertNotEqual(result.status, 'UNBOUNDED')
        self.assertEqual(result.history[0]['kind'], 'pivot')
        self.assertTrue(math.isclose(result.history[0]['step'], 5e11,
                                      rel_tol=1e-15, abs_tol=0.0))
        self.assertLessEqual(result.max_primal_violation, 1e-7)

    def test_nonzero_blocker_with_unrepresentable_ratio_is_not_infinite_ray(self):
        # 0.5/10^-310 = 5*10^309：数学上有限，binary64 计算会溢出为 inf。
        moving = np.array([1e-310])
        upper = np.array([0.5, math.inf])
        for ratio in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(ratio=ratio.__name__):
                with self.assertRaises(NumericalError):
                    ratio(self.basic, moving, self.values, self.lower, upper, 1e-9)
        result = revised_simplex(csc_matrix([[1.0, 1e-310]]), [0.0, -1.0],
                                 self.lower, upper, basic=self.basic,
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.history, [])
        self.assertLessEqual(result.max_primal_violation, 1e-7)

    def test_true_zero_direction_remains_unbounded(self):
        # A=[1,0] 时 x1 对受界基本量完全无影响，确实是改善射线。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1.0],
                                 self.lower, [0.5, math.inf], basic=self.basic,
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'UNBOUNDED')

    def test_entry_bound_before_small_basic_blocker_still_flips(self):
        # x1 上界 10^10 早于基本界对应的 5*10^11；真实 x0=-0.01。
        result = revised_simplex(csc_matrix([[1.0, 1e-12]]), [0.0, -1.0],
                                 self.lower, [0.5, 1e10], basic=self.basic,
                                 initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.history[0]['kind'], 'bound_flip')
        self.assertEqual(result.history[0]['step'], 1e10)
        self.assertTrue(math.isclose(result.values[0], -0.01, abs_tol=1e-15))
        self.assertLessEqual(result.max_primal_violation, 1e-7)

    def test_small_negative_direction_hits_basic_upper_bound(self):
        # x0-10^-12*x1=0，x1 上升使 x0 向上界0.5移动。
        step, leaving, at_upper = _ratio_limit_vectorized(
            self.basic, np.array([-1e-12]), self.values,
            self.lower, np.array([0.5, math.inf]), 1e-9)
        self.assertEqual(step, 5e11)
        self.assertEqual(leaving, 0)
        self.assertTrue(at_upper)

    def test_finite_subtraction_overflow_keeps_earliest_ratio(self):
        # 真比值2<另一行3；比例必须正确，后续不可表示的状态更新另行拒绝。
        basic = [0, 1]
        moving = np.array([-1e308, -1.0])
        values = np.array([-1e308, 0.0])
        lower = np.array([-1e308, 0.0])
        upper = np.array([1e308, 3.0])
        for ratio in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(ratio=ratio.__name__):
                self.assertEqual(ratio(basic, moving, values, lower, upper, 1e-9),
                                 (2.0, 0, True))

    def test_unrepresentable_alternative_ratio_keeps_ordinary_status(self):
        # 备选列的有限阻挡比>binary64上界；应拒绝该提议，不把异常抛出驱动器。
        matrix = csc_matrix([[1.0, -1.0, 1e-310, -1.0]])
        options = dict(basic=[0], initial={1: 0.0, 2: 0.0, 3: 0.5},
                       iteration_limit=1, positive_step_pricing=True,
                       positive_step_scan_after=0, positive_step_scan_every=1)
        result = revised_simplex(matrix, [0.0, -2.0, -1.0, 0.0],
                                 [-0.5, 0.0, 0.0, 0.5],
                                 [0.5, math.inf, 1e10, 0.5], **options)
        self.assertNotEqual(result.status, 'UNBOUNDED')
        self.assertEqual(result.history[0]['entering'], 1)
        self.assertLessEqual(result.max_primal_violation, 1e-7)

    def test_driver_rejects_physically_invalid_proposed_move_before_commit(self):
        # 注入一次错误比例以验证驱动边界：终点复核不能代替动作前防线。
        with patch('zyo.sparse_simplex._ratio_limit_vectorized',
                   return_value=(math.inf, -1, False)):
            result = revised_simplex(csc_matrix([[1.0, 1e-12]]), [0.0, -1.0],
                                     self.lower, self.upper, basic=self.basic,
                                     initial={1: 0.0}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.history, [])
        self.assertEqual(result.values[0], 0.0)
        self.assertEqual(result.values[1], 0.0)

    def test_finite_entering_flip_span_overflow_cannot_prove_unbounded(self):
        # 两侧界都有限，但 1e308-(-1e308) 在 binary64 上溢；并非无界射线。
        result = revised_simplex(csc_matrix([[1.0, 0.0]]), [0.0, -1.0],
                                 [0.0, -1e308], [0.0, 1e308], basic=[0],
                                 initial={1: -1e308}, iteration_limit=1)
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.history, [])
        self.assertEqual(result.values[1], -1e308)


if __name__ == '__main__':
    unittest.main()
