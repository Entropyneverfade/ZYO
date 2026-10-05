"""公共 LP 适配器保留事后认证来源、原目标与真实对偶界。"""

from fractions import Fraction
import unittest
from unittest.mock import patch

from zyo import Model
from zyo.status import Status


class NativeSimplexPosthocMetadataTests(unittest.TestCase):
    def _four_row_maximum(self, constant=0.0):
        # 手算最优值：四行在 p=q=v=1/2 取等；行权重可给出上界 1/2。
        model = Model('public_posthoc_four_row')
        p = model.add_var('p', lb=None, ub=None)
        q = model.add_var('q', lb=None, ub=None)
        v = model.add_var('v', lb=0, ub=None)
        model.add_constr(p + v <= 1)
        model.add_constr(q + v <= 1)
        model.add_constr(p - 2*q + v <= 0)
        model.add_constr(-2*p + q + v <= 0)
        model.maximize(v + constant)
        return model

    def test_public_recovery_keeps_underlying_numeric_stop_and_audited_bound(self):
        # 删除底层状态或把 incumbent 伪装为严格最优界，均须使本测试失败。
        result = self._four_row_maximum().solve('native_simplex', iteration_limit=100)
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, 0.5, places=12)
        self.assertEqual(result.metadata['raw_status'], 'NUMERICAL_ERROR')
        self.assertEqual(result.metadata['certified_status'], 'OPTIMAL')
        self.assertTrue(result.metadata['unpriced_optimality_stop'])
        self.assertTrue(result.metadata['posthoc_unpriced_basis_certificate']['accepted'])
        self.assertTrue(result.metadata['certificate']['verified'])
        self.assertFalse(result.metadata['fallback_used'])
        self.assertIsNotNone(result.best_bound)
        self.assertGreaterEqual(result.best_bound, result.objective)
        self.assertIsNone(result.mip_gap)
        self.assertIn('certificate', result.termination_reason)

    def test_public_objective_uses_original_binary64_certificate_not_rounded_products(self):
        # 两个固定变量的目标乘积先舍入会相消为零；原存储值的精确和并非零。
        first = 1151660734277.0
        second = -1421803362854.3792
        fixed = 1.23456789012345
        expected = float(Fraction.from_float(first)*Fraction.from_float(fixed)
                         + Fraction.from_float(second))
        self.assertNotEqual(expected, 0.0)
        model = Model('public_product_cancellation')
        x = model.add_var('x', lb=fixed, ub=fixed)
        y = model.add_var('y', lb=1.0, ub=1.0)
        model.minimize(first*x + second*y)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertEqual(result.objective, expected)
        self.assertEqual(result.metadata['raw_status'], 'OPTIMAL')
        self.assertFalse(result.metadata['unpriced_optimality_stop'])
        self.assertFalse(result.metadata['posthoc_unpriced_basis_certificate']['accepted'])
        self.assertLessEqual(result.best_bound, result.objective)
        self.assertIsNone(result.mip_gap)

    def test_public_recovered_maximum_preserves_original_objective_constant(self):
        # 原生内层只求线性部分；公开目标和对偶上界都必须加回常数。
        result = self._four_row_maximum(constant=7.0).solve(
            'native_simplex', iteration_limit=100)
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, 7.5, places=12)
        self.assertGreaterEqual(result.best_bound, result.objective)
        self.assertEqual(result.metadata['raw_status'], 'NUMERICAL_ERROR')

    def test_public_constant_cancellation_keeps_exact_original_objective_and_lower_bound(self):
        # 乘积先舍入为常数相反数时，先把证书目标转float再加常数会假报0。
        coefficient = 1151660734277.0
        fixed = 1.23456789012345
        constant = -1421803362854.3792
        exact = (Fraction.from_float(coefficient)*Fraction.from_float(fixed)
                 + Fraction.from_float(constant))
        self.assertNotEqual(exact, 0)
        model = Model('public_constant_cancellation')
        x = model.add_var('x', lb=fixed, ub=fixed)
        model.minimize(coefficient*x + constant)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertEqual(result.objective, float(exact))
        self.assertLessEqual(result.best_bound, float(exact))
        self.assertIsNone(result.mip_gap)

    def test_public_adapter_refuses_optimal_flag_without_verified_certificate(self):
        # 故障注入：即使下层误带 optimal 标志，证书拒绝时公开入口也必须拒绝升级。
        from zyo import native_lp

        actual_solve = native_lp.solve_certified_lp

        def rejected_certificate(*args, **kwargs):
            outcome = actual_solve(*args, **kwargs)
            self.assertTrue(outcome.optimal)
            outcome.certificate.verified = False
            outcome.certificate.reason = 'injected independent certificate refusal'
            return outcome

        with patch.object(native_lp, 'solve_certified_lp', side_effect=rejected_certificate):
            result = self._four_row_maximum().solve('native_simplex', iteration_limit=100)
        self.assertEqual(result.status, Status.NUMERICAL_ERROR)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.best_bound)
        self.assertFalse(result.metadata['certificate']['verified'])
        self.assertEqual(result.metadata['raw_status'], 'NUMERICAL_ERROR')

    def test_public_unpriced_stop_marker_is_independent_of_posthoc_attempt(self):
        # 标记来自底层停止记录；审计拒绝/未启动不能擦除真实停止原因。
        from zyo import native_lp

        actual_solve = native_lp.solve_certified_lp

        def unaudited_stop(*args, **kwargs):
            outcome = actual_solve(*args, **kwargs)
            outcome.status = 'NUMERICAL_ERROR'
            outcome.optimal = False
            outcome.record['unpriced_optimality_stop'] = True
            outcome.record['posthoc_unpriced_basis_certificate'] = dict(
                attempted=False, accepted=False, reason='injected audit refusal')
            return outcome

        with patch.object(native_lp, 'solve_certified_lp', side_effect=unaudited_stop):
            result = self._four_row_maximum().solve('native_simplex', iteration_limit=100)
        self.assertEqual(result.status, Status.NUMERICAL_ERROR)
        self.assertTrue(result.metadata['unpriced_optimality_stop'])
        self.assertFalse(result.metadata['posthoc_unpriced_basis_certificate']['attempted'])
        self.assertIsNone(result.best_bound)

    def test_public_positive_objective_overflow_returns_numerical_error(self):
        # 内层 8e307 可表出；公开常数后超过 binary64 最大值，不能返回 Inf。
        model = Model('public_positive_objective_overflow')
        x = model.add_var('x', lb=8e307, ub=8e307)
        model.maximize(x + 1.1e308)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.NUMERICAL_ERROR)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.best_bound)

    def test_public_negative_objective_overflow_returns_numerical_error(self):
        # 符号反转的同类反例，覆盖最小化原目标的负向溢出。
        model = Model('public_negative_objective_overflow')
        x = model.add_var('x', lb=8e307, ub=8e307)
        model.minimize(-x - 1.1e308)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.NUMERICAL_ERROR)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.best_bound)

    def test_public_bound_overflow_returns_numerical_error(self):
        # 故障注入对偶界极值：即使目标仍有限，界后处理也不能逃逸异常或报 Inf。
        from zyo import native_lp

        model = Model('public_bound_overflow')
        x = model.add_var('x', lb=0.0, ub=0.0)
        model.maximize(x + 1e308)
        actual_solve = native_lp.solve_certified_lp

        def extreme_dual(*args, **kwargs):
            outcome = actual_solve(*args, **kwargs)
            self.assertTrue(outcome.optimal)
            outcome.certificate.dual_objective = -1.1e308
            return outcome

        with patch.object(native_lp, 'solve_certified_lp', side_effect=extreme_dual):
            result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.NUMERICAL_ERROR)
        self.assertIsNone(result.objective)
        self.assertIsNone(result.best_bound)


if __name__ == '__main__':
    unittest.main()
