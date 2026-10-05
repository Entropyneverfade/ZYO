"""原域大数乘积抵消的独立反例：不以求解器自己的行活动量作预期。"""

from fractions import Fraction
import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from zyo import Model
from zyo.lp_certificate import lp_certificate
from zyo.native_lp import build_model, solve_certified_lp


class LPProductCancellationTests(unittest.TestCase):
    def setUp(self):
        # 二进制浮点存储的原始系数和候选值是验收对象；分数只用于独立核算。
        self.n0 = 1421803362854.3792
        self.d0 = 1151660734277.0
        self.n1 = 2029678531776.0918
        self.d1 = 1644039625535.0
        self.step = 1.23456789012345
        self.matrix = csc_matrix(np.array([[1.0, 0.0, -self.d0],
                                           [0.0, 1.0, -self.d1]]))
        self.values = np.array([self.n0, self.n1, self.step])

    def test_certificate_rejects_rounded_product_cancellation(self):
        model = build_model(self.matrix, [0.0, 0.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [self.n0, self.n1, math.inf], [0.0, 0.0],
                            sense=['==', '=='])
        # 浮点乘积先舍入时，两行误差都变成 0；精确乘积再相减并非 0。
        residuals = [Fraction.from_float(float(self.values[i]))
                     - Fraction.from_float(float(d))*Fraction.from_float(self.step)
                     for i, d in enumerate((self.d0, self.d1))]
        self.assertGreater(max(map(abs, residuals)), Fraction(1, 100000))
        certificate = lp_certificate(model, self.values,
                                     [8.683112745246013e-13, 0.0])
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.primal_feasible)
        self.assertGreater(certificate.row_violation, 1e-5)

    def test_nonoptimal_candidate_is_not_marked_primal_feasible(self):
        model = build_model(self.matrix, [0.0, 0.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [self.n0, self.n1, math.inf], [0.0, 0.0],
                            sense=['==', '=='])
        # 零乘子使 KKT 初筛失败；Phase-I 候选仍会读取 primal_feasible 字段。
        certificate = lp_certificate(model, self.values, [0.0, 0.0])
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.primal_feasible)
        self.assertGreater(certificate.row_violation, 1e-5)

    def test_public_entry_does_not_promote_cancellation_to_optimal(self):
        result = solve_certified_lp(
            self.matrix, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0],
            [self.n0, self.n1, math.inf], [0.0, 0.0],
            sense=['==', '=='], warm_basis=[0, 1],
            warm_nonbasic_at_upper=[], require_warm_start=True,
            iteration_limit=1)
        self.assertFalse(result.optimal)
        self.assertNotEqual(result.status, 'OPTIMAL')

    def test_exactly_feasible_but_suboptimal_point_rejects_cost_cancellation(self):
        coefficient = self.d0
        multiplier = self.step
        rounded_cost = float(coefficient*multiplier)
        model = build_model(csc_matrix([[coefficient, 1.0]]),
                            [rounded_cost, multiplier],
                            [0.0, -math.inf], [math.inf, math.inf],
                            [0.0], sense=['=='])
        # 约束 a*x+y=0 在 (1,-a) 与 (0,0) 均精确满足。
        # c=fl(a*lambda)，但原存储数字的真实 c-a*lambda > 0，故前者严格次优。
        true_gap = (Fraction.from_float(rounded_cost)
                    - Fraction.from_float(coefficient)*Fraction.from_float(multiplier))
        self.assertGreater(true_gap, Fraction(1, 100000))
        certificate = lp_certificate(model, [1.0, -coefficient], [multiplier])
        self.assertTrue(certificate.primal_feasible)
        self.assertFalse(certificate.verified)
        self.assertGreater(certificate.objective, 1e-5)
        self.assertGreater(certificate.reduced_costs[0], 1e-5)

    def test_negative_gap_is_not_hidden_by_large_roundoff_allowance(self):
        right = 1e8
        cost = 1e14
        candidate = float(np.nextafter(right, -math.inf))
        model = Model('small_row_large_multiplier')
        variable = model.add_var('x', lb=None, ub=None)
        model.add_constr(variable == right)
        model.minimize(cost*variable-cost*right)
        # 原行在给定容差内，但精确目标比唯一可行点低逾 10^6；
        # 行容差不应被当作无需核对的目标误差预算。
        self.assertLessEqual(abs(Fraction.from_float(candidate)
                                 - Fraction.from_float(right)), Fraction.from_float(1e-7))
        true_gap = Fraction.from_float(cost)*(Fraction.from_float(candidate)
                                             - Fraction.from_float(right))
        self.assertLess(true_gap, -1000000)
        certificate = lp_certificate(model, [candidate], [cost])
        self.assertTrue(certificate.primal_feasible)
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.strongly_dual)

    def test_unbounded_improvement_directions_reject_tiny_wrong_signs(self):
        cases = ((0.0, None, -1e-8),
                 (None, 0.0, 1e-8),
                 (None, None, 1e-8))
        for lower, upper, cost in cases:
            with self.subTest(lower=lower, upper=upper, cost=cost):
                model = Model('tiny_unbounded_direction')
                variable = model.add_var('x', lb=lower, ub=upper)
                model.minimize(cost*variable)
                # 下界-only 负成本、上界-only 正成本、自由变量非零成本
                # 各自都有严格改善射线；再小的系数也不能给全局最优证书。
                certificate = lp_certificate(model, [0.0], [])
                self.assertFalse(certificate.verified)
                self.assertFalse(certificate.dual_feasible)

    def test_exact_sign_gate_keeps_small_true_optima(self):
        cases = ((0.0, None, 1e-8),
                 (None, 0.0, -1e-8),
                 (None, None, 0.0))
        for lower, upper, cost in cases:
            with self.subTest(lower=lower, upper=upper, cost=cost):
                model = Model('tiny_correct_direction')
                variable = model.add_var('x', lb=lower, ub=upper)
                model.minimize(cost*variable)
                certificate = lp_certificate(model, [0.0], [])
                self.assertTrue(certificate.verified, certificate.reason)

    def test_small_rational_multiplier_witness_is_explicit(self):
        model = build_model(csc_matrix([[1.0, 1.0], [1.0, 3.0]]),
                            [-1.0, -2.0], [0.0, 0.0], [1.0, math.inf],
                            [4.0, 6.0], sense=['<=', '<='])
        candidate = [1.0, 5.0/3.0]
        # 原求解乘子 -0.666... 的二进制浮点尾差会给 y 留下负简约成本；
        # 有理 -2/3 是另一个可核查的证书见证，不可偷偷改写原乘子。
        raw_multipliers = [-0.0, -0.6666666666666666]
        certificate = lp_certificate(model, candidate, raw_multipliers)
        self.assertTrue(certificate.verified, certificate.reason)
        self.assertTrue(certificate.reconstruction_used)
        self.assertEqual(certificate.witness_multipliers, ['0', '-2/3'])

    def test_rational_reconstruction_respects_row_limit(self):
        matrix = np.zeros((17, 2))
        matrix[-1] = [1.0, 3.0]
        model = build_model(csc_matrix(matrix), [-1.0, -2.0],
                            [0.0, 0.0], [1.0, math.inf],
                            [0.0]*16+[6.0], sense=['==']*16+['<='])
        certificate = lp_certificate(model, [1.0, 5.0/3.0],
                                     [0.0]*16+[-0.6666666666666666])
        # 同样的小乘子误差在超过硬规模门后保持失败，不偷偷作昂贵修复。
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.reconstruction_attempted)
        self.assertFalse(certificate.reconstruction_used)


if __name__ == '__main__':
    unittest.main()
