"""原输入有限但中间浮点乘积上溢时，证书必须结构化拒绝。"""

import unittest

from scipy.sparse import csc_matrix

from zyo.native_lp import build_model
from zyo.lp_certificate import lp_certificate


class LPCertificateNonfiniteProductTests(unittest.TestCase):
    def test_objective_product_cancellation_reports_unverified(self):
        # 精确目标为 0，但两个浮点中间乘积为 +inf、-inf；不允许异常逸出或假通过。
        model = build_model(csc_matrix((0, 2)), [1e200, -1e200],
                            [1e200, 1e200], [1e200, 1e200], [])
        certificate = lp_certificate(model, [1e200, 1e200], [])
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.primal_feasible)
        self.assertIn('activity', certificate.reason)

    def test_row_product_cancellation_reports_unverified(self):
        model = build_model(csc_matrix([[1e200, -1e200]]), [0.0, 0.0],
                            [1e200, 1e200], [1e200, 1e200], [0.0], sense=['=='])
        certificate = lp_certificate(model, [1e200, 1e200], [0.0])
        self.assertFalse(certificate.verified)
        self.assertFalse(certificate.primal_feasible)
        self.assertIn('activity', certificate.reason)

    def test_dual_product_cancellation_reports_unverified(self):
        # 原行活动量各为0；对偶 RHS×乘子会先上溢成 +inf 与 -inf。
        model = build_model(csc_matrix([[1.0], [1.0]]), [0.0],
                            [1e200], [1e200], [1e200, 1e200], sense=['==', '=='])
        certificate = lp_certificate(model, [1e200], [1e200, -1e200])
        self.assertFalse(certificate.verified)
        self.assertIn('dual', certificate.reason)

    def test_dual_overflow_keeps_independent_primal_measurements(self):
        # 起点 x=0、目标0、盒可行均可核；只因对偶盒端点 -1e310 上溢拒绝最优。
        model = build_model(csc_matrix((0, 1)), [-1e308], [0.0], [100.0], [])
        certificate = lp_certificate(model, [0.0], [])
        self.assertFalse(certificate.verified)
        self.assertTrue(certificate.primal_feasible)
        self.assertEqual(certificate.objective, 0.0)
        self.assertIn('dual', certificate.reason)


if __name__ == '__main__':
    unittest.main()
