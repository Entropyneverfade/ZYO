# ZYO-033 独立验收：KKT/对偶证书必须接受真最优、拒绝伪最优、拒绝错误乘子。
# 预期值全部手算或由独立稠密公式给出，不使用被测证书模块的判定结果作为输入。
import math
import unittest

import numpy as np

from zyo import Model
from zyo.lp_certificate import lp_certificate


def build_lp():
    """min -x - 2y ; x + y <= 4 ; x + 3y <= 6 ; x, y >= 0。最优 (3, 1)，目标 -5。

    模型行以 "活动量 SENSE 0" 存储：<= 行为 sum(a x) - c <= 0。
    """
    model = Model('kkt_lp')
    x = model.add_var('x', lb=0, ub=None)
    y = model.add_var('y', lb=0, ub=None)
    model.add_constr(x + y <= 4)
    model.add_constr(x + 3*y <= 6)
    model.minimize(-x - 2*y)
    return model


class LPCertificateTests(unittest.TestCase):
    def setUp(self):
        self.model = build_lp()

    # ---------- 接受真最优 ----------

    def test_accepts_the_true_optimum_with_correct_multipliers(self):
        # 手算：最优点 (3,1)；两行均取等号，KKT 给出 λ=(-0.5,-0.5)（<= 行乘子为负）。
        result = lp_certificate(self.model, [3.0, 1.0], [-0.5, -0.5])
        self.assertTrue(result.verified, result.reason)
        self.assertTrue(result.primal_feasible)
        self.assertTrue(result.dual_feasible)
        self.assertTrue(result.complementary)
        self.assertTrue(result.strongly_dual)
        self.assertAlmostEqual(result.objective, -5.0, places=9)
        self.assertAlmostEqual(result.dual_objective, -5.0, places=7)
        self.assertLessEqual(abs(result.relative_gap), 1e-7)
        # 简约成本：基变量为 0，非基变量在界上且符号相容。
        self.assertAlmostEqual(result.reduced_costs[0], 0.0, places=7)
        self.assertAlmostEqual(result.reduced_costs[1], 0.0, places=7)

    # ---------- 拒绝伪最优 ----------

    def test_rejects_a_feasible_but_suboptimal_point(self):
        # (0,0) 可行但目标 0，远非最优；用正确的乘子也必须被拒。
        result = lp_certificate(self.model, [0.0, 0.0], [-0.5, -0.5])
        self.assertFalse(result.verified)
        self.assertFalse(result.strongly_dual)
        self.assertGreater(result.duality_gap, 1.0)

    def test_rejects_an_infeasible_point(self):
        # (5,0) 违反 x + y <= 4。
        result = lp_certificate(self.model, [5.0, 0.0], [-0.5, -0.5])
        self.assertFalse(result.verified)
        self.assertFalse(result.primal_feasible)
        self.assertGreater(result.row_violation, 0.5)

    def test_rejects_a_point_outside_variable_bounds(self):
        result = lp_certificate(self.model, [3.0, -1.0], [0.0, 0.0])
        self.assertFalse(result.verified)
        self.assertFalse(result.primal_feasible)
        self.assertGreater(result.bound_violation, 0.5)

    def test_rejects_a_multiplier_with_the_wrong_row_sign(self):
        # <= 行乘子必须非正；给正值应被拒。
        result = lp_certificate(self.model, [3.0, 1.0], [0.5, 0.5])
        self.assertFalse(result.verified)
        self.assertFalse(result.dual_feasible)
        self.assertFalse(result.checks['row_sign_convention'])

    def test_rejects_multipliers_that_violate_reduced_cost_domains(self):
        # x 只有下界，其简约成本必须 >= 0；取 λ=(1,1) 会让 r_x 变负。
        result = lp_certificate(self.model, [3.0, 1.0], [1.0, 1.0])
        self.assertFalse(result.verified)
        self.assertFalse(result.dual_feasible)

    def test_rejects_wrong_length_inputs(self):
        self.assertFalse(lp_certificate(self.model, [3.0, 1.0, 0.0], [-0.5, -0.5]).verified)
        self.assertFalse(lp_certificate(self.model, [3.0, 1.0], [-0.5]).verified)
        self.assertFalse(lp_certificate(self.model, [3.0, float('nan')], [-0.5, -0.5]).verified)

    # ---------- 域类型与方向 ----------

    def test_free_variable_requires_zero_reduced_cost(self):
        # min x ; x 自由 ; x >= 0 隐含 -> 用显式一行 x >= 0 表达，最优 x=0。
        model = Model('free_case')
        x = model.add_var('x', lb=None, ub=None)
        model.add_constr(x >= 0)
        model.minimize(x)
        # 最优 x=0，且自由变量的简约成本必须为 0：λ 必须取 1（>= 行乘子非负）。
        good = lp_certificate(model, [0.0], [1.0])
        self.assertTrue(good.verified, good.reason)
        bad = lp_certificate(model, [0.0], [0.0])
        self.assertFalse(bad.verified)

    def test_maximization_direction_flips_the_sign_convention(self):
        # max x + y ; x + y <= 4 ; x, y >= 0 最优 (4,0) 或等价，目标 4。
        model = Model('max_case')
        x = model.add_var('x', lb=0, ub=None)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(x + y <= 4)
        model.maximize(x + y)
        # 最大化下，<= 行的有符号乘子应为正。
        good = lp_certificate(model, [4.0, 0.0], [1.0])
        self.assertTrue(good.verified, good.reason)
        self.assertAlmostEqual(good.objective, 4.0, places=9)
        bad = lp_certificate(model, [4.0, 0.0], [-1.0])
        self.assertFalse(bad.verified)

    def test_integrality_violation_is_reported(self):
        model = Model('int_case')
        z = model.add_var('z', lb=0, ub=10, vtype='I')
        model.minimize(z)
        # z=2.5 时整性违反 0.5，必须被拒。
        result = lp_certificate(model, [2.5], [])
        self.assertFalse(result.verified)
        self.assertFalse(result.primal_feasible)
        self.assertAlmostEqual(result.integrality_violation, 0.5, places=9)

    def test_certificate_records_every_condition_for_audit(self):
        result = lp_certificate(self.model, [3.0, 1.0], [-0.5, -0.5])
        for key in ('row_sign_convention', 'reduced_cost_domain_compatible',
                    'primal_within_tolerance', 'dual_within_tolerance',
                    'complementarity_within_tolerance', 'strong_duality_within_tolerance'):
            self.assertIn(key, result.checks)
        self.assertEqual(len(result.reduced_costs), 2)
        self.assertEqual(len(result.row_activities), 2)
        self.assertGreaterEqual(result.roundoff_allowance, 0.0)
        self.assertIn('original model units', result.scope)

    # ---------- 双边有界变量的互补松弛（曾判错，回归锁定） ----------
    #
    # KKT 对 l_j <= x_j <= u_j 只要求**单向蕴含**：r_j > 0 则 x_j = l_j，r_j < 0 则
    # x_j = u_j。停在 l_j 时 r_j(u_j-x_j) 一般不为零（μ_j = r_j > 0、ν_j = 0），
    # 早期版本却同时要求两个乘积为零，因而拒绝了真最优。

    def test_upper_bounded_optimum_is_accepted(self):
        # min -x ; 0 <= x <= 1。最优点 x=1、r=-1，必须通过证书。
        model = Model('box_upper')
        x = model.add_var('x', lb=0.0, ub=1.0)
        model.minimize(-1.0*x)
        result = lp_certificate(model, [1.0], [])
        self.assertTrue(result.verified, result.reason)
        self.assertAlmostEqual(result.complementarity_violation, 0.0, places=9)
        self.assertAlmostEqual(result.objective, -1.0, places=9)

    def test_lower_bounded_optimum_of_the_same_box_is_accepted(self):
        # min x ; 0 <= x <= 1。最优点 x=0、r=+1，同样必须通过。
        model = Model('box_lower')
        x = model.add_var('x', lb=0.0, ub=1.0)
        model.minimize(1.0*x)
        result = lp_certificate(model, [0.0], [])
        self.assertTrue(result.verified, result.reason)
        self.assertAlmostEqual(result.complementarity_violation, 0.0, places=9)

    def test_optimum_on_the_wrong_side_of_a_positive_reduced_cost_is_rejected(self):
        # 同一 min x 的题，报 x=1（上界）而 r=+1：正简约成本要求停在下界，必须被拒。
        model = Model('box_wrong_side')
        x = model.add_var('x', lb=0.0, ub=1.0)
        model.minimize(1.0*x)
        result = lp_certificate(model, [1.0], [])
        self.assertFalse(result.verified)
        self.assertFalse(result.complementary)
        self.assertAlmostEqual(result.complementarity_violation, 1.0, places=9)

    def test_fixed_variable_has_an_unrestricted_reduced_cost(self):
        # 固定变量（上下界相等）的简约成本不受任何符号约束，必须能通过证书。
        model = Model('fixed_var')
        x = model.add_var('x', lb=2.0, ub=2.0)
        model.minimize(3.0*x)
        result = lp_certificate(model, [2.0], [])
        self.assertTrue(result.verified, result.reason)
        self.assertAlmostEqual(result.complementarity_violation, 0.0, places=9)

    def test_interior_variable_with_nonzero_reduced_cost_is_rejected(self):
        # x=0.5 严格位于界内而 r=-1 != 0：这才是真正的互补松弛违反。
        model = Model('interior')
        x = model.add_var('x', lb=0.0, ub=1.0)
        model.minimize(-1.0*x)
        result = lp_certificate(model, [0.5], [])
        self.assertFalse(result.verified)
        self.assertFalse(result.complementary)
        self.assertGreaterEqual(result.complementarity_violation, 0.4)


if __name__ == '__main__':
    unittest.main()
