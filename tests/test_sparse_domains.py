# 稀疏路径变量域扩展的验收测试：先要求失败，再最小实现。
# 独立预期全部来自手算或有理数精确核验，不读取被测算法的标准化、界或证书函数。
import importlib.util
import math
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

from zyo import Model, SolveOptions
from zyo.sparse_domain import domain_bound
from zyo.sparse_mip import _solve_domains
from zyo.status import Status
from lzyopt import quicksum

SCIPY = importlib.util.find_spec('scipy') is not None
OPTIONS = SolveOptions(time_limit=10, iteration_limit=2000, node_limit=100,
                       feasibility_tol=1e-7, integrality_tol=1e-7, objective_tol=1e-8)


def exact_activity(expression, point):
    """用 Fraction 精确复算表达式取值，避免复用被测实现。"""
    return Fraction(expression.constant) + sum(Fraction(a)*point[i] for i, a in expression.terms.items())


def exact_objective(model, point):
    return exact_activity(model.objective, point)


@unittest.skipUnless(SCIPY, 'SciPy sparse linear algebra required')
class VariableDomainAcceptanceTests(unittest.TestCase):
    def solve_checked(self, model, expected_objective, expected_values=None):
        before = model.to_dict()
        result = model.solve('native_sparse', options=OPTIONS)
        # 求解不得改写调用方的模型（含无穷界）。
        self.assertEqual(model.to_dict(), before, 'solver mutated the caller model')
        self.assertEqual(result.solver_name, 'native_sparse')
        self.assertFalse(result.metadata['fallback_used'])
        self.assertEqual(result.status, 'OPTIMAL', result.termination_reason)
        self.assertTrue(result.has_solution)
        # 原始模型单位下的有理数复核：目标、行、变量界。
        point = {i: Fraction.from_float(result.values[v.name]) for i, v in enumerate(model.variables)}
        self.assertLessEqual(abs(float(exact_objective(model, point))-expected_objective), 1e-7)
        for row in model.constraints:
            value = exact_activity(row.expression, point)
            violation = abs(value) if row.sense == '==' else max(Fraction(0), value if row.sense == '<=' else -value)
            self.assertLessEqual(float(violation), 1e-7, f'row violated: {value}')
        for i, var in enumerate(model.variables):
            if math.isfinite(var.lb):
                self.assertLessEqual(float(Fraction(var.lb)-point[i]), 1e-7, f'{var.name} below lb')
            if math.isfinite(var.ub):
                self.assertLessEqual(float(point[i]-Fraction(var.ub)), 1e-7, f'{var.name} above ub')
        # 报告的界必须在原问题域上有效：最小化时是下界，最大化时是上界。
        if result.best_bound is not None:
            direction = 1.0 if model.sense == 'min' else -1.0
            gap = direction*(expected_objective-result.best_bound)/max(1.0, abs(expected_objective))
            self.assertGreaterEqual(gap, -1e-7, 'reported bound is on the wrong side of the optimum')
            self.assertLessEqual(gap, min(OPTIONS.objective_tol, 1e-8), 'reported gap exceeds the declared tolerance')
        if expected_values:
            for name, value in expected_values.items():
                self.assertAlmostEqual(result.values[name], value, delta=1e-6, msg=f'{name} value')
        return result

    def test_lower_bounded_variables_are_solved_not_unknown(self):
        # min x + 2y, x,y >= 0, x + y >= 5 -> 最优 x=5,y=0，目标 5。
        # 只有一个方向有界：旧实现在此返回 UNKNOWN。
        model = Model('lower_only')
        x = model.add_var('x', lb=0, ub=None)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(x + y >= 5)
        model.minimize(x + 2*y)
        self.solve_checked(model, 5.0, {'x': 5.0, 'y': 0.0})

    def test_lower_bounded_maximization_with_finite_optimum(self):
        # max 3x + y, x,y >= 0, x + y <= 4, x <= 3 -> 最优 x=3,y=1，目标 10。
        model = Model('lower_max')
        x = model.add_var('x', lb=0, ub=None)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(x + y <= 4)
        model.add_constr(x <= 3)
        model.maximize(3*x + y)
        result = self.solve_checked(model, 10.0, {'x': 3.0, 'y': 1.0})
        self.assertAlmostEqual(result.best_bound, 10.0, delta=1e-7,
                               msg='max 目标的公开上界必须回到调用方目标方向')
        proof = result.metadata['domain_transform']['attempts'][0]['bound_certificate']
        self.assertEqual(proof['objective_sense'], 'max')
        self.assertAlmostEqual(proof['bound'], result.best_bound, delta=1e-7)
        self.assertAlmostEqual(proof['normalized_min_bound'], -result.best_bound,
                               delta=1e-7)

    def test_selected_earlier_width_preserves_root_and_full_attempt_trace(self):
        # 第一档界已闭合但未达提前返回门，后两档数值失败时仍须保留真实根证据和计数。
        model = Model('selected_earlier_width')
        x = model.add_var('x', lb=0, ub=None)
        model.minimize(x)
        inner = SimpleNamespace(
            status=Status.OPTIMAL, values={'x#l': 0.0},
            metadata={'root_relaxation': {'certificate': {
                'scaled_multipliers': [], 'multipliers': []}}, 'root_marker': 17},
            node_count=7, iteration_count=13, termination_reason='checked box')
        failed = SimpleNamespace(status=Status.NUMERICAL_ERROR, values={}, metadata={},
                                 node_count=1, iteration_count=2,
                                 termination_reason='later width failed')
        bound = dict(available=True, bound=-5e-9, raw_bound=-5e-9,
                     roundoff_allowance=0.0, multipliers_exact=[],
                     multiplier_source='test_exact', scope='normalized weak bound')
        with patch('zyo.sparse_mip._solve_box', side_effect=[inner, failed, failed]), \
             patch('zyo.sparse_domain.domain_bound', return_value=bound):
            result = _solve_domains(model, OPTIONS)
        self.assertEqual(result.status, Status.OPTIMAL)
        # 公开计数是全部三档实际消耗；被采纳的第一档本身仍单独保留。
        self.assertEqual((result.node_count, result.iteration_count), (9, 17))
        self.assertEqual(result.metadata['root_marker'], 17)
        self.assertEqual(len(result.metadata['domain_transform']['attempts']), 3)
        self.assertEqual(result.metadata['domain_transform']['selected_nodes'], 7)
        self.assertEqual(result.metadata['domain_transform']['selected_iterations'], 13)
        self.assertEqual(result.metadata['domain_transform']['selected_by'],
                         'tightest closed original-domain gap')

    def test_bounded_optimum_beyond_artificial_ladder_is_not_unbounded(self):
        # 原问题为 0 <= x <= 1e6，min -x 的最优值是 -1e6；人造宽度不是原域射线。
        model = Model('bounded_beyond_artificial_ladder')
        x = model.add_var('x', lb=0, ub=None)
        model.add_constr(x <= 1e6)
        model.minimize(-x)
        result = model.solve('native_sparse', options=OPTIONS)
        self.assertNotEqual(result.status, 'UNBOUNDED',
                            '有限原域被人造界处的负简约成本误报为无界')

    def test_tiny_negative_lower_domain_reduced_cost_has_no_finite_dual_bound(self):
        # 乘子为零时 L(x)=-1e-7*x 在原域 x>=0 的下确界为 -inf，尽管原 LP 最优值为 -0.1。
        model = Model('tiny_negative_lower_domain_reduced_cost')
        x = model.add_var('x', lb=0, ub=None)
        model.add_constr(x <= 1e6)
        model.minimize(-1e-7*x)
        candidate = domain_bound(model, [0.0])
        self.assertFalse(candidate['available'],
                         '舍入容差把 -inf 拉格朗日下确界误当作有限原域界')

    def test_tiny_nonzero_free_domain_reduced_cost_has_no_finite_dual_bound(self):
        # 自由变量沿负方向令 L(x)=1e-7*x 趋于 -inf；数值近零不等于严格零。
        model = Model('tiny_nonzero_free_domain_reduced_cost')
        x = model.add_var('x', lb=None, ub=None)
        model.minimize(1e-7*x)
        candidate = domain_bound(model, [])
        self.assertFalse(candidate['available'],
                         '舍入容差把自由变量的 -inf 下确界误当作有限原域界')

    def test_binary64_cancellation_cannot_certify_finite_original_domain_bound(self):
        # 顺序计算 1e16-1-1e16 会舍入为零，但已解析的 binary64 系数精确差为 -1。
        model = Model('binary64_reduced_cost_cancellation')
        x = model.add_var('x', lb=0, ub=None)
        model.add_constr(x == 0)
        model.add_constr(x == 0)
        model.minimize(1e16*x)
        multipliers = [1.0, 1e16]
        exact_reduced = Fraction.from_float(float(model.objective.terms[x.index]))
        for row, multiplier in zip(model.constraints, multipliers):
            exact_reduced -= (Fraction.from_float(float(row.expression.terms[x.index]))
                              * Fraction.from_float(multiplier))
        self.assertEqual(exact_reduced, Fraction(-1))
        candidate = domain_bound(model, multipliers)
        self.assertFalse(candidate['available'],
                         '精确简约成本为负时，不得用浮点抵消后的零值声明有限原域界')

    def test_upper_bounded_variables_are_solved_not_unknown(self):
        # min x + 2y, x <= 0, y <= 0, x + y >= -5。
        # 目标里 y 系数更大，故把全部 -5 放在 x 上更差（-5）；最优是 x=0, y=-5，目标 -10。
        # （网格核对：(0,-5) 可行且 x+2y=-10；(0,0) 为 0，(-5,0) 为 -5。）
        model = Model('upper_only')
        x = model.add_var('x', lb=None, ub=0)
        y = model.add_var('y', lb=None, ub=0)
        model.add_constr(x + y >= -5)
        model.minimize(x + 2*y)
        self.solve_checked(model, -10.0, {'x': 0.0, 'y': -5.0})

    def test_free_variables_are_solved_not_unknown(self):
        # min x + y, x,y 自由, x + y >= 5, x - y == 0 -> 最优 x=y=2.5，目标 5。
        model = Model('free_pair')
        x = model.add_var('x', lb=None, ub=None)
        y = model.add_var('y', lb=None, ub=None)
        model.add_constr(x + y >= 5)
        model.add_constr(x - y == 0)
        model.minimize(x + y)
        self.solve_checked(model, 5.0, {'x': 2.5, 'y': 2.5})

    def test_free_variable_optimum_exposes_original_domain_dual_witness(self):
        # 数值可行的原解需配有可按原始 >= 行重算的严格对偶界；人造盒乘子不充当证明。
        model = Model('free_pair_dual_witness')
        x = model.add_var('x', lb=None, ub=None)
        y = model.add_var('y', lb=None, ub=None)
        model.add_constr(x + y >= 5)
        model.add_constr(x - y == 0)
        model.minimize(x + y)
        result = model.solve('native_sparse', options=OPTIONS)
        self.assertEqual(result.status, 'OPTIMAL', result.termination_reason)
        attempts = result.metadata['domain_transform']['attempts']
        proofs = [attempt['bound_certificate'] for attempt in attempts
                  if 'bound_certificate' in attempt]
        self.assertTrue(proofs, 'original-domain weak-duality proof was not exported')
        proof = proofs[0]
        weights = [Fraction(value) for value in proof['multipliers_exact']]
        self.assertEqual(len(weights), len(model.constraints))
        self.assertGreaterEqual(weights[0], 0, 'original >= row requires nonnegative multiplier')
        reduced = [Fraction(model.objective.terms.get(j, 0)) - sum(
            Fraction(row.expression.terms.get(j, 0))*weight
            for row, weight in zip(model.constraints, weights))
            for j in range(len(model.variables))]
        self.assertEqual(reduced, [0, 0])
        exact = Fraction(model.objective.constant)-sum(
            Fraction(row.expression.constant)*weight
            for row, weight in zip(model.constraints, weights))
        self.assertEqual(exact, 5)
        self.assertLessEqual(Fraction.from_float(proof['bound']), exact)
        self.assertAlmostEqual(result.best_bound, proof['bound'], delta=1e-8)

    def test_mixed_one_sided_domains_with_bounded_optimum(self):
        # 混合域且有界最优：min a + 2b + c, a >= 0, b ∈ [-2,0], c 自由;
        # a - b + c == 3; a + b >= 1。
        # 由等式得 c = 3 - a + b，代入目标得 a + 2b + c = 3 + 3b，故最优在 b = -2，
        # 目标值 -3；此时 a >= 1 - b = 3，可选 a = 3, c = 2 或 a = 442.19, c = -437.19
        # 等多组解，因此只断言目标值与可行性，不断言唯一解。
        model = Model('mixed_bounded')
        a = model.add_var('a', lb=0, ub=None)
        b = model.add_var('b', lb=-2, ub=0)
        c = model.add_var('c', lb=None, ub=None)
        model.add_constr(a - b + c == 3)
        model.add_constr(a + b >= 1)
        model.minimize(a + 2*b + c)
        result = self.solve_checked(model, -3.0)
        self.assertLessEqual(abs(result.values['b'] + 2.0), 1e-6, 'optimum needs b at its lower bound')

    def test_unbounded_through_one_sided_variable_is_not_clipped(self):
        # 同一结构但 b 可以无限减小：min b - a, a>=0, b<=0, c 自由;
        # a - b + c == 3; a + b >= 1。取 a = t, b = 1 - t（t>=1）、c = 3 - a + b = 4 - 2t，
        # 目标 b - a = 1 - 2t -> -inf；原域射线尚无证书时可保守未决，绝不能把裁剪点当最优。
        model = Model('unbounded_one_sided')
        a = model.add_var('a', lb=0, ub=None)
        b = model.add_var('b', lb=None, ub=0)
        c = model.add_var('c', lb=None, ub=None)
        model.add_constr(a - b + c == 3)
        model.add_constr(a + b >= 1)
        model.minimize(b - a)
        result = model.solve('native_sparse', options=OPTIONS)
        self.assertNotEqual(result.status, 'OPTIMAL',
                            'a clipped candidate on an unbounded ray was reported as an optimum')
        self.assertFalse(result.has_solution)
        self.assertIsNone(result.best_bound)

    def test_genuinely_unbounded_model_is_never_reported_optimal(self):
        # min -x, x >= 0：目标无下界。人造界绝不能被当作最优解报出。
        model = Model('unbounded_lower')
        x = model.add_var('x', lb=0, ub=None)
        model.minimize(-x)
        result = model.solve('native_sparse', options=OPTIONS)
        self.assertNotEqual(result.status, 'OPTIMAL',
                            'clipped artificial bound was reported as an optimum')
        if not result.has_solution:
            self.assertEqual(dict(result.values), {})
            self.assertIsNone(result.best_bound)
            self.assertIsNone(result.mip_gap)
            self.assertTrue(result.termination_reason)

    def test_infeasible_one_sided_model_is_not_reported_optimal(self):
        model = Model('infeasible_lower')
        x = model.add_var('x', lb=0, ub=None)
        model.add_constr(x <= -1)
        model.minimize(x)
        result = model.solve('native_sparse', options=OPTIONS)
        self.assertNotEqual(result.status, 'OPTIMAL')
        self.assertFalse(result.has_solution)


if __name__ == '__main__':
    unittest.main()
