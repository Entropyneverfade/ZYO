# 变量域扩展前的解析契约：检验平移、反向、自由变量拆分和状态，不使用外部求解器预期。
import importlib.util
import math
import unittest
from fractions import Fraction

from zyo import Model, SolveOptions


# 仅供独立实验入口保存输入、实际结果与参数；正常回归不写文件、不改变生产类。
RECORDED_RUNS = []
OPTIONS = SolveOptions(time_limit=2, iteration_limit=1000, node_limit=100,
                       feasibility_tol=1e-7, integrality_tol=1e-7, objective_tol=1e-8)


def affine_model(domain, scale, sense):
    # 基础多面体：u,v>=0，u+v<=4，u,v<=3；2u+v=u+(u+v)<=7，在(3,1)取等。
    # 因而两种目标 -(2u+v)+11 的最小值4、(2u+v)+11的最大值18均由手算确定。
    model = Model(f'affine_{domain}_{scale}_{sense}')
    if domain == 'finite':
        bounds_x = (min(-7, -7+4*scale), max(-7, -7+4*scale))
        bounds_y, bounds_w = (-3, 5), (-4, 4)
    elif domain == 'half_line':
        bounds_x = (-7, None) if scale > 0 else (None, -7)
        bounds_y, bounds_w = (None, 5), (None, None)
    else:
        bounds_x = bounds_y = bounds_w = (None, None)
    x = model.add_var('x', lb=bounds_x[0], ub=bounds_x[1])
    y = model.add_var('y', lb=bounds_y[0], ub=bounds_y[1])
    w = model.add_var('w', lb=bounds_w[0], ub=bounds_w[1])
    z = model.add_var('fixed', lb=-2, ub=-2)
    u, v = (x+7)*(1/scale), (5-y)*0.5
    model.add_constr(u >= 0)
    model.add_constr(v >= 0)
    model.add_constr(u+v <= 4)
    model.add_constr(u <= 3)
    model.add_constr(v <= 3)
    model.add_constr(w == u-v)
    model.add_constr(2*w == 2*u-2*v)
    model.set_objective((2*u+v)*(-1 if sense == 'min' else 1)+3*z+17, sense)
    return model


class VariableDomainTests(unittest.TestCase):
    def recorded_solve(self, model, engine):
        before = model.to_dict()
        result = model.solve(engine, options=OPTIONS)
        RECORDED_RUNS.append(dict(model=before, result=result.to_dict()))
        # 防止为迎合有限盒内核而永久改写调用方的无穷边界或约束。
        self.assertEqual(model.to_dict(), before)
        self.assertEqual(result.solver_name, engine)
        self.assertFalse(result.metadata['fallback_used'])
        return result

    def check_optimum(self, model, result, expected):
        self.assertEqual(result.status, 'OPTIMAL', result.termination_reason)
        self.assertTrue(result.has_solution)
        self.assertAlmostEqual(result.objective, expected, delta=1e-7)
        self.assertEqual(set(result.values), {v.name for v in model.variables})
        # 用原始存储系数的有理数重算；不调用被测标准化、目标或可行性检查函数。
        point = {i: Fraction.from_float(result.values[v.name]) for i, v in enumerate(model.variables)}
        def activity(expression):
            return Fraction(expression.constant)+sum(Fraction(a)*point[i] for i, a in expression.terms.items())
        self.assertLessEqual(abs(float(activity(model.objective))-expected), 1e-7)
        for row in model.constraints:
            value = activity(row.expression)
            violation = abs(value) if row.sense == '==' else max(0, value if row.sense == '<=' else -value)
            self.assertLessEqual(float(violation), 1e-7)
        for i, var in enumerate(model.variables):
            if math.isfinite(var.lb):
                self.assertLessEqual(float(Fraction(var.lb)-point[i]), 1e-7)
            if math.isfinite(var.ub):
                self.assertLessEqual(float(point[i]-Fraction(var.ub)), 1e-7)

    def check_unknown(self, result):
        # 未支持不是数值证明：禁止出现伪候选、伪界或回退结果。
        self.assertEqual(result.status, 'UNKNOWN', result.termination_reason)
        self.assertFalse(result.has_solution)
        self.assertEqual(dict(result.values), {})
        self.assertIsNone(result.best_bound)
        self.assertIsNone(result.mip_gap)
        self.assertTrue(result.termination_reason)

    def test_dense_affine_offsets_signs_and_fixed_objective(self):
        # 错误的上界反向符号、固定量目标消去、最大化恢复或自由变量拆分均会破坏手算目标。
        for domain in ('finite', 'half_line', 'free'):
            for scale in (0.25, -1, 8):
                for sense, expected in (('min', 4), ('max', 18)):
                    with self.subTest(domain=domain, scale=scale, sense=sense):
                        model = affine_model(domain, scale, sense)
                        result = self.recorded_solve(model, 'native')
                        self.check_optimum(model, result, expected)
                        self.assertAlmostEqual(result.values['x'], -7+3*scale, delta=1e-7)
                        self.assertAlmostEqual(result.values['y'], 3, delta=1e-7)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy sparse linear algebra required')
    def test_sparse_affine_models_or_explicit_unsupported_state(self):
        for domain in ('finite', 'half_line', 'free'):
            for scale in (0.25, -1, 8):
                for sense, expected in (('min', 4), ('max', 18)):
                    with self.subTest(domain=domain, scale=scale, sense=sense):
                        model = affine_model(domain, scale, sense)
                        result = self.recorded_solve(model, 'native_sparse')
                        # 将来经正式验证支持半无限域后可接受正确最优；现在必须明确UNKNOWN。
                        if domain != 'finite' and result.status == 'UNKNOWN':
                            self.check_unknown(result)
                        else:
                            self.check_optimum(model, result, expected)

    def domain_status(self, target):
        for domain, bounds in [('lower', (0, None)), ('upper', (None, -2)), ('free', (None, None))]:
            for engine in ('native', 'native_sparse'):
                if engine == 'native_sparse' and not importlib.util.find_spec('scipy'):
                    continue
                with self.subTest(domain=domain, engine=engine, target=target):
                    model = Model(f'{domain}_{target}')
                    x = model.add_var('x', lb=bounds[0], ub=bounds[1])
                    if target == 'OPTIMAL':
                        if domain == 'upper':
                            model.maximize(x+9)
                            expected = 7
                        else:
                            if domain == 'free':
                                model.add_constr(x >= -3)
                            model.minimize(x+9)
                            expected = 6 if domain == 'free' else 9
                    elif target == 'INFEASIBLE':
                        if domain == 'lower':
                            model.add_constr(x <= -1)
                        elif domain == 'upper':
                            model.add_constr(x >= 0)
                        else:
                            model.add_constr(x >= 1)
                            model.add_constr(x <= 0)
                        model.minimize(x)
                    else:
                        model.minimize(-x if domain == 'lower' else x)
                    result = self.recorded_solve(model, engine)
                    if engine == 'native_sparse' and result.status == 'UNKNOWN':
                        self.check_unknown(result)
                    elif target == 'OPTIMAL':
                        self.check_optimum(model, result, expected)
                    else:
                        self.assertEqual(result.status, target, result.termination_reason)
                        self.assertFalse(result.has_solution)

    def test_infinite_bound_does_not_imply_unbounded_objective(self):
        self.domain_status('OPTIMAL')

    def test_inconsistent_rows_not_mistaken_for_unboundedness(self):
        self.domain_status('INFEASIBLE')

    def test_genuine_unbounded_rays_not_clipped_by_artificial_bounds(self):
        self.domain_status('UNBOUNDED')


if __name__ == '__main__':
    unittest.main()
