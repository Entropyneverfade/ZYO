# Phase-I 回归：预期来自独立整数枚举，不由被测求解器生成。
"""Native feasibility recovery on frozen three-variable counterexamples."""
import unittest
import time
import importlib.util
from unittest.mock import patch

import numpy as np

import zyo


CASES = (
    ('seed-641927-084', [[-2,-2,1],[-3,0,-1],[-3,-1,1],[3,2,3]],
     [-.683173839744974,-1.6737195079758411,-.8943577495896191,5.046146379642566],
     [-3,-4,-3], 'min', -7.046146379642566),
    ('seed-883109-075', [[3,1,-1],[-2,-3,2],[-2,-3,-1],[2,3,1]],
     [3.5607573269557764,1.8579484330669556,-1.88463030703515,4.934922710240439],
     [3,-2,-1], 'max', 5.11536969296485),
)


def counterexample(case):
    # 固定历史输入；两个整数枚举维度加一个连续区间可独立精确核对。
    name, A, b, c, sense, _ = case
    model = zyo.Model(name)
    x = [model.add_var(str(i), lb=lo, ub=hi, vtype=kind)
         for i, (lo, hi, kind) in enumerate(((-2,3,'I'),(-2,3,'I'),(-1,3,'C')))]
    for row, rhs in zip(A, b):
        model.add_constr(sum(a*v for a,v in zip(row,x)) <= rhs)
    objective = sum(a*v for a,v in zip(c,x)) + 2
    (model.minimize if sense == 'min' else model.maximize)(objective)
    return model


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy sparse algebra required')
class PhaseOneRecoveryTests(unittest.TestCase):
    def test_both_remaining_milps_close_infeasible_children(self):
        for case in CASES:
            with self.subTest(case=case[0]):
                result = counterexample(case).solve('native_sparse', time_limit=2,
                                                     iteration_limit=100, node_limit=500)
                self.assertEqual(result.status, zyo.Status.OPTIMAL, result.termination_reason)
                self.assertAlmostEqual(result.objective, case[-1], places=6)
                self.assertLessEqual(abs(result.best_bound-case[-1]), 1e-6)
                self.assertLessEqual(result.primal_residual, 1e-7)
                self.assertLessEqual(result.integrality_residual, 1e-7)

    def test_phase_one_has_a_strict_elastic_point_and_maps_mixed_rows(self):
        from zyo.sparse_phase_one import build_phase_one, original_multipliers
        model = zyo.Model()
        x = model.add_var('x', lb=-2, ub=4)
        fixed = model.add_var('fixed', lb=3, ub=3)
        model.add_constr(2*x+fixed <= 1)
        model.add_constr(-x >= -1)
        model.add_constr(x+fixed == 4)
        auxiliary, mapping = build_phase_one(model,[-2,3],[4,3],time.perf_counter()+2)
        self.assertEqual(len(auxiliary.variables), 2)  # 固定量消去，只余 z 和 tau。
        self.assertEqual(len(mapping), 4)
        for z in (0., .5, 1.):
            for row in auxiliary.constraints:
                self.assertLess(sum(a*[z,2.5][i] for i,a in row.expression.terms.items())
                                +row.expression.constant, 0.)
        lam = original_multipliers(model,mapping,[-1,-2,-3,-4])
        # 三个原始尺度分别 12、6、6；等式乘子 = (-3+4)/6。
        np.testing.assert_allclose(lam, [-1/12, 2/6, 1/6])

    def test_phase_one_positive_slack_needs_original_certificate(self):
        from zyo.sparse_lp import solve_relaxation, SparseLPResult
        from zyo.options import SolveOptions
        model = zyo.Model(); x = model.add_var('x', ub=1); model.add_constr(x <= 1)
        original = SparseLPResult('NUMERICAL_ERROR', iterations=7, message='synthetic failure')
        # 正松弛和成功状态都不能证明原问题不可行；零乘子无法通过原始证书门。
        auxiliary = SparseLPResult('OPTIMAL',objective=.3,multipliers=[0.],iterations=9)
        with patch('zyo.sparse_lp._solve_relaxation',side_effect=[original,auxiliary]) as core:
            result = solve_relaxation(model,[0],[1],SolveOptions(iteration_limit=20),time.perf_counter()+2)
        self.assertEqual(result.status,'NUMERICAL_ERROR')
        self.assertEqual(core.call_count,2)
        self.assertEqual(core.call_args.args[3].iteration_limit,13)
        self.assertEqual(result.iterations,16)
        self.assertIsNone(result.x)
        self.assertIsNone(result.infeasibility_certificate)

    def test_recovery_does_not_reset_iteration_or_time_budget(self):
        from zyo.sparse_lp import solve_relaxation, SparseLPResult
        from zyo.options import SolveOptions
        model = zyo.Model(); model.add_var('x',ub=1)
        for budget, deadline, expected in ((5,time.perf_counter()+10,'NUMERICAL_ERROR'),
                                           (20,0.,'TIME_LIMIT')):
            original = SparseLPResult('NUMERICAL_ERROR',iterations=5)
            with patch('zyo.sparse_lp._solve_relaxation',return_value=original) as core:
                result = solve_relaxation(model,[0],[1],SolveOptions(iteration_limit=budget),deadline)
            self.assertEqual(core.call_count,1)
            self.assertEqual(result.status,expected)
            self.assertEqual(result.iterations,5)

    def test_auxiliary_limits_and_failures_never_close_the_node(self):
        from zyo.sparse_lp import solve_relaxation, SparseLPResult
        from zyo.options import SolveOptions
        model = zyo.Model(); x = model.add_var('x',ub=1); model.add_constr(x<=1)
        for status in ('TIME_LIMIT','ITERATION_LIMIT','NUMERICAL_ERROR','INFEASIBLE'):
            with self.subTest(auxiliary_status=status):
                original = SparseLPResult('NUMERICAL_ERROR',iterations=5)
                auxiliary = SparseLPResult(status,iterations=4,multipliers=[-1.])
                with patch('zyo.sparse_lp._solve_relaxation',side_effect=[original,auxiliary]) as core:
                    result = solve_relaxation(model,[0],[1],SolveOptions(iteration_limit=20),time.perf_counter()+2)
                self.assertEqual(core.call_count,2)  # 辅助问题失败也不能递归。
                self.assertEqual(result.status,status if 'LIMIT' in status else 'NUMERICAL_ERROR')
                self.assertEqual(result.iterations,9)
                self.assertIsNone(result.infeasibility_certificate)

    def test_original_limit_does_not_start_recovery(self):
        from zyo.sparse_lp import solve_relaxation, SparseLPResult
        from zyo.options import SolveOptions
        model = zyo.Model(); model.add_var('x',ub=1)
        for status in ('TIME_LIMIT','ITERATION_LIMIT'):
            with patch('zyo.sparse_lp._solve_relaxation',return_value=SparseLPResult(status)) as core:
                result = solve_relaxation(model,[0],[1],SolveOptions(),time.perf_counter()+2)
            self.assertEqual(core.call_count,1)
            self.assertEqual(result.status,status)

    def test_certificate_work_crossing_deadline_preserves_time_limit(self):
        from zyo.sparse_lp import _recover_infeasibility, SparseLPResult
        from zyo.options import SolveOptions
        from zyo.sparse_certificate import box_infeasibility
        model = zyo.Model(); x = model.add_var('x',ub=2); y = model.add_var('y',ub=2)
        model.add_constr(x+y>=3); model.add_constr(x+y<=1)
        for mode in ('valid','rejected','arithmetic_error'):
            with self.subTest(check=mode):
                clock = [0.]
                def check_after_time(*args,**kwargs):
                    clock[0] = 2.
                    if mode == 'arithmetic_error':
                        raise ArithmeticError('late numerical rejection')
                    return box_infeasibility(*args,**kwargs)
                phase = SparseLPResult('OPTIMAL',multipliers=[-1.,-1.] if mode=='valid' else [0.,0.])
                with patch('zyo.sparse_lp.time.perf_counter',side_effect=lambda: clock[0]), \
                     patch('zyo.sparse_lp._solve_relaxation',return_value=phase), \
                     patch('zyo.sparse_lp.box_infeasibility',side_effect=check_after_time):
                    result = _recover_infeasibility(model,[0,0],[2,2],SolveOptions(),1.,
                                                   SparseLPResult('NUMERICAL_ERROR'))
                self.assertEqual(result.status,'TIME_LIMIT')
                self.assertIsNone(result.infeasibility_certificate)
                self.assertEqual(result.feasibility_recovery['runtime_seconds'],2.)

    def test_scaling_underflow_and_invalid_box_are_rejected(self):
        from zyo.sparse_phase_one import build_phase_one, original_multipliers
        model = zyo.Model(); x = model.add_var('x',ub=1)
        model.add_constr(1e-300*x <= 1e300)
        with self.assertRaises(FloatingPointError):
            build_phase_one(model,[0],[1],time.perf_counter()+2)
        for lo,hi in (([0],[float('inf')]),([1],[0]),([],[])):
            with self.assertRaises(ValueError):
                build_phase_one(model,lo,hi,time.perf_counter()+2)
        with self.assertRaises(FloatingPointError):
            original_multipliers(model,[(0,1e-300)],[-1e-300])
        with self.assertRaises(ValueError):
            original_multipliers(model,[(0,1.)],[])

    def test_feasible_numerical_failure_is_not_mistaken_for_infeasibility(self):
        from zyo.sparse_lp import _recover_infeasibility, SparseLPResult
        from zyo.options import SolveOptions
        model = zyo.Model(); x = model.add_var('x',ub=1); model.add_constr(x>=.3)
        # 即使原目标尺度极大，Phase-I 仍只看可行域；不把辅助可行解当成原最优解。
        model.maximize(1e200*x+1e200)
        result = _recover_infeasibility(model,[0],[1],SolveOptions(iteration_limit=100),
                                        time.perf_counter()+2,SparseLPResult('NUMERICAL_ERROR'))
        self.assertEqual(result.status,'NUMERICAL_ERROR')
        self.assertFalse(result.feasibility_recovery['certificate_verified'])
        self.assertIsNone(result.x)


if __name__ == '__main__':
    unittest.main()
