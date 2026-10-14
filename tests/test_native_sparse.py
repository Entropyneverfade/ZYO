# 回归测试：用解析模型与导入封锁验证自主稀疏路径及其有限盒限制。
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import time
import unittest

import numpy as np
import zyo


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy sparse linear algebra required')
class NativeSparseTests(unittest.TestCase):
    def test_public_analytic_original_direction_refinement(self):
        # 独立小矩阵给出精确方向；内部法方程残差不能替代原式方向门。
        from scipy.sparse import csc_matrix
        from zyo.sparse_lp import _normal_solver, _refine_original_newton_direction
        A=csc_matrix([[1.,0.,1.],[0.,1.,1.]])
        x=np.array([1e-10,1e-10,1.]); s=np.ones(3)
        rp=np.array([1.,-1.]); zeros=np.zeros(3)
        normal,_=_normal_solver(A,x/s)
        initial=normal(rp)
        dx,dy,ds,corrections=_refine_original_newton_direction(
            A,x,s,rp,zeros,zeros,initial,normal,max_corrections=3)
        self.assertGreaterEqual(corrections,1)
        self.assertLessEqual(float(np.max(np.abs(A@dx-rp))),2e-9)
        self.assertLessEqual(float(np.max(np.abs(s*dx+x*ds))),1e-9)

    def test_public_synthetic_full_day_sparse_uc(self):
        # 固定种子教学例：验收原生全周期整数调度及独立物理/成本重算。
        from zyo.validation import evaluate_objective, validate_candidate
        from zyo_power.unit_commitment import build_uc_model, extract_uc_trace, validate_uc
        source=Path(__file__).resolve().parents[1]/'examples'/'synthetic_uc_tight_reserve_20261014.json'
        case=json.loads(source.read_text(encoding='utf-8'))
        model,mapping=build_uc_model(case)
        self.assertEqual((len(model.variables),len(model.constraints)),(264,480))
        result=model.solve('native_sparse',options=zyo.SolveOptions(
            time_limit=10,threads=1,seed=20260908,mip_gap=0,
            feasibility_tol=1e-7,integrality_tol=1e-7))
        self.assertEqual(result.status,zyo.Status.OPTIMAL,result.termination_reason)
        self.assertFalse(result.metadata['fallback_used'])
        self.assertTrue(validate_candidate(model,result.values,1e-7,1e-7).is_feasible)
        physical=validate_uc(case,extract_uc_trace(mapping,result.values))
        self.assertTrue(physical['physical_pass'])
        self.assertEqual(physical['ens_mwh'],0.)
        self.assertAlmostEqual(evaluate_objective(model,result.values),39856.801,delta=1e-4)
        self.assertAlmostEqual(physical['cost'],result.objective,delta=1e-4)
        self.assertLessEqual(result.objective-result.best_bound,1e-4)

    def test_336_hour_sparse_newton_stability_regression(self):
        from zyo_power.examples import synthetic_case
        from zyo_power.optimization import optimize_dispatch
        from zyo_power.validation import validate_dispatch
        case=synthetic_case(336)[0]
        result=optimize_dispatch(case,engine='native_sparse',options=zyo.SolveOptions(time_limit=5,node_limit=10))
        self.assertEqual(result['status'],'OPTIMAL',result['metadata']['solver_result']['termination_reason'])
        self.assertTrue(validate_dispatch(case,result)['physical_pass'])

    def test_zero_cost_degenerate_teaching_milp_has_checked_zero_lower_bound(self):
        from zyo_power.examples import teaching_case
        from zyo_power.optimization import optimize_dispatch
        from zyo_power.validation import validate_dispatch
        result=optimize_dispatch(teaching_case(),engine='native_sparse',options=zyo.SolveOptions(time_limit=5,node_limit=100))
        self.assertEqual(result['status'],'OPTIMAL',result['metadata']['solver_result']['termination_reason'])
        self.assertLess(abs(result['metadata']['solver_result']['objective']),1e-7)
        self.assertTrue(validate_dispatch(teaching_case(),result)['physical_pass'])

    def test_public_sparse_binary_search_matches_hand_enumeration(self):
        m=zyo.Model()
        x=m.add_var('x',vtype='B')
        y=m.add_var('y',vtype='B')
        m.add_constr(2*x+2*y<=3)
        m.maximize(x+y)
        r=m.solve('NATIVE_SPARSE',time_limit=5)
        self.assertEqual(r.status,zyo.Status.OPTIMAL,r.termination_reason)
        self.assertEqual(r.objective,1.)
        self.assertGreaterEqual(r.best_bound,1.)
        self.assertLess(r.best_bound-1.,1e-7)
        self.assertGreater(r.node_count,1)
        self.assertEqual(r.metadata['backend_kind'],'self-developed')
        self.assertFalse(r.metadata['fallback_used'])

    def test_limits_and_missing_finite_boxes_never_report_optimal(self):
        m=zyo.Model()
        x=m.add_var('x',vtype='B')
        m.maximize(x)
        for kwargs,status in [({'time_limit':0},zyo.Status.TIME_LIMIT),
                               ({'node_limit':0},zyo.Status.NODE_LIMIT),
                               ({'iteration_limit':0},zyo.Status.ITERATION_LIMIT)]:
            self.assertEqual(m.solve('native_sparse',**kwargs).status,status)
        # 真正无界的模型：y 只有上界 0（必须显式写 lb=None，否则默认下界为 0）
        # 且目标为 min y，可令 y -> -inf。域解析路径必须报 UNBOUNDED（或至少未决），
        # 绝不报告最优，也绝不返回被人造上界裁剪出来的候选。
        unbounded_box=zyo.Model()
        y=unbounded_box.add_var('y',lb=None,ub=0)
        unbounded_box.minimize(y)
        r=unbounded_box.solve('native_sparse')
        self.assertNotEqual(r.status,zyo.Status.OPTIMAL,r.termination_reason)
        self.assertFalse(r.has_solution)
        # 有下界且目标在界上取到有限最优的模型仍须给出最优，而不是含糊未决。
        bounded=zyo.Model()
        z=bounded.add_var('z',lb=0,ub=None)
        bounded.minimize(z)
        r2=bounded.solve('native_sparse')
        self.assertEqual(r2.status,zyo.Status.OPTIMAL,r2.termination_reason)
        self.assertAlmostEqual(r2.objective,0.0,places=7)

    def test_fresh_process_blocks_every_external_optimizer_import(self):
        code='''
import sys
sys.path.insert(0,sys.argv[1])
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('scipy.optimize','highspy','gurobipy','coptpy')):
        raise AssertionError('Forbidden optimizer import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import zyo
m=zyo.Model(); x=m.add_var('x',ub=2); m.minimize(-x)
r=m.solve('native_sparse')
assert r.status==zyo.Status.OPTIMAL, r
assert abs(r.objective+2)<1e-7, r
assert 'lzyopt.solver' not in sys.modules, 'Sparse path loaded the legacy mixed-engine module'
print('SELF_DEVELOPED_SOLVE_WITH_OPTIMIZER_IMPORTS_BLOCKED')
'''
        r=subprocess.run([sys.executable,'-I','-c',code,str(Path(zyo.__file__).resolve().parent.parent)],text=True,capture_output=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('SELF_DEVELOPED_SOLVE',r.stdout)

    def test_roundoff_cannot_prove_infeasibility_and_fixed_objective_includes_constant(self):
        from zyo.sparse_lp import solve_relaxation
        m=zyo.Model()
        a=m.add_var('a',lb=1,ub=1)
        b=m.add_var('b',lb=1+2**-52,ub=1+2**-52)
        c=m.add_var('c',lb=1,ub=1)
        m.add_constr(1e10*a-1e10*b+2.1e-6*c<=0)
        r=solve_relaxation(m,np.array([v.lb for v in m.variables]),np.array([v.ub for v in m.variables]),zyo.SolveOptions(),time.perf_counter()+2)
        self.assertNotEqual(r.status,'INFEASIBLE')
        fixed=zyo.Model()
        x=fixed.add_var('x',lb=1,ub=1); y=fixed.add_var('y',lb=1,ub=1)
        fixed.minimize(-1e16*x-y+1e16)
        r=solve_relaxation(fixed,np.ones(2),np.ones(2),zyo.SolveOptions(),time.perf_counter()+2)
        self.assertEqual(r.objective,-1.)
        self.assertLessEqual(r.bound,-1.)

    def test_overflowing_preprocess_is_numerical_failure_not_an_exception(self):
        from zyo.sparse_lp import solve_relaxation
        m=zyo.Model()
        x=m.add_var('x',lb=1e200,ub=1e200); y=m.add_var('y',lb=1e200,ub=1e200)
        m.add_constr(1e200*x-1e200*y==0)
        r=solve_relaxation(m,np.full(2,1e200),np.full(2,1e200),zyo.SolveOptions(),time.perf_counter()+2)
        self.assertEqual(r.status,'NUMERICAL_ERROR')
        m=zyo.Model()
        x=m.add_var('fixed_overflow',lb=1e200,ub=1e200)
        m.minimize(1e200*x)
        self.assertEqual(m.solve('native_sparse').status,zyo.Status.NUMERICAL_ERROR)

    def test_box_bound_is_a_lower_bound_not_primal_objective(self):
        from zyo.sparse_certificate import box_bound
        m=zyo.Model()
        x=m.add_var('x',ub=4)
        m.add_constr(x>=1)
        m.minimize(2*x+3)
        self.assertAlmostEqual(box_bound(m,np.array([1.]))['bound'],4.,places=10)
        self.assertLessEqual(box_bound(m,np.array([2.]))['bound'],5.)
        with self.assertRaises(ValueError):
            box_bound(m,np.array([-1.]))
        with self.assertRaises(ValueError):
            box_bound(m,np.array([float('nan')]))

    def test_sparse_lp_handles_negative_bounds_equalities_max_and_constant(self):
        from zyo.sparse_lp import solve_relaxation
        m=zyo.Model()
        x=m.add_var('x',lb=-2,ub=3)
        y=m.add_var('y',ub=4)
        fixed=m.add_var('fixed',lb=2,ub=2)
        m.add_constr(x+y==4)
        m.add_constr(x-y>=-2)
        m.maximize(3*x+2*y+fixed+7)
        r=solve_relaxation(m,np.array([v.lb for v in m.variables]),
                          np.array([v.ub for v in m.variables]),zyo.SolveOptions(),time.perf_counter()+5)
        self.assertEqual(r.status,'OPTIMAL',r.message)
        self.assertAlmostEqual(r.objective,-20.,places=6)
        self.assertLessEqual(r.bound,-20.+1e-8)
        self.assertLess(-20.-r.bound,1e-6)
        np.testing.assert_allclose(r.x,[3,1,2],atol=1e-7)


if __name__=='__main__':
    unittest.main()
