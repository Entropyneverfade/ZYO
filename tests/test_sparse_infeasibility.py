# 回归测试：覆盖原始盒不可行证书、近可行反例、下溢拒绝和冻结开发失败集。
"""Independent hand examples and frozen development regressions, not held-out."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import zyo


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy sparse algebra required')
class SparseInfeasibilityTests(unittest.TestCase):
    def test_joint_constraints_have_a_certificate_without_newton_convergence(self):
        m=zyo.Model()
        x=m.add_var('x',ub=2); y=m.add_var('y',ub=2)
        m.add_constr(x+y>=3); m.add_constr(x+y<=1)
        m.minimize(x-y)
        result=m.solve('native_sparse',time_limit=2,iteration_limit=100)
        self.assertEqual(result.status,zyo.Status.INFEASIBLE,result.termination_reason)
        self.assertFalse(result.has_solution)
        proof=result.metadata['root_relaxation']['infeasibility_certificate']
        self.assertTrue(proof['verified'])
        self.assertGreater(proof['margin'],proof['tolerance_allowance'])

    def test_infeasible_child_is_pruned_and_the_feasible_sibling_is_solved(self):
        # LP optimum x=.4, y=.6. x=0 is jointly infeasible, although neither
        # constraint alone excludes that box. Integer optimum is x=1.
        m=zyo.Model()
        x=m.add_var('x',ub=2,vtype='I'); y=m.add_var('y',lb=-5,ub=5)
        m.add_constr(y>=x+.2); m.add_constr(y<=2*x-.2)
        m.minimize(x)
        result=m.solve('native_sparse',time_limit=2,iteration_limit=100)
        self.assertEqual(result.status,zyo.Status.OPTIMAL,result.termination_reason)
        self.assertAlmostEqual(result.objective,1.,places=7)
        self.assertLessEqual(result.best_bound,1.)
        self.assertLess(1.-result.best_bound,1e-7)
        proofs=[n['infeasibility_certificate'] for n in result.metadata['node_log']
                if n.get('infeasibility_certificate')]
        self.assertTrue(proofs)
        self.assertTrue(all(p['verified'] for p in proofs))

    def test_certificate_uses_node_box_and_ignores_objective_and_sense(self):
        from zyo.sparse_certificate import box_infeasibility
        m=zyo.Model()
        x=m.add_var('x',ub=2); y=m.add_var('y',lb=-5,ub=5)
        m.add_constr(y>=x+.2); m.add_constr(y<=2*x-.2)
        m.maximize(1e200*x-1e200*y+1e200)
        self.assertFalse(box_infeasibility(m,[1.,-1.])['verified'])
        for factor in (1.,1e200,1e-200):
            proof=box_infeasibility(m,np.array([1.,-1.])*factor,lower=[0,-5],upper=[0,5])
            self.assertTrue(proof['verified'])
            self.assertAlmostEqual(proof['raw_margin'],.4,places=12)

    def test_feasibility_tolerance_is_not_reinterpreted_as_infeasibility(self):
        from zyo.sparse_certificate import box_infeasibility
        m=zyo.Model()
        x=m.add_var('x',ub=2); y=m.add_var('y',ub=2)
        m.add_constr(x+y>=1+1e-9); m.add_constr(x+y<=1)
        proof=box_infeasibility(m,[1,-1],feasibility_tol=1e-7)
        self.assertFalse(proof['verified'])
        self.assertEqual(proof['tolerance_allowance'],2e-7)
        result=m.solve('native_sparse',time_limit=1,iteration_limit=30)
        self.assertNotEqual(result.status,zyo.Status.INFEASIBLE)

    def test_invalid_certificate_data_fail_closed(self):
        from zyo.sparse_certificate import box_infeasibility
        m=zyo.Model(); x=m.add_var('x',ub=2); m.add_constr(x>=1)
        for lam in ([-1.],[float('nan')],[float('inf')],[],[1,1]):
            with self.assertRaises(ValueError): box_infeasibility(m,lam)
        for tol in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): box_infeasibility(m,[1],feasibility_tol=tol)
        with self.assertRaises(ValueError): box_infeasibility(m,[1],lower=[2],upper=[1])
        self.assertFalse(box_infeasibility(m,[0.])['verified'])

    def test_redundant_feasible_equalities_are_not_pruned(self):
        m=zyo.Model(); x=m.add_var('x',ub=1); y=m.add_var('y',ub=1)
        m.add_constr(x+y==1); m.add_constr(2*x+2*y==2)
        m.maximize(x)
        result=m.solve('native_sparse',time_limit=2)
        self.assertEqual(result.status,zyo.Status.OPTIMAL,result.termination_reason)
        self.assertAlmostEqual(result.objective,1.,places=7)

    def test_underflow_cannot_certify_a_feasible_extreme_box(self):
        from fractions import Fraction as F
        from zyo.sparse_certificate import box_infeasibility
        m=zyo.Model()
        xs=[m.add_var(str(i),ub=1e308) for i in range(300)]
        m.add_constr(sum(1e-4*x for x in xs)>=2e306)
        m.add_constr(xs[0]-xs[0]==0)
        # All upper endpoints satisfy the row in exact stored-float arithmetic.
        self.assertGreater(300*F(1e-4)*F(1e308)-F(2e306),0)
        for tol in (0.,1e-16,1e-7):
            with self.subTest(tolerance=tol):
                with self.assertRaises(FloatingPointError):
                    box_infeasibility(m,[1e-320,1.],feasibility_tol=tol)

    def test_witness_normalization_underflow_fails_closed(self):
        from zyo.sparse_certificate import box_infeasibility
        m=zyo.Model(); x=m.add_var('x',ub=2)
        m.add_constr(x>=1); m.add_constr(x<=2)
        with self.assertRaises(FloatingPointError):
            box_infeasibility(m,[1e-300,-1e300])

    def test_original_box_bound_rejects_underflow_too(self):
        from zyo.sparse_certificate import box_bound
        m=zyo.Model()
        xs=[m.add_var(str(i),ub=1e308) for i in range(300)]
        m.add_constr(sum(1e-4*x for x in xs)>=2e306)
        m.add_constr(xs[0]-xs[0]==0)
        m.minimize(0*xs[0])  # Every feasible point has objective exactly zero.
        with self.assertRaises(FloatingPointError):
            box_bound(m,[1e-320,1.])

    def test_all_sixty_frozen_development_models_match_independent_enumeration(self):
        fixture=json.loads((Path(__file__).parent/'data/native_sparse_60.json').read_text(encoding='utf-8'))
        for case in fixture['cases']:
            with self.subTest(case=case['case']):
                m=zyo.Model()
                x=[m.add_var(str(i),lb=b[0],ub=b[1],vtype=k)
                   for i,(b,k) in enumerate(zip(case['bounds'],case['kinds']))]
                for a,b in zip(case['A'],case['b']): m.add_constr(sum(v*w for v,w in zip(a,x))<=b)
                objective=sum(v*w for v,w in zip(case['c'],x))+case['objective_constant']
                (m.minimize if case['sense']=='min' else m.maximize)(objective)
                result=m.solve('native_sparse',time_limit=2,iteration_limit=100,node_limit=500)
                self.assertEqual(result.status,zyo.Status.OPTIMAL,result.termination_reason)
                self.assertLessEqual(abs(result.objective-case['expected_objective']),1e-6)
                direction=1 if case['sense']=='min' else -1
                self.assertLessEqual(direction*result.best_bound,direction*case['expected_objective']+1e-6)
                self.assertLessEqual(result.primal_residual,1e-7)
                self.assertLessEqual(result.integrality_residual,1e-7)
