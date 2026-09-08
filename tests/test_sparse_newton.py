# 秩亏牛顿系统的独立反例；手算解与对偶界不从被测内核生成，不调用外部优化器。
"""Independent regressions for rank-deficient native Newton systems."""
from fractions import Fraction
import importlib.util
import math
import unittest

import numpy as np
import zyo


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy sparse linear algebra required')
class SparseNewtonTests(unittest.TestCase):
    def test_assignment_with_redundant_equalities_reaches_exact_optimum_88(self):
        # 可行点(0,1),(1,2),(2,0)成本88；行势(0,9,29)、列势(7,8,35)给出下界88。
        costs=[[7,8,69],[23,17,44],[36,71,70]]
        u=[0,9,29]; v=[7,8,35]
        self.assertEqual(sum(u)+sum(v),88)
        self.assertTrue(all(u[i]+v[j]<=costs[i][j] for i in range(3) for j in range(3)))
        m=zyo.Model('independent_assignment_88')
        x=m.add_vars(((i,j) for i in range(3) for j in range(3)),name='x',ub=1)
        for i in range(3):
            m.add_constr(zyo.quicksum(x[i,j] for j in range(3))==1)
        for j in range(3):
            m.add_constr(zyo.quicksum(x[i,j] for i in range(3))==1)
        m.minimize(zyo.quicksum(costs[i][j]*x[i,j] for i in range(3) for j in range(3)))
        r=m.solve('native_sparse',time_limit=10,iteration_limit=100000,mip_gap=0,
                  feasibility_tol=1e-7,integrality_tol=1e-7,objective_tol=1e-8)
        self.assertEqual(r.status,zyo.Status.OPTIMAL,r.termination_reason)
        values=np.array([[r.values[x[i,j].name] for j in range(3)] for i in range(3)])
        self.assertLessEqual(max(np.max(abs(values.sum(axis=0)-1)),np.max(abs(values.sum(axis=1)-1))),1e-7)
        self.assertGreaterEqual(float(values.min()),-1e-7)
        self.assertLessEqual(float(values.max()),1+1e-7)
        objective=math.fsum(costs[i][j]*values[i,j] for i in range(3) for j in range(3))
        self.assertLess(abs(objective-88),1e-7)
        self.assertLessEqual(r.best_bound,88)
        self.assertLess(88-r.best_bound,1e-7)
        # Fraction用原系数重算该次乘子的盒下界，防止把可行点目标充当对偶界。
        lam=[Fraction(float(a)) for a in r.metadata['root_relaxation']['multipliers']]
        raw=sum(lam)+sum(min(Fraction(0),Fraction(costs[i][j])-lam[i]-lam[3+j]) for i in range(3) for j in range(3))
        self.assertLessEqual(Fraction(r.best_bound),raw)
        self.assertFalse(r.metadata['fallback_used'])

    def test_consistent_dependent_rows_preserve_original_newton_equations(self):
        from scipy.sparse import csc_matrix
        from zyo.sparse_lp import _augmented_solver
        # 第三行是前两行的非平凡组合，不是运输专用的全1冗余关系。
        A=csc_matrix([[1.,0.],[0.,1.],[1.,2.]])
        x=np.array([1.,3.]); s=np.array([2.,4.])
        rp=np.array([.25,-.5,-.75]); rd=np.array([4.,7.]); rc=np.array([3.5,13.])
        dx,dy,ds=_augmented_solver(A,x,s,rp,rd)(rc)
        self.assertLessEqual(float(np.max(abs(A@dx-rp))),1e-9*(1+max(abs(rp))))
        self.assertLessEqual(float(np.max(abs(A.T@dy+ds-rd))),1e-9*(1+max(abs(rd))))
        self.assertLessEqual(float(np.max(abs(s*dx+x*ds-rc))),1e-9*(1+max(abs(rc))))
        np.testing.assert_allclose(dx,[.25,-.5],rtol=0,atol=1e-9)
        self.assertLess(float(np.max(abs(dy))),10.)

    def test_inconsistent_dependent_rows_cannot_pass_by_regularization(self):
        from scipy.sparse import csc_matrix
        from zyo.sparse_lp import _augmented_solver
        # 两个相同左端的方向方程却要求0和1：正则化系统可逆不等于原系统可解。
        A=csc_matrix([[1.,2.],[1.,2.]])
        with self.assertRaises((ArithmeticError,RuntimeError)):
            _augmented_solver(A,np.ones(2),np.ones(2),np.array([0.,1.]),np.zeros(2))(np.zeros(2))

    def test_nearly_dependent_full_rank_direction_retains_verified_alternative(self):
        from scipy.sparse import csc_matrix
        from zyo.sparse_lp import _augmented_solver
        # A仍满秩；主元风险筛选会选中稳定化，但该分解可能不能消除足够的扰动。
        # 必须保留能通过原方程检查的未正则化方向，而非将启发式当成精确秩判定。
        A=csc_matrix([[1.,0.],[1.,1e-8]])
        rp=np.array([.25,.25-5e-9]);rd=np.array([1.,2.]);rc=np.zeros(2)
        statistics={}
        dx,dy,ds=_augmented_solver(A,np.ones(2),np.ones(2),rp,rd,statistics)(rc)
        self.assertLessEqual(float(np.max(abs(A@dx-rp))),1e-9*(1+max(abs(rp))))
        self.assertLessEqual(float(np.max(abs(A.T@dy+ds-rd))),1e-9*(1+max(abs(rd))))
        self.assertLessEqual(float(np.max(abs(dx+ds-rc))),1e-9)
        self.assertTrue(statistics['unregularized_fallback'])


if __name__=='__main__':
    unittest.main()
