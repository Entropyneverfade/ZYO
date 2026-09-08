# 增广矩阵装配的独立块等式测试；用手工 NumPy 块定义而非被测装配器产生预期。
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csc_matrix
from zyo.sparse_lp import _augmented_matrix
import zyo.sparse_lp as sparse_lp


class SparseAssemblyTests(unittest.TestCase):
    def assert_block(self, actual, matrix):
        # 独立定义三个非零块，预期不经过旧/新装配器。
        m,n=matrix.shape
        expected=np.zeros((m+n,m+n))
        expected[:n,:n]=-np.eye(n)
        expected[:n,n:]=matrix.toarray().T
        expected[n:,:n]=matrix.toarray()
        np.testing.assert_array_equal(actual.toarray(),expected)
        self.assertTrue(actual.has_canonical_format)

    def test_cached_structure_refreshes_values_and_isolates_storage(self):
        self.assertTrue(hasattr(sparse_lp,'_AugmentedAssembly'))
        assemble=sparse_lp._AugmentedAssembly()
        matrix=csc_matrix([[2.,0.,-3.],[0.,4.,5.]])
        first=assemble(matrix); before=first.toarray().copy()
        matrix.data*=2
        second=assemble(matrix)
        self.assert_block(second,matrix)
        np.testing.assert_array_equal(first.toarray(),before)
        # 修改已返回的结构/数值，不能污染缓存或下次结果。
        second.data[:]=0; second.indices[:]=0; second.indptr[:]=0
        self.assert_block(assemble(matrix),matrix)
        self.assertEqual((assemble.builds,assemble.reuses),(1,2))

    def test_cached_structure_invalidates_equal_nnz_and_shape_changes(self):
        self.assertTrue(hasattr(sparse_lp,'_AugmentedAssembly'))
        assemble=sparse_lp._AugmentedAssembly()
        # 相同nnz：先改变行索引，再改变列指针，最后改变形状，均须重建。
        cases=[[[1.,0.],[0.,2.]],[[0.,1.],[2.,0.]],[[1.,2.],[0.,0.]],
               [[1.,2.,0.]],np.zeros((0,3)),np.zeros((3,0))]
        for i,raw in enumerate(cases,1):
            matrix=csc_matrix(raw)
            self.assert_block(assemble(matrix),matrix)
            self.assertEqual(assemble.builds,i)
        self.assertEqual(assemble.reuses,0)

    def test_cache_handles_duplicates_explicit_zeros_and_zero_elimination(self):
        self.assertTrue(hasattr(sparse_lp,'_AugmentedAssembly'))
        assemble=sparse_lp._AugmentedAssembly()
        matrix=csc_matrix((np.array([2.,3.,-1.,0.]),np.array([1,0,1,0]),
                           np.array([0,3,4])),shape=(2,2))
        before=tuple(a.copy() for a in (matrix.data,matrix.indices,matrix.indptr))
        self.assert_block(assemble(matrix),matrix)
        self.assert_block(assemble(matrix),matrix)
        for a,b in zip((matrix.data,matrix.indices,matrix.indptr),before):
            np.testing.assert_array_equal(a,b)
        self.assertEqual((assemble.builds,assemble.reuses),(1,1))
        matrix.sum_duplicates();matrix.eliminate_zeros()
        self.assert_block(assemble(matrix),matrix)
        self.assertEqual(assemble.builds,2)

    def test_one_cache_per_lp_and_fresh_lp_after_rhs_and_bound_changes(self):
        import zyo
        self.assertTrue(hasattr(sparse_lp,'_AugmentedAssembly'))
        factory=sparse_lp._AugmentedAssembly; instances=[]
        def capture():
            value=factory();instances.append(value);return value
        with patch.object(sparse_lp,'_AugmentedAssembly',side_effect=capture):
            for rhs,upper in ((1.,3.),(2.,5.)):
                model=zyo.Model('different_rhs_and_box')
                x=model.add_var(name='x',ub=upper)
                model.add_constr(x>=rhs);model.minimize(x)
                result=model.solve('native_sparse')
                self.assertEqual(result.status,zyo.Status.OPTIMAL)
                self.assertAlmostEqual(result.objective,rhs,places=7)
        self.assertEqual(len(instances),2)
        self.assertIsNot(instances[0],instances[1])
        self.assertTrue(all(a.builds==1 and a.reuses>0 for a in instances))

    def test_cached_newton_direction_matches_original_equations(self):
        self.assertTrue(hasattr(sparse_lp,'_AugmentedAssembly'))
        assemble=sparse_lp._AugmentedAssembly()
        A=csc_matrix([[1.,0.],[0.,1.],[1.,2.]])
        for x,s in ((np.array([1.,3.]),np.array([2.,4.])),
                    (np.array([2.,1.]),np.array([3.,2.]))):
            # 已知dx=(.25,-.5), dy=0, ds=(4,7)，不同缩放复用结构。
            rp=np.array([.25,-.5,-.75]);rd=np.array([4.,7.])
            rc=s*np.array([.25,-.5])+x*rd
            dx,dy,ds=sparse_lp._augmented_solver(A,x,s,rp,rd,assembly=assemble)(rc)
            np.testing.assert_allclose(dx,[.25,-.5],atol=1e-9,rtol=0)
            self.assertLessEqual(float(max(abs(A@dx-rp))),2e-9)
            self.assertLessEqual(float(max(abs(A.T@dy+ds-rd))),8e-9)
            self.assertLessEqual(float(max(abs(s*dx+x*ds-rc))),1e-9*(1+max(abs(rc))))
        self.assertEqual(assemble.builds,1)
        self.assertGreaterEqual(assemble.reuses,1)

    def test_row_scaling_promotes_before_multiplication(self):
        # 整数乘小数必须保留小数；float32 极小值乘 float64 后仍是非零，不能先下溢再删零。
        cases=[(csc_matrix([[2,-3]],dtype=np.int64),np.array([.5]),np.array([[1.,-1.5]])),
               (csc_matrix([[1e-38]],dtype=np.float32),np.array([1e-10]),
                np.array([[float(np.float32(1e-38))*1e-10]]))]
        for matrix,scale,expected in cases:
            with self.subTest(dtype=matrix.dtype):
                before=tuple(a.copy() for a in (matrix.data,matrix.indices,matrix.indptr))
                try:
                    actual=sparse_lp._row_scaled_csc(matrix,scale)
                except (TypeError,ValueError) as error:
                    self.fail(f'合法的混合精度行缩放失败：{error}')
                self.assertEqual(actual.dtype,np.dtype('float64'))
                self.assertEqual(actual.nnz,np.count_nonzero(expected))
                np.testing.assert_array_equal(actual.toarray(),expected)
                for value,original in zip((matrix.data,matrix.indices,matrix.indptr),before):
                    np.testing.assert_array_equal(value,original)

    def test_row_scaling_matches_independent_dense_values_without_mutation(self):
        # 逐行乘法的期望由手工稠密数组定义，含重复索引、显式零和空矩形。
        self.assertTrue(hasattr(sparse_lp,'_row_scaled_csc'))
        duplicate=csc_matrix((np.array([2.,3.,-1.,0.]),np.array([1,0,1,0]),
                              np.array([0,3,4])),shape=(2,2))
        for C in (duplicate,csc_matrix([[2.,0.,-3.],[0.,4.,5.]]),
                  csc_matrix((0,3)),csc_matrix((2,0))):
            scale=np.arange(1,C.shape[0]+1,dtype=float)*.25
            before=tuple(v.copy() for v in (C.data,C.indices,C.indptr))
            expected=scale[:,None]*C.toarray()
            actual=sparse_lp._row_scaled_csc(C,scale)
            np.testing.assert_array_equal(actual.toarray(),expected)
            self.assertEqual(actual.format,'csc')
            self.assertTrue(actual.has_canonical_format)
            self.assertFalse(np.any(actual.data==0))
            for value,original in zip((C.data,C.indices,C.indptr),before):
                np.testing.assert_array_equal(value,original)

    def test_noncanonical_duplicates_are_merged_without_mutating_input(self):
        # 同一列乱序且含重复行；装配器须复制后规范化，不能改动调用方的CSC存储。
        C=csc_matrix((np.array([2.,3.,-1.]),np.array([1,0,1]),np.array([0,3,3])),shape=(2,2))
        self.assertFalse(C.has_canonical_format)
        before=tuple(v.copy() for v in (C.data,C.indices,C.indptr))
        K=_augmented_matrix(C)
        np.testing.assert_array_equal(K[2:,:2].toarray(),[[3.,0.],[1.,0.]])
        self.assertTrue(K.has_canonical_format)
        for value,expected in zip((C.data,C.indices,C.indptr),before):
            np.testing.assert_array_equal(value,expected)

    def test_rectangular_empty_and_explicit_zero_blocks(self):
        for raw in ([[2.,0.,-3.],[0.,4.,5.]], [[0.]], np.zeros((0,3)), np.zeros((2,0))):
            C = csc_matrix(raw)
            before = C.copy()
            K = _augmented_matrix(C)
            m,n = C.shape
            expected = np.zeros((m+n,m+n))
            expected[:n,:n] = -np.eye(n)
            expected[:n,n:] = C.toarray().T
            expected[n:,:n] = C.toarray()
            np.testing.assert_array_equal(K.toarray(),expected)
            np.testing.assert_array_equal(C.toarray(),before.toarray())
            self.assertTrue(K.has_sorted_indices)
            self.assertEqual(K.format,'csc')

    def test_sparse_values_and_input_storage_are_preserved(self):
        # 非正方形与显式零不应错位；直接比较每个输入存储数组，防止偷偷改写矩阵。
        C = csc_matrix((np.array([3.,0.,-2.]), np.array([1,0,1]), np.array([0,1,1,3])),shape=(2,3))
        before = tuple(x.copy() for x in (C.data,C.indices,C.indptr))
        K = _augmented_matrix(C)
        np.testing.assert_array_equal(K[:3,3:].toarray(),C.toarray().T)
        np.testing.assert_array_equal(K[3:,:3].toarray(),C.toarray())
        for value,expected in zip((C.data,C.indices,C.indptr),before):
            np.testing.assert_array_equal(value,expected)


if __name__=='__main__': unittest.main()
