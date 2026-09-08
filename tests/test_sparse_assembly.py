# 增广矩阵装配的独立块等式测试；用手工 NumPy 块定义而非被测装配器产生预期。
import unittest
import numpy as np
from scipy.sparse import csc_matrix
from zyo.sparse_lp import _augmented_matrix
import zyo.sparse_lp as sparse_lp


class SparseAssemblyTests(unittest.TestCase):
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
