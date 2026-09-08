# 增广矩阵装配的独立块等式测试；用手工 NumPy 块定义而非被测装配器产生预期。
import unittest
import numpy as np
from scipy.sparse import csc_matrix
from zyo.sparse_lp import _augmented_matrix


class SparseAssemblyTests(unittest.TestCase):
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
